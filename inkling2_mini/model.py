"""MLX implementation of Inkling 2 Mini (`inkling2_loop`).

Inkling 2 Mini is the looped variant of Inkling 2: a 24-layer hybrid
(sliding / full attention) MoE decoder whose middle layers [core_start,
core_end) run `num_passes` times with shared weights. Passes after the first
restart from the pre-loop state and receive "messages" from the previous pass,
which are mixed into the query / key / value / MLP inputs of each loop layer
through small forget / gain gates.

The per-layer building blocks (relative-position-biased attention with sinks,
short convolutions, block attention residuals, softcapped SwiGLU MoE, the
vision tower) are identical to Inkling 2 Small, so they are reused from
mlx-vlm's `inkling2` port. Everything specific to the loop lives here.

Reference: transformers 5.24 `models/inkling2_loop/modular_inkling2_loop.py`.
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from mlx_vlm.models.inkling.inkling import masked_scatter
from mlx_vlm.models.inkling.language import _cache_padding_mask
from mlx_vlm.models.inkling2.config import TextConfig as Inkling2TextConfig
from mlx_vlm.models.inkling2.config import VisionConfig
from mlx_vlm.models.inkling2.language import (
    AttentionResidual,
    DecoderLayer,
    SparseMoE,
    _attention_residual,
    _combine_routes,
    _route,
)
from mlx_vlm.models.inkling2.vision import VisionModel
from mlx_vlm.models.switch_layers import SwitchLinear, _gather_sort, _scatter_unsort

from . import kernels
from . import decode_kernels
from . import moe_kernels
from . import prefill_kernels
from .caches import QuantSinkKVCache, SinkKVCache, SinkRingKVCache
from .fused import FusedAttention, gates, stack_gates


class GroupedSparseMoE(SparseMoE):
    """Inkling 2 MoE with a fused gate/up projection and grouped multi-token routes.

    * gate and up come from one checkpoint tensor (w13) with one NVFP4 global scale
      per expert, so they stay one [gate; up] matrix: one gather-matmul per visit
      instead of two.
    * mlx-vlm sends 2-8 token blocks through per-route kernels that keep speculative
      verification bit-identical to single-token decode, but they read each route's
      expert separately. Acceptance here compares against the target's own samples,
      so routes are sorted by expert instead and gather_qmm reads each distinct expert
      once (L=8 verify: 169 ms -> 128 ms on an M6).
    """

    def __init__(self, config):
        super().__init__(config)
        sw = self.switch_mlp
        delattr(sw, "gate_proj")
        delattr(sw, "up_proj")
        sw.gate_up = SwitchLinear(config.hidden_size, 2 * config.intermediate_size,
                                  config.n_routed_experts, bias=False)
        self.inter = config.intermediate_size

    def _moe_decode_ok(self):
        ok = getattr(self, "_decode_ok", None)
        if ok is None:
            act = self.switch_mlp.activation
            self._caps = (act.gate_cap, act.up_cap)
            ok = self._decode_ok = decode_kernels.supports_moe_decode(self.switch_mlp)
        return ok

    def _route(self, x, tokens):
        if tokens <= 64 and decode_kernels.active() and moe_kernels.route_ok(self, x):
            # the same router GEMV + top-k as the fused single-token kernels, so verification
            # passes route each token exactly like decode does
            return moe_kernels.route(x, self.gate_weight, self.e_score_correction_bias, self.top_k, self.route_scale)
        if tokens <= 64 and decode_kernels.active():
            wt = getattr(self, "_gate_t32", None)
            if wt is None or wt.shape[0] != x.shape[-1]:
                wt = self._gate_t32 = mx.contiguous(self.gate_weight.astype(mx.float32).T)
            # FP32 GEMV + one kernel for sigmoid / bias / top-k / normalization
            return decode_kernels.route(x, self.gate_weight, self.e_score_correction_bias,
                                        self.top_k, self.route_scale, wt)
        return _route(x, self.gate_weight, self.e_score_correction_bias, self.top_k, self.route_scale)

    def __call__(self, x):
        sw = self.switch_mlp
        tokens = x.shape[0] * x.shape[1]
        indices, weights = self._route(x, tokens)
        routes = tokens * self.top_k
        if tokens == 1 and decode_kernels.active() and self._moe_decode_ok():
            # single-token decode: gate_up + SwiGLU, then down + weighted sum (2 kernels)
            return decode_kernels.moe_decode(x, sw, indices, weights, self.inter,
                                             self._caps[0], self._caps[1])
        if (
            GROUPED_ROUTES[0] <= routes <= GROUPED_ROUTES[1]
            and kernels.use_kernels()
            and kernels.supports_grouped_nvfp4(sw.gate_up)
            and kernels.supports_grouped_nvfp4(sw.down_proj)
        ):
            # Batched decode / speculative verify: read each distinct expert once
            # per 4 routes instead of once per route (see kernels.grouped_nvfp4_qmv).
            flat = indices.reshape(-1).astype(mx.int32)
            order = mx.argsort(flat)
            ids = flat[order]
            rows = x.reshape(tokens, -1)[order // self.top_k]
            gate_up = kernels.grouped_nvfp4_qmv(rows, sw.gate_up, ids)
            act = sw.activation(gate_up[:, self.inter :], gate_up[:, : self.inter])
            out = kernels.grouped_nvfp4_qmv(act, sw.down_proj, ids)
            if prefill_kernels.use_kernels():
                comb = prefill_kernels.combine_routes(out, mx.argsort(order), weights.reshape(tokens, -1))
                return comb.reshape(*x.shape)
            out = out[mx.argsort(order)].reshape(*indices.shape, -1)
            return _combine_routes(out, weights)
        xe = mx.expand_dims(x, (-2, -3))
        grouped = tokens > 1
        if grouped:
            xe, idx, inverse = _gather_sort(xe, indices)
        else:
            idx = indices
        gate_up = sw.gate_up(xe, idx, sorted_indices=grouped)
        gate, up = gate_up[..., : self.inter], gate_up[..., self.inter :]
        out = sw.down_proj(sw.activation(up, gate), idx, sorted_indices=grouped)
        if prefill_kernels.use_kernels():
            # un-sort + weighted top-k sum in one pass (prefill_kernels.combine_routes); single
            # tokens use it too, so decode and verification sum the routes in the same order
            if not grouped:
                inverse = mx.arange(out.size // out.shape[-1], dtype=mx.uint32)
            comb = prefill_kernels.combine_routes(out.reshape(-1, out.shape[-1]), inverse, weights.reshape(tokens, -1))
            return comb.reshape(*x.shape)
        if grouped:
            out = _scatter_unsort(out, inverse, indices.shape)
        return _combine_routes(out.squeeze(-2), weights)


# Route counts (tokens x top-8) where the grouped expert kernel beats gather_qmm on
# an M6: below ~32 routes gather_qmm's vector kernel is faster; from 512 routes
# gather_qmm switches to its own grouped matrix kernel.
GROUPED_ROUTES = (32, 511)


@dataclass
class TextConfig(Inkling2TextConfig):
    model_type: str = "inkling2_loop_text"
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    swa_num_attention_heads: int = 24
    log_scaling_n_floor: Optional[int] = 8192
    intermediate_size: int = 1152
    n_routed_experts: int = 128
    loop_config: dict = field(default_factory=dict)

    @property
    def loop_start(self) -> int:
        return self.loop_config.get("core_start", 6)

    @property
    def loop_end(self) -> int:
        return self.loop_config.get("core_end", 18)

    @property
    def num_passes(self) -> int:
        return self.loop_config.get("num_passes", 3)

    @property
    def gate_rank(self) -> int:
        return self.loop_config.get("gate_rank", 64)

    @property
    def num_loop_layers(self) -> int:
        return self.loop_end - self.loop_start

    @property
    def num_cache_layers(self) -> int:
        """One KV + conv cache per layer *application* (48 for the release)."""
        return self.num_hidden_layers + (self.num_passes - 1) * self.num_loop_layers


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    image_token_id: int = 200054
    eos_token_id: int = 200006

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        text = TextConfig.from_dict(d.get("text_config", {}))
        if text.loop_config.get("is_varloop"):
            raise NotImplementedError("variable-loop checkpoints are not supported")
        vision = VisionConfig.from_dict(d.get("vision_config", {}))
        vision.text_hidden_size = text.hidden_size
        vision.out_hidden_size = text.hidden_size
        return cls(
            text_config=text,
            vision_config=vision,
            image_token_id=d.get("image_token_id", 200054),
            eos_token_id=d.get("eos_token_id", 200006),
        )


def _unweighted_rms_norm(x, eps):
    return mx.fast.rms_norm(x, None, eps)


class MessageGate(nn.Module):
    """`(1 - tanh(F z)) * h + tanh(G z) * m` with `z = D [rms(h), rms(m)]`.

    Checkpoint names: controllers.{attn,k,v,mlp}.{down,forget,gain}_weight.
    The down projection runs in BF16 (as stored); the gates run in FP32, as in
    the reference implementation.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.down_weight = mx.zeros((config.gate_rank, 2 * config.hidden_size))
        self.forget_weight = mx.zeros((config.hidden_size, config.gate_rank))
        self.gain_weight = mx.zeros((config.hidden_size, config.gate_rank))

    def __call__(self, h, message):
        h = h.astype(mx.float32)
        message = message.astype(mx.float32)
        z = mx.concatenate(
            [_unweighted_rms_norm(h, self.eps), _unweighted_rms_norm(message, self.eps)],
            axis=-1,
        )
        z = (z.astype(self.down_weight.dtype) @ self.down_weight.T).astype(mx.float32)
        forget = mx.tanh(z @ self.forget_weight.T)
        gain = mx.tanh(z @ self.gain_weight.T)
        return (1 - forget) * h + gain * message


def _attnres(states, query, norm_weight=None, write_mixed=True, out_dtype=mx.bfloat16):
    """Block attention residual (+ optional RMSNorm): fused kernel on the GPU,
    the unfused reference elsewhere."""
    if kernels.use_kernels():
        return kernels.attention_residual(states, query, norm_weight, write_mixed, out_dtype=out_dtype)
    return kernels.attention_residual_reference(states, query, norm_weight, out_dtype=out_dtype)


def _rms_weight(x, weight, eps):
    return mx.fast.rms_norm(x, weight, eps)


class Layer(DecoderLayer):
    """One Inkling 2 decoder layer (pre/post-loop, or a loop layer on pass 1).

    Same computation as mlx-vlm's DecoderLayer, restructured for decode latency:
    the attention residual and the RMSNorm after it run as one kernel, and the
    attention projections are fused (see fused.py) once `fuse()` has been called.
    """

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.eps = config.rms_norm_eps
        self.mlp = GroupedSparseMoE(config)
        self.self_attn = FusedAttention(config, layer_idx)
        prefill_kernels.speed_up_convs(self.attn_sconv, self.mlp_sconv)

    def fuse(self):
        self.self_attn.fuse()
        if self.is_sliding:
            self._qkv_norm = None
        else:
            self._qkv_norm = mx.stack(
                [self.q_input_layernorm.weight, self.k_input_layernorm.weight, self.v_input_layernorm.weight]
            )
            mx.eval(self._qkv_norm)

    def _qkv_weight(self):
        if self.is_sliding:
            return self.input_layernorm.weight
        w = getattr(self, "_qkv_norm", None)
        if w is None:
            w = mx.stack([self.q_input_layernorm.weight, self.k_input_layernorm.weight, self.v_input_layernorm.weight])
        return w

    def _attention_inputs(self, x, blocks, messages):
        B, L, H = x.shape
        dtype = x.dtype
        if self.attention_input is None:  # layer 0 (sliding): no residual mix
            h = self.input_layernorm(x).astype(dtype)
            return h, h, h
        if messages is None:
            _, normed = _attnres([*blocks, x], self.attention_input.query, self._qkv_weight(),
                                 write_mixed=False, out_dtype=dtype)
            if self.is_sliding:
                h = normed[..., 0, :]
                return h, h, h
            return normed[..., 0, :], normed[..., 1, :], normed[..., 2, :]
        mixed, _ = _attnres([*blocks, x], self.attention_input.query)
        n = B * L
        if prefill_kernels.use_kernels():  # fused gate kernels (prefill_kernels.message_gates)
            down, fg = self._gates3
            normed = prefill_kernels.message_gates(
                mixed.reshape(n, -1, H), messages.reshape(n, 4, H), 0, 3, down, fg, self._qkv_weight()
            ).reshape(3, B, L, H)
            return normed[0], normed[1], normed[2]
        if self.is_sliding:
            h3 = mx.broadcast_to(mixed.reshape(1, n, H), (3, n, H))
        else:
            h3 = mixed.reshape(n, 3, H).transpose(1, 0, 2)
        m3 = messages[..., :3, :].reshape(n, 3, H).transpose(1, 0, 2).astype(mx.float32)
        down, fg = self._gates3
        g3 = gates(h3, m3, down, fg)
        if self.is_sliding:
            normed = _rms_weight(g3, self.input_layernorm.weight, self.eps)
        else:
            normed = mx.fast.rms_norm(g3, None, self.eps) * self._qkv_weight()[:, None, :].astype(mx.float32)
        normed = normed.astype(dtype).reshape(3, B, L, H)
        return normed[0], normed[1], normed[2]

    def __call__(self, x, blocks, cache=None, messages=None):
        dtype = x.dtype
        conv = cache[1] if cache is not None else None
        mask = _cache_padding_mask(conv, x.shape[1])
        q_in, k_in, v_in = self._attention_inputs(x, blocks, messages)
        attn = self.self_attn(q_in, k_in, v_in, cache=cache, conv_mask=mask)
        if self.is_block_boundary:
            residual = self.attn_sconv(attn, conv, mask, out_dtype=mx.float32)
            blocks.append(x)
        else:  # conv + residual add in one pass on the prefill kernel path
            residual = self.attn_sconv(attn, conv, mask, residual=x, out_dtype=mx.float32)
        if messages is None:
            _, hidden = _attnres([*blocks, residual], self.mlp_input.query,
                                 self.post_attention_layernorm.weight, write_mixed=False, out_dtype=dtype)
            hidden = hidden[..., 0, :]
        else:
            mixed, _ = _attnres([*blocks, residual], self.mlp_input.query)
            B, L, H = x.shape
            down, fg = self._gate_mlp
            if prefill_kernels.use_kernels():
                hidden = prefill_kernels.message_gates(
                    mixed.reshape(B * L, 1, H), messages.reshape(B * L, 4, H), 3, 1, down, fg,
                    self.post_attention_layernorm.weight,
                )[0].reshape(B, L, H)
                return self._mlp_tail(x, residual, hidden, conv, mask, dtype)
            h = gates(mixed.reshape(1, B * L, H), messages[..., 3, :].reshape(1, B * L, H).astype(mx.float32),
                      down, fg)
            hidden = self.post_attention_layernorm(h.reshape(B, L, H)).astype(dtype)
        return self._mlp_tail(x, residual, hidden, conv, mask, dtype)

    def _mlp_tail(self, x, residual, hidden, conv, mask, dtype):
        sc = self.mlp_sconv
        if (sc.kernel_size == 4 and moe_kernels.fused_ok(self.mlp, hidden)
                and prefill_kernels.conv_fast_ok(hidden, conv, mask)):
            # one decode token: route, experts, weighted sum, short conv and residual in three
            # kernels (moe_kernels.moe_decode_conv)
            state = conv[sc.conv_idx] if conv is not None else None
            if state is None:
                state = mx.zeros((1, 3, hidden.shape[-1]), mx.float32)
            out, nstate = moe_kernels.moe_decode_conv(hidden, self.mlp, state, sc.conv.weight, residual, dtype)
            if conv is not None:
                conv[sc.conv_idx] = nstate
        else:
            out = sc(self.mlp(hidden), conv, mask, residual=residual, out_dtype=dtype)
        if conv is not None:
            conv.advance(x.shape[1])
        return out


class LoopDecoderLayer(Layer):
    """A decoder layer inside the loop; identical to `Layer` on the first pass, and
    gated by the previous pass's messages on later passes."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_message_gate = MessageGate(config)
        self.k_message_gate = MessageGate(config)
        self.v_message_gate = MessageGate(config)
        self.mlp_message_gate = MessageGate(config)

    def fuse(self):
        super().fuse()
        self._stack_gates()

    def _stack_gates(self):
        self._gates3 = stack_gates([self.q_message_gate, self.k_message_gate, self.v_message_gate])
        self._gate_mlp = stack_gates([self.mlp_message_gate])
        mx.eval(self._gates3, self._gate_mlp)

    def __call__(self, x, blocks, cache=None, messages=None):
        if messages is not None and not hasattr(self, "_gates3"):
            self._stack_gates()
        return super().__call__(x, blocks, cache, messages)


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            (
                LoopDecoderLayer(config, i)
                if config.loop_start <= i < config.loop_end
                else Layer(config, i)
            )
            for i in range(config.num_hidden_layers)
        ]
        self.final_input = AttentionResidual(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # (num_loop_layers, 4, hidden): query / key / value / mlp message queries
        self.message_query = mx.zeros((config.num_loop_layers, 4, config.hidden_size))

    def embed(self, input_ids):
        return self.embed_norm(self.embed_tokens(input_ids))

    def fuse(self):
        for layer in self.layers:
            layer.fuse()

    def __call__(self, input_ids=None, cache=None, input_embeddings=None, capture=None):
        """capture: optional *visit* indices (0..num_cache_layers). Visit k is the
        k-th layer application in execution order (pre-loop, pass 1, pass 2, ...,
        post-loop); its capture is the hidden state entering it. The last index is
        the final normalized state. Returns (normed, [captures]) when given."""
        cfg = self.config
        x = self.embed(input_ids) if input_embeddings is None else input_embeddings
        caches = cache if cache is not None else [None] * cfg.num_cache_layers
        start, end = cfg.loop_start, cfg.loop_end
        wanted = set(capture or ())
        captured = {}
        visit = 0
        blocks = []

        def run(i, x, blocks, c, messages=None):
            nonlocal visit
            if visit in wanted:
                captured[visit] = x
            visit += 1
            if messages is None:
                return self.layers[i](x, blocks, c)
            return self.layers[i](x, blocks, c, messages=messages)

        for i in range(start):
            x = run(i, x, blocks, caches[i])

        pre_x, pre_blocks = x, list(blocks)
        num_pre_blocks = len(pre_blocks)

        for i in range(start, end):
            x = run(i, x, blocks, caches[i])

        for p in range(1, cfg.num_passes):
            # Skip the block appended at the loop entry: it is pre_x on every pass.
            blocks_prev, x_prev = blocks[num_pre_blocks + 1 :], x
            x, blocks = pre_x, list(pre_blocks)
            offset = cfg.num_hidden_layers + (p - 1) * cfg.num_loop_layers
            # Every loop layer of this pass reads the same previous-pass states, so
            # one attention residual with all (layers x 4) queries replaces 12 calls
            # (decode: ~15 ms -> ~1 ms per token). Long prefill chunks keep per-layer
            # calls to bound the (tokens x 48 x hidden) FP32 intermediate.
            batched = x.shape[0] * x.shape[1] <= 64
            if batched:
                queries = self.message_query.reshape(-1, cfg.hidden_size)
                prev = mx.stack([*blocks_prev, x_prev], axis=-2)
                all_messages = _attention_residual(prev, queries, True)
            prev_states = [*blocks_prev, x_prev]
            for j, i in enumerate(range(start, end)):
                if batched:
                    messages = all_messages[..., 4 * j : 4 * j + 4, :]
                else:  # fused attention-residual kernel, reading the states in place
                    messages = _attnres(prev_states, self.message_query[j])[0]
                x = run(i, x, blocks, caches[offset + j], messages)

        for i in range(end, cfg.num_hidden_layers):
            x = run(i, x, blocks, caches[i])

        _, out = _attnres([*blocks, x], self.final_input.query, self.norm.weight,
                          write_mixed=False, out_dtype=x.dtype)
        out = out[..., 0, :]
        if capture is None:
            return out
        captured[visit] = out
        return out, [captured[k] for k in capture]


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def logits(self, h):
        logits = self.lm_head(h).astype(mx.float32)
        cap = self.config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits[..., : self.config.unpadded_vocab_size]

    def __call__(self, input_ids=None, cache=None, input_embeddings=None, last_only=True):
        h = self.model(input_ids, cache=cache, input_embeddings=input_embeddings)
        if last_only:
            h = h[:, -1:]
        return self.logits(h)

    def make_cache(self, kv_bits=None):
        """kv_bits: quantize the full-attention visits' K/V (8 or 4); None = BF16.
        Sliding-window visits are capped at 1024 tokens and stay BF16."""
        cfg = self.config
        caches = []
        for app in range(cfg.num_cache_layers):
            layer = (
                app
                if app < cfg.num_hidden_layers
                else cfg.loop_start + (app - cfg.num_hidden_layers) % cfg.num_loop_layers
            )
            if cfg.layer_is_sliding(layer):
                kv = SinkRingKVCache(cfg.sliding_window_size)
            elif kv_bits:
                kv = QuantSinkKVCache(bits=kv_bits)
            else:
                kv = SinkKVCache()
            caches.append(CacheList(kv, ArraysCache(4)))
        return caches


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)

    def embed(self, input_ids, pixel_values=None, image_grid_thw=None):
        h = self.language_model.model.embed(input_ids)
        if pixel_values is not None:
            features = self.vision_tower(pixel_values, image_grid_thw=image_grid_thw)
            features = features.astype(h.dtype)
            slots = input_ids == self.config.image_token_id
            if int(slots.sum()) != features.shape[0]:
                raise ValueError(
                    f"{int(slots.sum())} image tokens but {features.shape[0]} image features"
                )
            h = masked_scatter(h, mx.broadcast_to(slots[..., None], h.shape), features)
        return h

    def make_cache(self, kv_bits=None):
        return self.language_model.make_cache(kv_bits)
