# 本项目与 Strata 的对比（Inkling 2 Mini）

两者都在本地跑 Thinking Machines 的 Inkling 2 Mini，也都提供 OpenAI 兼容接口。Strata 指 [Niko1221/Strata](https://github.com/Niko1221/Strata)，核对的是 2026-11-23 的代码。下文关于 Strata 的说法都标了它仓库里的出处。

## 设计

| | 本项目 | Strata |
| --- | --- | --- |
| 范围 | 只支持 Inkling 2 Mini 和 Apple Silicon | 多个模型（Qwen3.8-Flash-Next、Qwen3.6-35B-A3B、Hy4-Flash、Inkling 2 Mini）。后端有 CUDA、HIP、SYCL、Metal、CPU（`README.md`） |
| 实现 | Python + MLX，加上一些自定义 Metal kernel；启动器和 HTTP 服务用 Rust | C++ 引擎 `strata-inkling`。Inkling 有单独的 Metal 后端（`engine/src/inkling2/metal/`）。服务端用 Python（`terminal/serve/`），另有桌面 app |
| 权重 | 直接读官方 safetensors。专家保持训练时的 NVFP4 / MXFP8；attention 投影和 lm_head 用仿射 6-bit（可选 8 / 16） | 自家 `.strata` 包：SQ-20 为 19.3 GB，"near-lossless native SF4+MX8"；SQ-15 为 14.3 GB；SQ-10 为 10.5 GB（`docs/MODELS.md`）。也支持 Unsloth GGUF，或直接读官方 safetensors：NVFP4 转 SF4、MXFP8 转 MX8，主干保持 BF16（`ink_safetensors_loader.h`） |
| 内存 | 全部常驻统一内存，加载后 18.91 GB | 专家按显存 / 内存 / SSD 分层存放（`docs/HOW_IT_WORKS.md`）。在 Mac 上要么常驻，要么从 SSD 读 |
| 投机解码 | 用自带的 DSpark，按线性链验证。每轮按草稿自身的置信度选验证长度，不划算就退回普通解码；草稿层为 8-bit | 用自带的 DSpark，做树状验证。按一个校准过的验证耗时模型选长度；草稿的 Markov 步只对前 512 个候选打分（`ink_metal_runner.h`、`ink_metal_dspark.metal`） |
| 并发 | 服务端一次处理一个请求，另有离线 batch API | `--parallel N` 个序列槽（`ink_metal_main.cpp`） |
| 接口 | `/v1/chat/completions`；`/v1/responses`，支持 `previous_response_id` 和 GET / DELETE；`/v1/models` | `/v1/chat/completions`、`/v1/messages`（Anthropic 格式）、`/v1/responses`、`/v1/models`，另有图片等接口（`terminal/serve/server.py`） |
| Responses | 有状态：服务端保存对话和 KV，后续轮只预填新输入 | 无状态：什么都不存，遇到 `previous_response_id` 返回 unsupported，客户端每次都要重发完整输入（`terminal/serve/responses.py`） |
| 跨请求 KV 复用 | 多条目 LRU，默认 4 GB，按渲染后的前缀匹配。条目用时克隆，所以重试和分支不会挤掉原对话。较长的 system + 工具前缀会单独存快照 | 依靠引擎的对话前缀缓存。客户端回传 reasoning 内容后，提示词与模型实际写过的内容一致，缓存才能命中（`responses.py` 注释提到 "one prefix cache"） |
| 客户端适配 | 通用 OpenAI 格式 | 有大量 Codex 专用处理：namespace / custom 工具、tool_search、上下文压缩、线程标题（`responses.py`） |
| 其他 | 没有图形界面 | 桌面 app、终端 TUI、监控、MCP、多 GPU |

## 同机实测（2026-11-23）

- 测试机：Mac mini M6，12 核 GPU，32 GB 内存。两边测试时都没有其他模型进程在跑。
- Strata：commit `73d4f41`，即 strata-inkling 0.1.66。用它自带的 CMake 从源码编译统一引擎，Metal 后端，加载的是本项目用的同一份官方 safetensors。
- Strata 的专用包（SQ-20 / 15 / 10）要从 Hugging Face 下载，这台机器连不上，所以没有测。
- 提示词：两边喂完全相同的 token id，都是贪心解码。

### Strata 原版为什么输出乱码

原版 Strata 在 Metal 上读官方 safetensors，输出是乱码。用 CPU FP32 真值评估（固定的中英文文本，437 个位置）：平均 KL 10.87，top-1 0.23%，困惑度 232,675。原因有三处，都只出现在"Metal 后端 + 官方 safetensors"这条路径上：

1. **BF16 小张量被当成 FP32 读。** 官方 checkpoint 把各种范数权重、`q/rel_l2_clip.scale`、短卷积系数（`*_sconv.weight`、`static_w`）、`rel_logits_proj`、循环控制器的 `gain/forget_weight` 都存成 BF16。Metal 内核和加载时的预处理却按 `float*` 读取它们（例如 `ink_embed_bf16` 的 `device const float* norm_w`），而 `.strata` 包里这些张量本来就是 FP32。safetensors 加载器做的是"零转换"，把 BF16 原样交了过去。结果每两个 BF16 被拼成一个 FP32 值，后半段还会读越界。从 embedding 后的第一个 RMSNorm 开始，结果就是错的：实测 Strata 用的权重恰好等于 BF16 字节按 float32 解读的值。
   - CPU 后端按 dtype 读取，所以同一份权重在 CPU 上是对的：KL 0.0022，top-1 97.5%。
   - 从机制看这是字节解释错误，与芯片无关，不是 M6 特有的。不过其他芯片没有实测。
2. **DSpark 草稿张量命名不匹配。** 加载器按 `mtp.safetensors` 里的原名登记，Metal 后端却按 `.strata` 的 `dspark.` 前缀查找，于是报 `missing dspark tensor`，草稿模型加载不了。
3. **`--kld-eval` 工具在 Metal 上超过 64 个 token 就失效。** Metal 一次前向最多只算 64 行 logits（`b_logits` 只分配了 64 行），工具却一次性请求全部位置，第 64 个位置以后全是空值。这只影响评估，不影响生成。

用逐层导出定位：我在调试版里导出每次层访问的残差流，和本项目的结果对照，看哪一步开始偏离。修复方式是本地补丁（[`strata-metal-safetensors-fix.diff`](strata-metal-safetensors-fix.diff)）：
- 加载时把上述 345 个 BF16 张量转成 FP32；
- 给 DSpark 张量名补上 `dspark.` 前缀；
- 把 kld-eval 改成每 64 个 token 一块。

修复后 Strata 在 Metal 上完全正确。下面的数据都是修复后测的。

### 精度

用同一份 CPU FP32 真值做 teacher-forced 评估，437 个位置，按 Strata `--kld-eval` 的口径计算：

| | 平均 KL | top-1 与真值一致 | 困惑度 |
| --- | --- | --- | --- |
| Strata 原版（Metal） | 10.87 | 0.23 % | 232,675 |
| Strata 修复后（Metal，BF16 权重，FP32 激活） | 9.3e-4 | 99.1 % | 5.11 |
| Strata CPU 后端 | 2.2e-3 | 97.5 % | 5.13 |
| 本项目，BF16 dense | 4.3e-3 | 95.9 % | 5.08 |
| 本项目，8-bit dense | 3.4e-3 | 96.6 % | 5.10 |
| 本项目，6-bit dense（默认） | 8.2e-3 | 95.0 % | 5.13 |

原版那一行也受第 3 点影响，但只看 1 个 token 时 KL 也有 10.4，原版本身确实是错的。修复后的 Strata 比本项目更接近真值：它的激活一直用 FP32，而本项目的 MLX 激活是 BF16。

### 速度

Strata 已经修复，输出正确。

| | Strata（修复后，BF16 主干） | 本项目 BF16 dense | 本项目 6-bit dense（默认） |
| --- | --- | --- | --- |
| 普通解码，短上下文（4 个提示词，300 token） | 21.7–21.8 tok/s | 20.4–20.7 tok/s | 29.9–30.3 tok/s |
| DSpark，散文（story / zh） | 21.2 / 21.0 tok/s | 21.2 / 20.5 tok/s | 30.2 / 29.9 tok/s |
| DSpark，代码（code / code2） | 26.1 / 22.2 tok/s | 29.3 / 20.0 tok/s | 39.3 / 40.1 tok/s |
| prefill 8k / 32k | 718 / 627 tok/s | 967 / 867 tok/s | 879 / 805 tok/s |
| 8k / 32k 上下文时的解码 | 58.6 / 56.7 ms/token | 48.1 / 57.1 ms/token | 35.9 / 38.6 ms/token |
| 加载时间 | 6.4–7.8 s | 5.0 s | 4.7 s |
| 显存 / 内存 | 分配 22.0 GiB（23.6 GB） | 20.3 GB | 18.9 GB |

- **同为 BF16 权重：**
  - Strata 的普通解码快约 6%。
  - 本项目 prefill 快 35%（8k）和 38%（32k），8k 上下文的解码快 22%，32k 时两边相当。
  - DSpark 两边都只在代码上有收益，同一个代码提示词的结果因生成的文本不同而波动很大。
- **本项目默认的 6-bit dense：** 每个 token 读约 3.8 GB，BF16 要读约 6.4 GB。所以普通解码比 Strata 快 38%，代码上的 DSpark 快 50–80%，长上下文解码快 47–63%。代价是精度：KL 8.2e-3，而 Strata 修复后是 9.3e-4。

## 两边各自的公开数据

### 本项目：实测

测试机为 Mac mini M6，12 核 GPU，32 GB 内存，实测带宽约 140 GB/s。配置为 6-bit dense、8-bit KV、8-bit 草稿层。

| 项目 | 结果 |
| --- | --- |
| 普通解码，短上下文 | 31–32 tok/s，约为带宽上限的 90%；每个 token 读约 3.8 GB |
| DSpark，代码（6 个提示词的几何平均） | 36.7 tok/s，比普通解码快 17% |
| DSpark，中英文散文 | 31–33 tok/s，与普通解码相差 −2% 到 +3% |
| 32k 上下文时的解码 | 39.1 ms/token |
| prefill | 8k 时 984 tok/s，32k 时 879 tok/s |
| 经过启动器的解码 | 与直接调运行时相同（同一时段测得 29.5 vs 29.6 tok/s） |
| 8k 对话第二轮的首 token 延迟 | 冷启动 9.25 s；Chat Completions 重发历史 0.32 s；Responses 用 `previous_response_id` 0.29 s |
| 加载后常驻内存 | 18.91 GB |

### Strata：公开资料

- Strata 没有公布 Inkling 2 Mini 在 Mac 上的速度。它在 Mac 上公布的数据都来自 Qwen3.8-Flash-Next（`docs/MAC.md`）。
- 它的调度器里写着一套验证耗时模型的常数（`ink_metal_runner.h`）。这些是常数，不是基准测试：
  - M5 Max 实测标定：单 token 一步 14.4 ms（SQ-10 为 13.0 ms，BF16 主干为 17.0 ms）。每多验证一个 token 约多 4.2 ms，相当于一步的 29%。
  - M4（代码注释写的是 120 GB/s）：耗时为 34 + 1.85 × 涉及的专家数（ms），单 token 约 48.8 ms，每多验证一个 token 约多 28%。
- 同一项指标在本项目里也约为一步的 30%（每个 token 多约 10 ms，单步约 33 ms）。所以两边投机解码的收益结构相同：代码能赚，散文赚不到。
- SQ-15 / SQ-10 每个 token 读的字节更少，在受带宽限制的 Mac 上解码会更快。代价是专家不再是训练时的精度，而 Strata 没有给出这些包的 KL 或准确率数据。本项目的专家保持原生精度，事后量化的只有 dense 部分，实测误差见 README 第 9 条。
