//! `inkling2`: launcher for the Inkling 2 Mini MLX runtime.
//!
//!   inkling2 serve [--port 8000]   local OpenAI-compatible server
//!   inkling2 chat [chat.py args]   the interactive terminal chat
//!
//! The server starts one model worker (`inkling2_mini.server_worker` in the project's
//! .venv) and exposes POST /v1/chat/completions, POST /v1/responses,
//! GET|DELETE /v1/responses/{id}, GET /v1/models and GET /health.

mod api;
mod chat;
mod responses;
mod worker;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Args, Parser, Subcommand};
use serde_json::json;
use tower_http::cors::CorsLayer;

use api::{check_auth, now, AppState, Defaults, ResponseStore, St};
use worker::{Worker, WorkerConfig};

#[derive(Parser)]
#[command(name = "inkling2", version, about = "Launcher and local OpenAI-compatible server for Inkling 2 Mini (MLX)",
          args_conflicts_with_subcommands = true)]
struct Cli {
    /// Project directory (contains inkling2_mini/, .venv/ and models/). Default: found from the binary's location.
    #[arg(long, global = true, env = "INKLING2_HOME")]
    root: Option<PathBuf>,
    /// Python interpreter. Default: <root>/.venv/bin/python
    #[arg(long, global = true, env = "INKLING2_PYTHON")]
    python: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// `inkling2 [serve options]` without a subcommand serves
    #[command(flatten)]
    serve: ServeArgs,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the local OpenAI-compatible server (the default)
    Serve(ServeArgs),
    /// Interactive terminal chat (runs chat.py; arguments are passed through)
    Chat {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

#[derive(Args, Clone)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1", env = "INKLING2_HOST")]
    host: String,
    #[arg(long, default_value_t = 8000, env = "INKLING2_PORT")]
    port: u16,
    /// Model directory. Default: <root>/models/Inkling-2-Mini
    #[arg(long, env = "INKLING2_MODEL")]
    model: Option<PathBuf>,
    /// Model id reported by /v1/models (requests may use any model name)
    #[arg(long, default_value = "inkling-2-mini")]
    model_id: String,
    /// Attention projections + lm_head precision (6, 8 or 16)
    #[arg(long, default_value_t = 6)]
    dense_bits: u32,
    /// Full-attention KV cache precision (8, 4 or 16)
    #[arg(long, default_value_t = 8)]
    kv_bits: u32,
    /// Disable DSpark speculative decoding
    #[arg(long)]
    no_spec: bool,
    /// Loop passes (default: the checkpoint's, 3). Variable-loop checkpoints
    /// (TML-d24-varloop-research) take 1 to 5; requests can override it with "passes"
    #[arg(long)]
    passes: Option<u32>,
    /// Requests decoding at once (continuous batching; 1 = one at a time)
    #[arg(long, default_value_t = 16)]
    max_batch: u32,
    /// Memory for the running batch's KV caches, GB (requests wait beyond it)
    #[arg(long, default_value_t = 6.0)]
    batch_gb: f64,
    /// Longest prompt + reply in tokens
    #[arg(long, default_value_t = 65536)]
    max_context: u64,
    /// Memory for stored conversation states (KV reuse across requests), GB
    #[arg(long, default_value_t = 4.0)]
    cache_gb: f64,
    /// Default reasoning effort when a request sets none: none, minimal, low, medium, high, xhigh, max
    #[arg(long, default_value = "low")]
    effort: String,
    /// Default max output tokens when a request sets none
    #[arg(long, default_value_t = 8192)]
    max_tokens: u64,
    /// Default temperature when a request sets none
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    /// Require `Authorization: Bearer <key>`
    #[arg(long, env = "INKLING2_API_KEY")]
    api_key: Option<String>,
    /// Stored Responses API responses (for GET and previous_response_id)
    #[arg(long, default_value_t = 1000)]
    max_stored_responses: usize,
}

fn is_root(p: &Path) -> bool {
    p.join("inkling2_mini").join("server_worker.py").is_file()
}

fn find_root(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(r) = explicit {
        if is_root(&r) {
            return Ok(r);
        }
        bail!("{} does not contain inkling2_mini/server_worker.py", r.display());
    }
    let mut starts = vec![];
    if let Ok(exe) = std::env::current_exe() {
        starts.push(exe.canonicalize().unwrap_or(exe));
    }
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    for s in starts {
        for dir in s.ancestors() {
            if is_root(dir) {
                return Ok(dir.to_path_buf());
            }
        }
    }
    bail!("cannot find the project directory; pass --root or set INKLING2_HOME")
}

fn python_for(root: &Path, explicit: Option<PathBuf>) -> Result<PathBuf> {
    let p = explicit.unwrap_or_else(|| root.join(".venv").join("bin").join("python"));
    if !p.exists() {
        bail!("Python not found at {} (create the venv as in README.md, or pass --python)", p.display());
    }
    Ok(p)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = find_root(cli.root)?;
    let python = python_for(&root, cli.python)?;
    match cli.cmd.unwrap_or(Cmd::Serve(cli.serve)) {
        Cmd::Chat { args } => {
            let status = tokio::process::Command::new(&python)
                .arg(root.join("chat.py"))
                .args(args)
                .current_dir(&root)
                .status()
                .await?;
            std::process::exit(status.code().unwrap_or(1));
        }
        Cmd::Serve(a) => serve(root, python, a).await,
    }
}

async fn serve(root: PathBuf, python: PathBuf, a: ServeArgs) -> Result<()> {
    let model = a.model.clone().unwrap_or_else(|| root.join("models").join("Inkling-2-Mini"));
    if !model.join("config.json").is_file() {
        bail!("no model at {} (pass --model)", model.display());
    }
    let mut args = vec![
        "--model".to_string(), model.display().to_string(),
        "--dense-bits".into(), a.dense_bits.to_string(),
        "--kv-bits".into(), a.kv_bits.to_string(),
        "--max-context".into(), a.max_context.to_string(),
        "--cache-gb".into(), a.cache_gb.to_string(),
        "--max-batch".into(), a.max_batch.to_string(),
        "--batch-gb".into(), a.batch_gb.to_string(),
    ];
    if a.no_spec {
        args.push("--no-spec".into());
    }
    if let Some(p) = a.passes {
        args.extend(["--passes".into(), p.to_string()]);
    }
    let addr = format!("{}:{}", a.host, a.port);
    // Bind first, so a busy port fails before the model loads.
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("binding {addr}"))?;
    eprintln!("[inkling2] loading {} ...", model.display());
    let t0 = std::time::Instant::now();
    let worker = Worker::start(WorkerConfig { python, root: root.clone(), args }).await?;
    let info = worker.ready_info().await.unwrap_or_default();

    let st: St = Arc::new(AppState {
        worker: worker.clone(),
        model_id: a.model_id.clone(),
        api_key: a.api_key.clone(),
        defaults: Defaults {
            effort: Some(a.effort.clone()).filter(|e| !e.is_empty()),
            max_tokens: a.max_tokens,
            temperature: a.temperature,
        },
        responses: Mutex::new(ResponseStore::new(a.max_stored_responses)),
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat::completions))
        .route("/v1/responses", post(responses::create))
        .route("/v1/responses/{id}", get(responses::get).delete(responses::delete))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(models))
        .route("/health", get(health))
        .route("/", get(health))
        .layer(CorsLayer::permissive())
        .with_state(st);

    let base = format!("http://{addr}/v1");
    eprintln!(
        "[inkling2] ready in {:.0} s: {base}  (model \"{}\", DSpark {}, KV {}-bit, dense {}-bit, {} pass{}{}, batch up to {})\n\
         [inkling2]   POST /v1/chat/completions   POST /v1/responses   GET /v1/models\n\
         [inkling2]   clients: OPENAI_BASE_URL={base}  OPENAI_API_KEY={}",
        t0.elapsed().as_secs_f64(),
        a.model_id,
        if info["speculative"].as_bool().unwrap_or(false) { "on" } else { "off" },
        info["kv_bits"],
        a.dense_bits,
        info["passes"].as_u64().unwrap_or(3),
        if info["passes"].as_u64().unwrap_or(3) == 1 { "" } else { "es" },
        if info["varloop"].as_bool().unwrap_or(false) { ", variable-loop" } else { "" },
        a.max_batch,
        a.api_key.as_deref().unwrap_or("local"),
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("\n[inkling2] shutting down");
        })
        .await?;
    worker.shutdown().await;
    Ok(())
}

async fn models(State(st): State<St>, headers: HeaderMap) -> Response {
    if let Err(e) = check_auth(&st, &headers) {
        return e.into_response();
    }
    let m = json!({"id": st.model_id, "object": "model", "created": now(), "owned_by": "local"});
    Json(json!({"object": "list", "data": [m]})).into_response()
}

async fn health(State(st): State<St>) -> Response {
    let info = st.worker.ready_info().await;
    let status = if info.is_some() { "ok" } else { "restarting" };
    Json(json!({"status": status, "model": st.model_id, "worker": info})).into_response()
}
