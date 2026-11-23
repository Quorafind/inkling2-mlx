//! The Python model worker (`inkling2_mini.server_worker`): one child process that owns
//! the model and the conversation states, spoken to in JSON lines over stdin / stdout.
//! Events are routed to the waiting request by id. If the worker dies, the requests in
//! flight get an error and the worker is started again.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};

#[derive(Clone, Debug)]
pub struct WorkerConfig {
    pub python: PathBuf,
    pub root: PathBuf,
    pub args: Vec<String>,
}

pub struct Conn {
    stdin: Mutex<ChildStdin>,
    routes: StdMutex<HashMap<String, mpsc::UnboundedSender<Value>>>,
    child: Mutex<Child>,
    pub ready: Value,
}

impl Conn {
    pub async fn send(&self, v: &Value) -> std::io::Result<()> {
        let mut line = serde_json::to_string(v).unwrap_or_default();
        line.push('\n');
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await
    }

    fn unroute(&self, id: &str) {
        self.routes.lock().unwrap().remove(id);
    }
}

pub struct Worker {
    cfg: WorkerConfig,
    conn: RwLock<Option<Arc<Conn>>>,
    stopping: AtomicBool,
}

/// A generation in flight. Dropping it before `finish()` (client gone) cancels it.
pub struct Generation {
    pub id: String,
    pub events: mpsc::UnboundedReceiver<Value>,
    conn: Arc<Conn>,
    armed: bool,
}

impl Generation {
    pub fn finish(&mut self) {
        self.armed = false;
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        if self.armed {
            let (conn, id) = (self.conn.clone(), self.id.clone());
            tokio::spawn(async move {
                let _ = conn.send(&json!({"op": "cancel", "id": id})).await;
                conn.unroute(&id);
            });
        }
    }
}

async fn spawn_once(cfg: &WorkerConfig) -> Result<(Arc<Conn>, oneshot::Receiver<()>)> {
    let mut child = Command::new(&cfg.python)
        .arg("-m")
        .arg("inkling2_mini.server_worker")
        .args(&cfg.args)
        .current_dir(&cfg.root)
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("starting {}", cfg.python.display()))?;
    let stdin = child.stdin.take().context("worker stdin")?;
    let stdout = child.stdout.take().context("worker stdout")?;
    let mut lines = BufReader::new(stdout).lines();
    let ready = loop {
        match lines.next_line().await? {
            None => bail!("the model worker exited while loading (see the log above)"),
            Some(line) => {
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    match v["type"].as_str() {
                        Some("ready") => break v,
                        Some("fatal") => bail!("model worker: {}", v["message"].as_str().unwrap_or("?")),
                        _ => {}
                    }
                }
            }
        }
    };
    let conn = Arc::new(Conn {
        stdin: Mutex::new(stdin),
        routes: StdMutex::new(HashMap::new()),
        child: Mutex::new(child),
        ready,
    });
    let (done_tx, done_rx) = oneshot::channel();
    let reader = conn.clone();
    tokio::spawn(async move {
        while let Ok(Some(line)) = lines.next_line().await {
            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
            let id = v["id"].as_str().unwrap_or_default().to_string();
            let last = matches!(v["type"].as_str(), Some("done") | Some("error"));
            let mut routes = reader.routes.lock().unwrap();
            if let Some(tx) = routes.get(&id) {
                let _ = tx.send(v);
            }
            if last {
                routes.remove(&id);
            }
        }
        let pending: Vec<_> = reader.routes.lock().unwrap().drain().collect();
        for (id, tx) in pending {
            let _ = tx.send(json!({"type": "error", "id": id, "message": "the model worker exited"}));
        }
        let _ = done_tx.send(());
    });
    Ok((conn, done_rx))
}

impl Worker {
    pub async fn start(cfg: WorkerConfig) -> Result<Arc<Worker>> {
        let (conn, done) = spawn_once(&cfg).await?;
        let w = Arc::new(Worker { cfg, conn: RwLock::new(Some(conn)), stopping: AtomicBool::new(false) });
        tokio::spawn(w.clone().supervise(done));
        Ok(w)
    }

    async fn supervise(self: Arc<Self>, mut done: oneshot::Receiver<()>) {
        loop {
            let _ = (&mut done).await;
            *self.conn.write().await = None;
            if self.stopping.load(Ordering::SeqCst) {
                return;
            }
            eprintln!("[inkling2] the model worker exited; starting it again");
            loop {
                match spawn_once(&self.cfg).await {
                    Ok((conn, d)) => {
                        *self.conn.write().await = Some(conn);
                        done = d;
                        eprintln!("[inkling2] model worker ready again");
                        break;
                    }
                    Err(e) => {
                        eprintln!("[inkling2] restart failed: {e:#}; retrying in 5 s");
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                }
            }
        }
    }

    pub async fn ready_info(&self) -> Option<Value> {
        self.conn.read().await.as_ref().map(|c| c.ready.clone())
    }

    pub async fn generate(&self, mut req: Value) -> Result<Generation, String> {
        let conn = self.conn.read().await.clone().ok_or("the model worker is restarting; retry shortly")?;
        let id = format!("req-{}", uuid::Uuid::new_v4().simple());
        req["op"] = json!("generate");
        req["id"] = json!(id);
        let (tx, rx) = mpsc::unbounded_channel();
        conn.routes.lock().unwrap().insert(id.clone(), tx);
        if let Err(e) = conn.send(&req).await {
            conn.unroute(&id);
            return Err(format!("cannot reach the model worker: {e}"));
        }
        Ok(Generation { id, events: rx, conn, armed: true })
    }

    pub async fn forget(&self, key: &str) {
        if let Some(conn) = self.conn.read().await.clone() {
            let _ = conn.send(&json!({"op": "forget", "key": key})).await;
        }
    }

    pub async fn shutdown(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        let conn = self.conn.read().await.clone();
        if let Some(conn) = conn {
            let _ = conn.send(&json!({"op": "shutdown"})).await;
            let mut child = conn.child.lock().await;
            if tokio::time::timeout(Duration::from_secs(10), child.wait()).await.is_err() {
                let _ = child.kill().await;
            }
        }
    }
}
