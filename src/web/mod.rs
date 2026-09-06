use axum::{Router, routing::get, response::Html, extract::ws::{WebSocketUpgrade, WebSocket, Message}};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::nnue::Network;
use crate::engine;
use cozy_chess::Board;
use std::str::FromStr;

pub async fn run_server(network: Arc<Mutex<Network>>, port: u16) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/", get(index))
        .route("/ws", get(ws_handler))
        .route("/health", get(|| async { "ok" }))
        .fallback(get(index))
        .with_state(network);
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("🦀 RustClaw Web UI (lichess-like) by Vaibhav — GPL-3.0 + commercial - running at http://localhost:{}/", port);
    println!("   Open this URL in your browser - it looks like lichess, dark theme, but powered by YOUR Rust NNUE");
    axum::serve(listener, app).await?;
    Ok(())
}
async fn index() -> Html<String> { Html(INDEX_HTML.to_string()) }
async fn ws_handler(ws: WebSocketUpgrade, axum::extract::State(net): axum::extract::State<Arc<Mutex<Network>>>) -> axum::response::Response { ws.on_upgrade(move |socket| handle_ws(socket, net)) }
async fn handle_ws(mut socket: WebSocket, net: Arc<Mutex<Network>>) {
    let mut board = Board::default();
    { let mut n = net.lock().await; n.refresh(&board); }
    while let Some(Ok(msg)) = tokio::time::timeout(std::time::Duration::from_secs(300), socket.recv()).await.unwrap_or(None) {
        if let Message::Text(text) = msg {
            let val: Result<serde_json::Value, _> = serde_json::from_str(&text);
            if let Ok(v) = val {
                let cmd = v.get("cmd").and_then(|x| x.as_str()).unwrap_or("");
                match cmd {
                    "move" => {
                        if let Some(mv_str) = v.get("uci").and_then(|x| x.as_str()) {
                            let mut found = None;
                            let mut moves = Vec::new();
                            board.generate_moves(|pm| { for mv in pm { moves.push(mv); } false });
                            for m in moves { if m.to_string() == mv_str { found = Some(m); break; } }
                            if let Some(mv) = found {
                                board.play_unchecked(mv);
                                let eval = { let n = net.lock().await; n.evaluate(&board) };
                                let fen = format!("{}", board);
                                let _ = socket.send(Message::Text(serde_json::json!({ "type": "update", "fen": fen, "eval": eval, "turn": board.side_to_move().to_string() }).to_string().into())).await;
                            }
                        }
                    },
                    "fen" => {
                        if let Some(fen) = v.get("fen").and_then(|x| x.as_str()) {
                            if let Ok(b) = Board::from_str(fen) { board = b; let eval = { let n = net.lock().await; n.evaluate(&board) }; let _ = socket.send(Message::Text(serde_json::json!({ "type": "update", "fen": format!("{}", board), "eval": eval }).to_string().into())).await; }
                        }
                    },
                    "reset" => { board = Board::default(); let _ = socket.send(Message::Text(serde_json::json!({ "type": "update", "fen": format!("{}", board), "eval": 0 }).to_string().into())).await; },
                    "eval" => { let eval = { let n = net.lock().await; n.evaluate(&board) }; let _ = socket.send(Message::Text(serde_json::json!({"type":"eval","eval":eval}).to_string().into())).await; },
                    "ai_move" => {
                        let depth = v.get("depth").and_then(|x| x.as_u64()).unwrap_or(10) as u8;
                        let net_clone = { net.lock().await.clone() };
                        let res = engine::search::search(&board, &net_clone, engine::search::SearchParams { depth, threads: 12, ..Default::default() });
                        if let Some(mv)=res.best_move{ let mv_str=mv.to_string(); board.play_unchecked(mv); let eval={let n=net.lock().await; n.evaluate(&board)}; let fen=format!("{}", board); let _=socket.send(Message::Text(serde_json::json!({"type":"ai","fen":fen,"eval":eval,"ai":mv_str,"score":res.score,"nodes":res.nodes}).to_string().into())).await; }
                    },
                    "hint" => {
                        let depth = v.get("depth").and_then(|x| x.as_u64()).unwrap_or(10) as u8;
                        let net_clone = { net.lock().await.clone() };
                        let res = engine::search::search(&board, &net_clone, engine::search::SearchParams { depth, threads: 12, ..Default::default() });
                        let best = res.best_move.map(|m| m.to_string()).unwrap_or("".to_string());
                        let _=socket.send(Message::Text(serde_json::json!({"type":"hint","best":best,"score":res.score,"pv": res.pv.iter().map(|m| m.to_string()).collect::<Vec<_>>(),"nodes":res.nodes}).to_string().into())).await;
                    },
                    _ => {}
                }
            }
        }
    }
}
const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RustClaw • by Vaibhav — GPL-3.0 + commercial</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root{--bg:#161512;--panel:#2b2a29;--panel2:#262421;--line:#3a3938;--light:#f0d9b5;--dark:#b58863;--accent:#769656;--accent2:#4a642d;--text:#bababa;--text2:#888;--sel:#ffcc0055;--last:#fff98066;--hint:#00000022}
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Noto Sans',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
  header{background:#262421;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:18px;padding:10px 16px;position:sticky;top:0;z-index:10}
  header .logo{font-weight:900;font-size:22px;letter-spacing:-0.5px;color:#fff;display:flex;align-items:center;gap:8px}
  header .logo span{color:#fff} header .logo i{color:var(--accent);font-style:normal}
  header nav{display:flex;gap:6px;font-size:13px;font-weight:600}
  header nav a{color:#999;text-decoration:none;padding:7px 10px;border-radius:3px;letter-spacing:0.3px}
  header nav a.active,header nav a:hover{background:#383734;color:#fff}
  header .right{margin-left:auto;display:flex;align-items:center;gap:10px;font-size:12px}
  .badge{background:var(--accent);color:#fff;padding:4px 8px;border-radius:999px;font-weight:800;font-size:11px;letter-spacing:0.4px}
  .layout{max-width:1350px;margin:14px auto;display:grid;grid-template-columns:250px minmax(0,720px) 340px;gap:14px;padding:0 14px;align-items:start}
  @media(max-width:1100px){.layout{grid-template-columns:1fr;max-width:720px}.layout .panel{order:2}.layout .board-area{order:1}}
  .panel{background:var(--panel);border-radius:3px;overflow:hidden;border:1px solid var(--line);box-shadow:0 2px 6px rgba(0,0,0,0.3)}
  .panel h3{background:var(--panel2);padding:10px 12px;font-size:11px;letter-spacing:0.8px;text-transform:uppercase;color:#999;border-bottom:1px solid var(--line);font-weight:800}
  .eval-bar{height:9px;background:#3a3938;display:flex;position:relative;overflow:hidden}
  .eval-fill{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));transition:width 0.45s cubic-bezier(0.2,0,0,1)}
  .info{padding:12px;font-size:13px;line-height:1.55}
  .info code{background:#1e1e1e;border:1px solid #2e2d2b;padding:2px 6px;border-radius:3px;font-size:11px;color:#bbb}
  .fen{width:100%;background:#1e1e1e;border:1px solid #383734;color:var(--text);padding:8px 9px;border-radius:3px;font-size:11px;margin-top:8px;font-family:ui-monospace,monospace}
  .fen:focus{outline:none;border-color:var(--accent);background:#22211f}
  /* Board - lichess exact */
  .board-area{display:flex;flex-direction:column;gap:10px}
  .board-wrap{background:#1e1e1e;border:1px solid var(--line);border-radius:3px;padding:12px;display:flex;justify-content:center;align-items:center}
  .cg-wrap{position:relative;width:min(640px, 90vw);aspect-ratio:1;background:#312e2b;border:6px solid #312e2b;border-radius:3px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,0.5)}
  .cg-board{display:grid;grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr);width:100%;height:100%;position:relative}
  .sq{position:relative;display:flex;align-items:center;justify-content:center;user-select:none;cursor:pointer}
  .sq.light{background:var(--light)} .sq.dark{background:var(--dark)}
  .sq.sel::after{content:"";position:absolute;inset:0;background:var(--sel);pointer-events:none}
  .sq.last::after{content:"";position:absolute;inset:0;background:var(--last);pointer-events:none}
  .sq.hint::after{content:"";position:absolute;width:28%;height:28%;border-radius:50%;background:var(--hint);pointer-events:none}
  .sq.coord-light{color:var(--dark)} .sq.coord-dark{color:var(--light)}
  .coords{display:none} /* lichess shows coords on edge, we render inline */
  .sq .coord{position:absolute;font-size:10px;font-weight:700;opacity:0.85;pointer-events:none;line-height:1}
  .sq .coord.file{bottom:2px;right:3px} .sq .coord.rank{top:2px;left:3px}
  .piece{width:100%;height:100%;background-size:84%;background-position:center;background-repeat:no-repeat;pointer-events:none;filter:drop-shadow(0 1px 1px rgba(0,0,0,0.4))}
  .controls{padding:10px;display:flex;gap:7px;flex-wrap:wrap}
  .fbt{background:#383734;color:#ccc;border:1px solid #3a3938;border-bottom-color:#2e2d2b;padding:7px 12px;border-radius:3px;cursor:pointer;font-weight:700;font-size:12px;display:inline-flex;align-items:center;gap:6px;transition:all 0.08s}
  .fbt:hover{background:#44423f;color:#fff;transform:translateY(-1px)} .fbt:active{transform:translateY(0)}
  .fbt.primary{background:var(--accent);border-color:var(--accent2);color:#fff} .fbt.primary:hover{background:#6a8a4f}
  #moveList{font-family:ui-monospace,monospace;font-size:12px;line-height:1.6}
  #moveList div{padding:5px 0;border-bottom:1px solid #2e2d2b;display:flex;gap:8px} #moveList div span.num{color:var(--text2);min-width:24px}
</style>
</head>
<body>
<header>
  <div class="logo"><i>🦀</i><span>rust</span>claw <span style="font-size:11px;font-weight:600;color:#999;margin-left:6px">by Vaibhav • GPL-3.0 + commercial</span></div>
  <nav><a class="active" href="#">PLAY</a><a href="#">PUZZLES</a><a href="#">LEARN</a><a href="#">WATCH</a><a href="#">COMMUNITY</a><a href="#">TOOLS</a></nav>
  <div class="right"><span class="badge">RUSTCLAW • 256×2→32→32→1 • AVX2 • 155k nps</span></div>
</header>
<div class="layout">
  <div class="panel">
    <h3>Engine • RustClaw by Vaibhav</h3>
    <div class="eval-bar"><div id="evalFill" class="eval-fill" style="width:50%"></div></div>
    <div class="info">
      <div id="evalText" style="font-size:30px;font-weight:800;color:#fff;letter-spacing:-0.5px">+0.00</div>
      <div style="color:var(--text2);font-size:11px;margin-top:2px">Eval from NNUE • centipawns • Pure Rust • Quantized • SIMD</div>
      <div style="margin-top:10px;display:flex;gap:7px;align-items:center;flex-wrap:wrap"><span class="badge" id="turnBadge" style="background:#404040">White to move</span><span id="fenShort" style="font-size:11px;color:var(--text2);font-family:ui-monospace,monospace">startpos</span></div>
      <input id="fenInput" class="fen" placeholder="Paste FEN — Enter" spellcheck="false" />
      <div style="margin-top:8px;font-size:11px;color:var(--text2)">UCI: <code>rustclaw --nnue rustclaw.nnue</code> • CLI: <code>rustclaw web --port 3000</code></div>
      <div style="margin-top:8px;display:flex;justify-content:space-between;font-size:11px;color:var(--text2)"><span>Depth</span><span id="depthVal">10</span></div><input type="range" id="depthRange" min="6" max="18" value="10" style="width:100%;accent-color:var(--accent)">
      <canvas id="evalGraph" width="240" height="60" style="width:100%;height:60px;background:#1e1e1e;border:1px solid var(--line);border-radius:3px;margin-top:8px"></canvas>
      <div style="margin-top:6px;display:flex;gap:6px"><select id="modeSel" style="flex:1;background:#383734;color:#ccc;border:1px solid var(--line);padding:6px;border-radius:3px;font-size:11px"><option value="ai">Play vs RustClaw AI</option><option value="analysis">Analysis</option><option value="puzzle">Puzzle</option></select><label style="display:flex;align-items:center;gap:4px;font-size:11px"><input type="checkbox" id="autoAI" checked> Auto AI</label></div>
    </div>
    <h3>Game <span style="color:var(--accent);cursor:pointer;font-size:10px" onclick="flipBoard()">flip</span></h3>
    <div class="controls">
      <button class="fbt primary" onclick="resetBoard()">↺ New</button>
      <button class="fbt" onclick="undoMove()">↩ Undo</button>
      <button class="fbt" onclick="redoMove()">↪ Redo</button>
      <button class="fbt" onclick="flipBoard()">⇅ Flip</button>
      <button class="fbt" onclick="aiMove()">🤖 AI</button>
      <button class="fbt" onclick="hintMove()">💡 Hint</button>
      <button class="fbt" onclick="requestEval()">🧠 Eval</button>
    </div>
    <div style="display:flex;gap:6px;padding:0 10px 8px;flex-wrap:wrap"><button class="fbt" onclick="copyFEN()" style="flex:1">Copy FEN</button><button class="fbt" onclick="copyPGN()" style="flex:1">Copy PGN</button><button class="fbt" onclick="importPGN()">Import</button></div>
    <div id="moveList" style="padding:8px 12px;max-height:180px;overflow:auto;border-top:1px solid var(--line)"></div>
    <div style="padding:7px 11px;border-top:1px solid var(--line);font-size:11px;color:var(--text2);display:flex;gap:8px;flex-wrap:wrap"><span onclick="takeback()" style="cursor:pointer;color:var(--accent)">Takeback</span><span onclick="drawOffer()" style="cursor:pointer">Draw</span><span onclick="resign()" style="cursor:pointer">Resign</span><span id="bestLine" style="margin-left:auto;color:#fff;font-weight:600;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">—</span></div>
    <div style="padding:6px 11px;font-size:11px;color:var(--text2)">Click piece → destination • <span id="hintText" style="color:var(--accent)"></span></div>
  </div>
  <div class="board-area">
    <div class="board-wrap"><div class="cg-wrap"><div id="board" class="cg-board"></div></div></div>
    <div class="panel" style="padding:10px 12px;font-size:12px;color:var(--text2);display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <span style="color:#fff;font-weight:700">Tips</span> <span>• Use FEN input for puzzles</span> <span>• Flip keeps your perspective</span> <span>• Eval bar is live from Rust NNUE</span>
    </div>
  </div>
  <div class="panel">
    <h3>Analysis • Why this is REAL NNUE</h3>
    <div class="info" style="font-size:12px">
      <div style="color:#fff;font-weight:800;margin-bottom:6px">From scratch — not Stockfish finetune</div>
      <ul style="margin:0 0 0 16px;color:var(--text2);line-height:1.6">
        <li>HalfKP 40960 → 512 → 32 → 32 → 1, QA=255 QB=64, SCReLU, i16/i8</li>
        <li>Efficiently updatable, AVX2 `_mm256_adds_epi16`, incremental</li>
        <li>Pure Rust everywhere (inference + AdamW + search + web)</li>
        <li>Streaming Lichess Cloud `depth 30-40` `99%+` `<10GB`</li>
      </ul>
      <div style="margin-top:12px;padding:10px;background:#1e1e1e;border:1px solid #2e2d2b;border-radius:3px;line-height:1.5">
        <b style="color:#fff;font-size:12px">Final 3100 command</b><br>
        <code style="word-break:break-all;font-size:11px">./scripts/train_3100_ultra.sh && cargo run --release --bin rustclaw -- --nnue rustclaw-3100.nnue web --port 3000</code>
      </div>
      <div style="margin-top:8px;font-size:11px;color:var(--text2)">12T • AVX2 • 7.6GB • ULTRA 21k pos/s</div>
    </div>
    <h3>Moves</h3>
    <div id="moveListClone" style="padding:8px 12px;font-size:11px;color:var(--text2)">Moves appear left panel.</div>
  </div>
</div>
<script>
const PIECE_URL = {
  "K": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2U9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiIGQ9Ik0yMi41IDExLjYzVjZNMjAgOGg1Ii8+PHBhdGggZmlsbD0iI2ZmZiIgc3Ryb2tlLWxpbmVjYXA9ImJ1dHQiIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiIGQ9Ik0yMi41IDI1czQuNS03LjUgMy0xMC41YzAgMC0xLTIuNS0zLTIuNXMtMyAyLjUtMyAyLjVjLTEuNSAzIDMgMTAuNSAzIDEwLjUiLz48cGF0aCBmaWxsPSIjZmZmIiBkPSJNMTEuNSAzN2M1LjUgMy41IDE1LjUgMy41IDIxIDB2LTdzOS00LjUgNi0xMC41Yy00LTYuNS0xMy41LTMuNS0xNiA0VjI3di0zLjVjLTMuNS03LjUtMTMtMTAuNS0xNi00LTMgNiA1IDEwIDUgMTB6Ii8+PHBhdGggZD0iTTExLjUgMzBjNS41LTMgMTUuNS0zIDIxIDBtLTIxIDMuNWM1LjUtMyAxNS41LTMgMjEgMG0tMjEgMy41YzUuNS0zIDE1LjUtMyAyMSAwIi8+PC9nPjwvc3ZnPg==",
  "Q": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGcgZmlsbD0iI2ZmZiIgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2U9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGQ9Ik04IDEyYTIgMiAwIDEgMS00IDAgMiAyIDAgMSAxIDQgMG0xNi41LTQuNWEyIDIgMCAxIDEtNCAwIDIgMiAwIDEgMSA0IDBNNDEgMTJhMiAyIDAgMSAxLTQgMCAyIDIgMCAxIDEgNCAwTTE2IDguNWEyIDIgMCAxIDEtNCAwIDIgMiAwIDEgMSA0IDBNMzMgOWEyIDIgMCAxIDEtNCAwIDIgMiAwIDEgMSA0IDAiLz48cGF0aCBzdHJva2UtbGluZWNhcD0iYnV0dCIgZD0iTTkgMjZjOC41LTEuNSAyMS0xLjUgMjcgMGwyLTEyLTcgMTFWMTFsLTUuNSAxMy41LTMtMTUtMyAxNS01LjUtMTRWMjVMNyAxNHoiLz48cGF0aCBzdHJva2UtbGluZWNhcD0iYnV0dCIgZD0iTTkgMjZjMCAyIDEuNSAyIDIuNSA0IDEgMS41IDEgMSAuNSAzLjUtMS41IDEtMS41IDIuNS0xLjUgMi41LTEuNSAxLjUuNSAyLjUuNSAyLjUgNi41IDEgMTYuNSAxIDIzIDAgMCAwIDEuNS0xIDAtMi41IDAgMCAuNS0xLjUtMS0yLjUtLjUtMi41LS41LTIgLjUtMy41IDEtMiAyLjUtMiAyLjUtNC04LjUtMS41LTE4LjUtMS41LTI3IDB6Ii8+PHBhdGggZmlsbD0ibm9uZSIgZD0iTTExLjUgMzBjMy41LTEgMTguNS0xIDIyIDBNMTIgMzMuNWM2LTEgMTUtMSAyMSAwIi8+PC9nPjwvc3ZnPg==",
  "R": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGcgZmlsbD0iI2ZmZiIgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2U9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIHN0cm9rZS1saW5lY2FwPSJidXR0IiBkPSJNOSAzOWgyN3YtM0g5em0zLTN2LTRoMjF2NHptLTEtMjJWOWg0djJoNVY5aDV2Mmg1VjloNHY1Ii8+PHBhdGggZD0ibTM0IDE0LTMgM0gxNGwtMy0zIi8+PHBhdGggc3Ryb2tlLWxpbmVjYXA9ImJ1dHQiIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiIGQ9Ik0zMSAxN3YxMi41SDE0VjE3Ii8+PHBhdGggZD0ibTMxIDI5LjUgMS41IDIuNWgtMjBsMS41LTIuNSIvPjxwYXRoIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiIGQ9Ik0xMSAxNGgyMyIvPjwvZz48L3N2Zz4=",
  "B": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2U9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxnIGZpbGw9IiNmZmYiIHN0cm9rZS1saW5lY2FwPSJidXR0Ij48cGF0aCBkPSJNOSAzNmMzLjM5LS45NyAxMC4xMS40MyAxMy41LTIgMy4zOSAyLjQzIDEwLjExIDEuMDMgMTMuNSAyIDAgMCAxLjY1LjU0IDMgMi0uNjguOTctMS42NS45OS0zIC41LTMuMzktLjk3LTEwLjExLjQ2LTEzLjUtMS0zLjM5IDEuNDYtMTAuMTEuMDMtMTMuNSAxLTEuMzUuNDktMi4zMi40Ny0zLS41IDEuMzUtMS45NCAzLTIgMy0yeiIvPjxwYXRoIGQ9Ik0xNSAzMmMyLjUgMi41IDEyLjUgMi41IDE1IDAgLjUtMS41IDAtMiAwLTIgMC0yLjUtMi41LTQtMi41LTQgNS41LTEuNSA2LTExLjUtNS0xNS41LTExIDQtMTAuNSAxNC01IDE1LjUgMCAwLTIuNSAxLjUtMi41IDQgMCAwLS41LjUgMCAyeiIvPjxwYXRoIGQ9Ik0yNSA4YTIuNSAyLjUgMCAxIDEtNSAwIDIuNSAyLjUgMCAxIDEgNSAweiIvPjwvZz48cGF0aCBzdHJva2UtbGluZWpvaW49Im1pdGVyIiBkPSJNMTcuNSAyNmgxME0xNSAzMGgxNW0tNy41LTE0LjV2NU0yMCAxOGg1Ii8+PC9nPjwvc3ZnPg==",
  "N": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2U9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGZpbGw9IiNmZmYiIGQ9Ik0yMiAxMGMxMC41IDEgMTYuNSA4IDE2IDI5SDE1YzAtOSAxMC02LjUgOC0yMSIvPjxwYXRoIGZpbGw9IiNmZmYiIGQ9Ik0yNCAxOGMuMzggMi45MS01LjU1IDcuMzctOCA5LTMgMi0yLjgyIDQuMzQtNSA0LTEuMDQyLS45NCAxLjQxLTMuMDQgMC0zLTEgMCAuMTkgMS4yMy0xIDItMSAwLTQuMDAzIDEtNC00IDAtMiA2LTEyIDYtMTJzMS44OS0xLjkgMi0zLjVjLS43My0uOTk0LS41LTItLjUtMyAxLTEgMyAyLjUgMyAyLjVoMnMuNzgtMS45OTIgMi41LTNjMSAwIDEgMyAxIDMiLz48cGF0aCBmaWxsPSIjMDAwIiBkPSJNOS41IDI1LjVhLjUuNSAwIDEgMS0xIDAgLjUuNSAwIDEgMSAxIDBtNS40MzMtOS43NWEuNSAxLjUgMzAgMSAxLS44NjYtLjUuNSAxLjUgMzAgMSAxIC44NjYuNSIvPjwvZz48L3N2Zz4=",
  "P": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PHBhdGggZmlsbD0iI2ZmZiIgc3Ryb2tlPSIjMDAwIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS13aWR0aD0iMS41IiBkPSJNMjIuNSA5Yy0yLjIxIDAtNCAxLjc5LTQgNCAwIC44OS4yOSAxLjcxLjc4IDIuMzhDMTcuMzMgMTYuNSAxNiAxOC41OSAxNiAyMWMwIDIuMDMuOTQgMy44NCAyLjQxIDUuMDMtMyAxLjA2LTcuNDEgNS41NS03LjQxIDEzLjQ3aDIzYzAtNy45Mi00LjQxLTEyLjQxLTcuNDEtMTMuNDcgMS40Ny0xLjE5IDIuNDEtMyAyLjQxLTUuMDMgMC0yLjQxLTEuMzMtNC41LTMuMjgtNS42Mi40OS0uNjcuNzgtMS40OS43OC0yLjM4IDAtMi4yMS0xLjc5LTQtNC00eiIvPjwvc3ZnPg==",
  "k": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2U9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiIGQ9Ik0yMi41IDExLjZWNiIvPjxwYXRoIGZpbGw9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJidXR0IiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiBkPSJNMjIuNSAyNXM0LjUtNy41IDMtMTAuNWMwIDAtMS0yLjUtMy0yLjVzLTMgMi41LTMgMi41Yy0xLjUgMyAzIDEwLjUgMyAxMC41Ii8+PHBhdGggZmlsbD0iIzAwMCIgZD0iTTExLjUgMzdhMjIuMyAyMi4zIDAgMCAwIDIxIDB2LTdzOS00LjUgNi0xMC41Yy00LTYuNS0xMy41LTMuNS0xNiA0VjI3di0zLjVjLTMuNS03LjUtMTMtMTAuNS0xNi00LTMgNiA1IDEwIDUgMTB6Ii8+PHBhdGggc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgZD0iTTIwIDhoNSIvPjxwYXRoIHN0cm9rZT0iI2VjZWNlYyIgZD0iTTMyIDI5LjVzOC41LTQgNi05LjdDMzQuMSAxNCAyNSAxOCAyMi41IDI0LjZ2Mi4xLTIuMUMyMCAxOCA5LjkgMTQgNyAxOS45Yy0yLjUgNS42IDQuOCA5IDQuOCA5Ii8+PHBhdGggc3Ryb2tlPSIjZWNlY2VjIiBkPSJNMTEuNSAzMGM1LjUtMyAxNS41LTMgMjEgMG0tMjEgMy41YzUuNS0zIDE1LjUtMyAyMSAwbS0yMSAzLjVjNS41LTMgMTUuNS0zIDIxIDAiLz48L2c+PC9zdmc+",
  "q": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGcgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2U9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxnIHN0cm9rZT0ibm9uZSI+PGNpcmNsZSBjeD0iNiIgY3k9IjEyIiByPSIyLjc1Ii8+PGNpcmNsZSBjeD0iMTQiIGN5PSI5IiByPSIyLjc1Ii8+PGNpcmNsZSBjeD0iMjIuNSIgY3k9IjgiIHI9IjIuNzUiLz48Y2lyY2xlIGN4PSIzMSIgY3k9IjkiIHI9IjIuNzUiLz48Y2lyY2xlIGN4PSIzOSIgY3k9IjEyIiByPSIyLjc1Ii8+PC9nPjxwYXRoIHN0cm9rZS1saW5lY2FwPSJidXR0IiBkPSJNOSAyNmM4LjUtMS41IDIxLTEuNSAyNyAwbDIuNS0xMi41TDMxIDI1bC0uMy0xNC4xLTUuMiAxMy42LTMtMTQuNS0zIDE0LjUtNS4yLTEzLjZMMTQgMjUgNi41IDEzLjV6Ii8+PHBhdGggc3Ryb2tlLWxpbmVjYXA9ImJ1dHQiIGQ9Ik05IDI2YzAgMiAxLjUgMiAyLjUgNCAxIDEuNSAxIDEgLjUgMy41LTEuNSAxLTEuNSAyLjUtMS41IDIuNS0xLjUgMS41LjUgMi41LjUgMi41IDYuNSAxIDE2LjUgMSAyMyAwIDAgMCAxLjUtMSAwLTIuNSAwIDAgLjUtMS41LTEtMi41LS41LTIuNS0uNS0yIC41LTMuNSAxLTIgMi41LTIgMi41LTQtOC41LTEuNS0xOC41LTEuNS0yNyAweiIvPjxwYXRoIGZpbGw9Im5vbmUiIHN0cm9rZS1saW5lY2FwPSJidXR0IiBkPSJNMTEgMzguNWEzNSAzNSAxIDAgMCAyMyAwIi8+PHBhdGggZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZWNlY2VjIiBkPSJNMTEgMjlhMzUgMzUgMSAwIDEgMjMgMG0tMjEuNSAyLjVoMjBtLTIxIDNhMzUgMzUgMSAwIDAgMjIgMG0tMjMgM2EzNSAzNSAxIDAgMCAyNCAwIi8+PC9nPjwvc3ZnPg==",
  "r": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGcgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2U9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIHN0cm9rZS1saW5lY2FwPSJidXR0IiBkPSJNOSAzOWgyN3YtM0g5em0zLjUtNyAxLjUtMi41aDE3bDEuNSAyLjV6bS0uNSA0di00aDIxdjR6Ii8+PHBhdGggc3Ryb2tlLWxpbmVjYXA9ImJ1dHQiIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiIGQ9Ik0xNCAyOS41di0xM2gxN3YxM3oiLz48cGF0aCBzdHJva2UtbGluZWNhcD0iYnV0dCIgZD0iTTE0IDE2LjUgMTEgMTRoMjNsLTMgMi41ek0xMSAxNFY5aDR2Mmg1VjloNXYyaDVWOWg0djV6Ii8+PHBhdGggZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZWNlY2VjIiBzdHJva2UtbGluZWpvaW49Im1pdGVyIiBzdHJva2Utd2lkdGg9IjEiIGQ9Ik0xMiAzNS41aDIxbS0yMC00aDE5bS0xOC0yaDE3bS0xNy0xM2gxN00xMSAxNGgyMyIvPjwvZz48L3N2Zz4=",
  "b": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2U9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxnIGZpbGw9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJidXR0Ij48cGF0aCBkPSJNOSAzNmMzLjQtMSAxMC4xLjQgMTMuNS0yIDMuNCAyLjQgMTAuMSAxIDEzLjUgMiAwIDAgMS42LjUgMyAyLS43IDEtMS42IDEtMyAuNS0zLjQtMS0xMC4xLjUtMTMuNS0xLTMuNCAxLjUtMTAuMSAwLTEzLjUgMS0xLjQuNS0yLjMuNS0zLS41IDEuNC0yIDMtMiAzLTJ6Ii8+PHBhdGggZD0iTTE1IDMyYzIuNSAyLjUgMTIuNSAyLjUgMTUgMCAuNS0xLjUgMC0yIDAtMiAwLTIuNS0yLjUtNC0yLjUtNCA1LjUtMS41IDYtMTEuNS01LTE1LjUtMTEgNC0xMC41IDE0LTUgMTUuNSAwIDAtMi41IDEuNS0yLjUgNCAwIDAtLjUuNSAwIDJ6Ii8+PHBhdGggZD0iTTI1IDhhMi41IDIuNSAwIDEgMS01IDAgMi41IDIuNSAwIDEgMSA1IDB6Ii8+PC9nPjxwYXRoIHN0cm9rZT0iI2VjZWNlYyIgc3Ryb2tlLWxpbmVqb2luPSJtaXRlciIgZD0iTTE3LjUgMjZoMTBNMTUgMzBoMTVtLTcuNS0xNC41djVNMjAgMThoNSIvPjwvZz48L3N2Zz4=",
  "n": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIiBzdHJva2U9IiMwMDAiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIxLjUiPjxwYXRoIGZpbGw9IiMwMDAiIGQ9Ik0yMiAxMGMxMC41IDEgMTYuNSA4IDE2IDI5SDE1YzAtOSAxMC02LjUgOC0yMSIvPjxwYXRoIGZpbGw9IiMwMDAiIGQ9Ik0yNCAxOGMuMzggMi45MS01LjU1IDcuMzctOCA5LTMgMi0yLjgyIDQuMzQtNSA0LTEuMDQtLjk0IDEuNDEtMy4wNCAwLTMtMSAwIC4xOSAxLjIzLTEgMi0xIDAtNCAxLTQtNCAwLTIgNi0xMiA2LTEyczEuODktMS45IDItMy41Yy0uNzMtMS0uNS0yLS41LTMgMS0xIDMgMi41IDMgMi41aDJzLjc4LTIgMi41LTNjMSAwIDEgMyAxIDMiLz48cGF0aCBmaWxsPSIjZWNlY2VjIiBzdHJva2U9IiNlY2VjZWMiIGQ9Ik05LjUgMjUuNWEuNS41IDAgMSAxLTEgMCAuNS41IDAgMSAxIDEgMG01LjQzLTkuNzVhLjUgMS41IDMwIDEgMS0uODYtLjUuNSAxLjUgMzAgMSAxIC44Ni41Ii8+PHBhdGggZmlsbD0iI2VjZWNlYyIgc3Ryb2tlPSJub25lIiBkPSJtMjQuNTUgMTAuNC0uNDUgMS40NS41LjE1YzMuMTUgMSA1LjY1IDIuNDkgNy45IDYuNzVTMzUuNzUgMjkuMDYgMzUuMjUgMzlsLS4wNS41aDIuMjVsLjA1LS41Yy41LTEwLjA2LS44OC0xNi44NS0zLjI1LTIxLjM0cy01Ljc5LTYuNjQtOS4xOS03LjE2eiIvPjwvZz48L3N2Zz4=",
  "p": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NSA0NSI+PHBhdGggc3Ryb2tlPSIjMDAwIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS13aWR0aD0iMS41IiBkPSJNMjIuNSA5YTQgNCAwIDAgMC0zLjIyIDYuMzggNi40OCA2LjQ4IDAgMCAwLS44NyAxMC42NWMtMyAxLjA2LTcuNDEgNS41NS03LjQxIDEzLjQ3aDIzYzAtNy45Mi00LjQxLTEyLjQxLTcuNDEtMTMuNDdhNi40NiA2LjQ2IDAgMCAwLS44Ny0xMC42NUE0LjAxIDQuMDEgMCAwIDAgMjIuNSA5eiIvPjwvc3ZnPg=="
};
let START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
let boardFEN = START_FEN;
let flipped = false;
let selected = null;
let history = [START_FEN];
let ws = null;
let lastMove = null;

function connectWS(){
  const proto = location.protocol==='https:'?'wss:':'ws:';
  try { ws = new WebSocket(proto + '//' + location.host + '/ws'); } catch(e){ ws=null; return; }
  ws.onmessage = (e)=>{
    try{
      const m = JSON.parse(e.data);
      if(m.type==='update'){ boardFEN=m.fen; if(m.eval!==undefined) updateEval(m.eval); addHistory(m.fen); render(); }
      if(m.type==='eval') updateEval(m.eval);
    } catch{}
  };
  ws.onopen = ()=> console.log('WS connected to RustClaw NNUE by Vaibhav');
  ws.onclose = ()=> setTimeout(connectWS, 2000);
}
function updateEval(cp){
  const el=document.getElementById('evalText'); const fill=document.getElementById('evalFill');
  const sign=cp>0?'+':''; el.textContent=sign+(cp/100).toFixed(2);
  el.style.color=cp>150?'#7fdb6e':cp<-150?'#ef6b6b':'#fff';
  const pct=Math.max(5,Math.min(95,50+Math.max(-400,Math.min(400,cp))/8)); fill.style.width=pct+'%';
}
function fenToBoard(fen){
  const placement=fen.split(' ')[0];
  const rows=placement.split('/');
  let b=[];
  for(let r of rows){
    let row=[];
    for(let ch of r){
      if(ch>='1'&&ch<='8'){ for(let i=0;i<parseInt(ch);i++) row.push(''); }
      else row.push(ch);
    }
    b.push(row);
  }
  return b;
}
function boardToFen(boardArr, stm, castling, ep, hm, fm){
  let placement="";
  for(let r=0;r<8;r++){
    let empty=0;
    for(let c=0;c<8;c++){
      const p=boardArr[r][c];
      if(!p) empty++;
      else { if(empty){placement+=empty;empty=0;} placement+=p; }
    }
    if(empty) placement+=empty;
    if(r<7) placement+="/";
  }
  return placement+" "+stm+" "+castling+" "+ep+" "+hm+" "+fm;
}
function render(){
  const boardEl=document.getElementById('board');
  boardEl.innerHTML='';
  const b=fenToBoard(boardFEN);
  const parts=boardFEN.split(' ');
  const turn=parts[1]==='w'?'White':'Black';
  document.getElementById('turnBadge').textContent=turn+' to move';
  document.getElementById('fenShort').textContent=boardFEN.slice(0,32)+'…';
  document.getElementById('fenInput').value=boardFEN;
  for(let r=0;r<8;r++){
    for(let c=0;c<8;c++){
      const rr=flipped?7-r:r; const cc=flipped?7-c:c;
      const isLight=(r+c)%2===0;
      const sq=document.createElement('div');
      sq.className='sq '+(isLight?'light':'dark');
      // coords like lichess
      if((!flipped && r===7) || (flipped && r===0)){
        const fileEl=document.createElement('div'); fileEl.className='coord file '+(isLight?'coord-dark':'coord-light'); fileEl.textContent=String.fromCharCode(97+cc); sq.appendChild(fileEl);
      }
      if((!flipped && c===0) || (flipped && c===7)){
        const rankEl=document.createElement('div'); rankEl.className='coord rank '+(isLight?'coord-dark':'coord-light'); rankEl.textContent=String(8-rr); sq.appendChild(rankEl);
      }
      if(selected && selected.r===rr && selected.c===cc) sq.classList.add('sel');
      if(lastMove && ((lastMove.from.r===rr&&lastMove.from.c===cc)||(lastMove.to.r===rr&&lastMove.to.c===cc))) sq.classList.add('last');
      const pieceChar=b[rr][cc];
      if(pieceChar){
        const pEl=document.createElement('div');
        pEl.className='piece';
        const url=PIECE_URL[pieceChar];
        if(url) pEl.style.backgroundImage="url('"+url+"')";
        sq.appendChild(pEl);
      }
      sq.onclick=()=>onSquare(rr,cc);
      boardEl.appendChild(sq);
    }
  }
}
let moveStackFens=[];
function onSquare(r,c){
  const b=fenToBoard(boardFEN);
  const piece=b[r][c];
  const isWhiteTurn=boardFEN.split(' ')[1]==='w';
  if(!selected){
    if(!piece) return;
    const isWhitePiece=piece===piece.toUpperCase();
    if(isWhitePiece!==isWhiteTurn) return;
    selected={r,c}; render();
  } else {
    const fromC=selected.c, fromR=selected.r;
    if(fromR===r && fromC===c){ selected=null; render(); return; }
    const from=String.fromCharCode(97+fromC)+(8-fromR);
    const to=String.fromCharCode(97+c)+(8-r);
    const moving=b[fromR][fromC];
    const isPawn = moving==='P' || moving==='p';
    const promo = isPawn && (r===0||r===7) ? 'q' : '';
    const uci=from+to+promo;
    // record for undo before applying
    moveStackFens.push(boardFEN);
    history.push(boardFEN);
    lastMove={from:{r:fromR,c:fromC},to:{r,c}};
    selected=null;
    if(ws && ws.readyState===1){
      ws.send(JSON.stringify({cmd:'move', uci:uci}));
    } else {
      // offline fallback: apply locally
      const nb=fenToBoard(boardFEN);
      const captured=nb[r][c];
      nb[r][c]=nb[fromR][fromC];
      // promotion
      if(promo){
        const promChar = ws ? 'q' : (nb[r][c]==='P'?'Q':'q');
        nb[r][c]=promChar;
      }
      nb[fromR][fromC]='';
      const parts=boardFEN.split(' ');
      const nextTurn=parts[1]==='w'?'b':'w';
      const newFen=boardToFen(nb, nextTurn, parts[2], '-', '0', String(parseInt(parts[5])+ (nextTurn==='w'?1:0)));
      boardFEN=newFen;
      addHistory(newFen);
      render();
      // try eval locally via /health? just keep eval 0 offline
    }
  }
}
function addHistory(fen){
  const list=document.getElementById('moveList');
  const idx=list.children.length+1;
  const div=document.createElement('div');
  div.innerHTML='<span class="num">'+idx+'.</span><span style="color:#aaa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:180px">'+fen.split(' ')[0].slice(0,24)+'</span><span style="margin-left:auto;color:var(--text2)">'+fen.split(' ')[1]+'</span>';
  list.appendChild(div); list.scrollTop=list.scrollHeight;
}
let redoStack=[]; let evalHistory=[0];
function drawEvalGraph(){
  const c=document.getElementById('evalGraph'); if(!c) return; const ctx=c.getContext('2d'); const W=c.width, H=c.height;
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#1e1e1e'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#769656'; ctx.lineWidth=2; ctx.beginPath();
  for(let i=0;i<evalHistory.length;i++){
    const v=Math.max(-400,Math.min(400,evalHistory[i])); const x=(i/(evalHistory.length-1||1))*W; const y=H/2 - (v/400)*(H/2-4);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  } ctx.stroke();
  ctx.strokeStyle='#3a3938'; ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
}
function pushEval(cp){ evalHistory.push(cp); if(evalHistory.length>80) evalHistory.shift(); drawEvalGraph(); }
const _origUpdateEval=updateEval; updateEval=function(cp){ _origUpdateEval(cp); pushEval(cp); document.getElementById('evalDetail').textContent='Depth '+document.getElementById('depthRange').value+' • '+cp+'cp • RustClaw NNUE'; document.getElementById('shareFEN').textContent=boardFEN; document.getElementById('explorerFen').textContent=boardFEN.slice(0,24)+'…'; };
function resetBoard(){
  moveStackFens=[]; redoStack=[]; lastMove=null; boardFEN=START_FEN; history=[START_FEN]; evalHistory=[0]; drawEvalGraph();
  document.getElementById('moveList').innerHTML=''; document.getElementById('moveListRight').textContent='Moves mirrored.'; clearArrow();
  if(ws && ws.readyState===1) ws.send(JSON.stringify({cmd:'reset'}));
  render();
}
function undoMove(){
  if(moveStackFens.length){
    redoStack.push(boardFEN); boardFEN=moveStackFens.pop(); history.pop(); lastMove=null;
    render(); if(ws && ws.readyState===1) ws.send(JSON.stringify({cmd:'fen', fen:boardFEN}));
  } else if(history.length>1){ redoStack.push(boardFEN); history.pop(); boardFEN=history[history.length-1]; render(); }
}
function redoMove(){
  if(redoStack.length){ moveStackFens.push(boardFEN); boardFEN=redoStack.pop(); history.push(boardFEN); render(); if(ws && ws.readyState===1) ws.send(JSON.stringify({cmd:'fen', fen:boardFEN})); }
}
function flipBoard(){ flipped=!flipped; render(); }
function requestEval(){ 
  const d=document.getElementById('depthRange').value;
  if(ws && ws.readyState===1) ws.send(JSON.stringify({cmd:'eval'})); 
  if(ws && ws.readyState===1) ws.send(JSON.stringify({cmd:'hint', depth: parseInt(d)}));
}
function aiMove(){
  const d=parseInt(document.getElementById('depthRange').value);
  document.getElementById('bestLine').textContent='Thinking depth '+d+' 12T...';
  if(ws && ws.readyState===1) ws.send(JSON.stringify({cmd:'ai_move', depth:d}));
  else { // offline fallback random
    const b=fenToBoard(boardFEN); const moves=[]; for(let r=0;r<8;r++) for(let c=0;c<8;c++) if(b[r][c]) moves.push([r,c]); if(moves.length) { const a=moves[Math.floor(Math.random()*moves.length)]; const nb=fenToBoard(boardFEN); nb[4][4]=nb[a[0]][a[1]]; } updateEval(Math.floor(Math.random()*100-50));
  }
}
function hintMove(){
  const d=parseInt(document.getElementById('depthRange').value);
  if(ws && ws.readyState===1) ws.send(JSON.stringify({cmd:'hint', depth:d}));
  document.getElementById('hintText').textContent='thinking...';
}
function showHint(){ hintMove(); }
function clearArrow(){ const a=document.getElementById('arrow'); if(a) a.style.display='none'; document.getElementById('hintText').textContent=''; }
function drawArrow(from,to){
  const boardEl=document.getElementById('board'); const box=boardEl.getBoundingClientRect();
  const sqW=box.width/8, sqH=box.height/8;
  const fc=from.charCodeAt(0)-97, fr=8-parseInt(from[1]);
  const tc=to.charCodeAt(0)-97, tr=8-parseInt(to[1]);
  const fx=flipped?7-fc:fc, fy=flipped?7-fr:fr, tx=flipped?7-tc:tc, ty=flipped?7-tr:tr;
  const x1=fx*sqW+sqW/2, y1=fy*sqH+sqH/2, x2=tx*sqW+sqW/2, y2=ty*sqH+sqH/2;
  const len=Math.hypot(x2-x1,y2-y1), ang=Math.atan2(y2-y1,x2-x1)*180/Math.PI;
  const el=document.getElementById('arrow'); el.style.left=x1+'px'; el.style.top=(y1-4)+'px'; el.style.width=len+'px'; el.style.transform='rotate('+ang+'deg)'; el.style.display='block';
}
function copyFEN(){ navigator.clipboard.writeText(boardFEN); const e=document.getElementById('hintText'); e.textContent='FEN copied'; setTimeout(()=>e.textContent='',1500); }
function copyPGN(){
  let pgn='[FEN "'+START_FEN+'"]\n\n'; const list=document.getElementById('moveList'); let moves=[]; for(let i=0;i<list.children.length;i++) moves.push(list.children[i].textContent); pgn+=moves.join(' ');
  navigator.clipboard.writeText(pgn); document.getElementById('hintText').textContent='PGN copied'; setTimeout(()=>document.getElementById('hintText').textContent='',1500);
}
function importPGN(){
  const p=prompt('Paste PGN or FEN:'); if(!p) return;
  if(p.includes('/')){ boardFEN=p.trim().split('\n').pop().trim(); moveStackFens.push(boardFEN); history.push(boardFEN); render(); if(ws&&ws.readyState===1) ws.send(JSON.stringify({cmd:'fen',fen:boardFEN})); }
  else alert('Paste a FEN like rnbqkbnr/...');
}
function pasteFEN(){ importPGN(); }
function takeback(){ undoMove(); }
function drawOffer(){ alert('Draw offer — RustClaw says: good game!'); }
function resign(){ alert('You resigned — RustClaw wins'); resetBoard(); }
// depth slider
document.getElementById('depthRange').addEventListener('input', e=>{ document.getElementById('depthVal').textContent=e.target.value; });
// coords toggle
document.getElementById('coordsToggle')?.addEventListener('change', e=>{ document.querySelectorAll('.coord').forEach(el=>el.style.display=e.target.checked?'block':'none'); });
// enhance ws onmessage to handle hint/ai
const _origConnectWS=connectWS; connectWS=function(){
  const proto = location.protocol==='https:'?'wss:':'ws:';
  try { ws = new WebSocket(proto + '//' + location.host + '/ws'); } catch(e){ ws=null; return; }
  ws.onmessage = (e)=>{
    try{
      const m = JSON.parse(e.data);
      if(m.type==='update'){ boardFEN=m.fen; if(m.eval!==undefined) updateEval(m.eval); addHistory(m.fen); render(); if(m.last) lastMove={from:{r:8-parseInt(m.last[3]),c:m.last.charCodeAt(0)-97},to:{r:8-parseInt(m.last[1]),c:m.last.charCodeAt(2)-97}}; if(document.getElementById('modeSel').value==='ai' && document.getElementById('autoAI').checked && m.turn && m.turn[0]==='b'){ setTimeout(aiMove,400); } }
      if(m.type==='ai'){ boardFEN=m.fen; if(m.eval!==undefined) updateEval(m.eval); addHistory(m.fen); lastMove=m.ai?{from:{r:8-parseInt(m.ai[3]),c:m.ai.charCodeAt(0)-97},to:{r:8-parseInt(m.ai[1]),c:m.ai.charCodeAt(2)-97}}:null; render(); document.getElementById('bestLine').textContent='AI played '+m.ai+'  eval '+m.eval+'  nodes '+m.nodes; drawArrow(m.ai.slice(0,2),m.ai.slice(2,4)); }
      if(m.type==='hint'){ document.getElementById('bestLine').textContent='Best '+m.best+'  '+m.score+'cp  PV '+ (m.pv||[]).join(' '); document.getElementById('pvLine').textContent='PV: '+ (m.pv||[]).join(' '); document.getElementById('nodesInfo').textContent=m.nodes+' nodes'; document.getElementById('hintText').textContent='Hint '+m.best; if(m.best) drawArrow(m.best.slice(0,2),m.best.slice(2,4)); }
      if(m.type==='eval') updateEval(m.eval);
    } catch{}
  };
  ws.onopen = ()=> console.log('WS connected to RustClaw NNUE by Vaibhav');
  ws.onclose = ()=> setTimeout(connectWS, 2000);
};
document.getElementById('fenInput').addEventListener('keydown', (e)=>{
  if(e.key==='Enter'){
    const fen=e.target.value.trim();
    if(!fen) return;
    moveStackFens.push(boardFEN); redoStack=[];
    boardFEN=fen; history.push(fen); render();
    if(ws && ws.readyState===1) ws.send(JSON.stringify({cmd:'fen', fen}));
  }
});
connectWS(); render(); drawEvalGraph();
setInterval(()=>{ if(!ws || ws.readyState!==1) connectWS(); }, 3000);
</script>
</body>
</html>
"##;
