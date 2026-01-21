from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
import json
from typing import Any, Dict, Optional

from app.core.inference import SolarPredictor

# --- 全局变量 ---
ml_models: Dict[str, Any] = {}


# --- WebSocket 连接管理器 ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def _safe_send_text(self, ws: WebSocket, message: str):
        try:
            await ws.send_text(message)
        except Exception:
            self.disconnect(ws)

    async def _safe_send_bytes(self, ws: WebSocket, data: bytes):
        try:
            await ws.send_bytes(data)
        except Exception:
            self.disconnect(ws)

    async def broadcast(self, message: str):
        # 给所有连接发文本；失败则移除该连接，避免卡住
        for connection in list(self.active_connections):
            await self._safe_send_text(connection, message)

    async def broadcast_bytes(self, data: bytes):
        # 给所有连接发二进制；失败则移除该连接
        for connection in list(self.active_connections):
            await self._safe_send_bytes(connection, data)


manager = ConnectionManager()


# --- 生命周期 (加载 AI 模型) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models/city_sun_model.pth")
    print(f"[System] Loading model from: {model_path}")

    try:
        ml_models["predictor"] = SolarPredictor(model_path=model_path)
        ml_models["palette_ready"] = False
        print("[System] Model loaded.")
    except Exception as e:
        ml_models["predictor"] = None
        ml_models["palette_ready"] = False
        print(f"[Warning] Failed to load model: {e}")

    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

# --- CORS 设置 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- /sync_model 的数据格式 ---
class GeometryPayload(BaseModel):
    vertices: list
    faces: list


def _json_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


# --- 1) WebSocket 路由：接收 LEGEND_PALETTE ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 兼容 text / bytes
            recv = await websocket.receive()

            msg_text: Optional[str] = None
            if "text" in recv and recv["text"] is not None:
                msg_text = recv["text"]
            elif "bytes" in recv and recv["bytes"] is not None:
                # 如果你未来误发了 bytes，这里直接忽略
                continue
            else:
                continue

            # 尝试解析 JSON（前端会发 LEGEND_PALETTE）
            try:
                msg = json.loads(msg_text)
            except Exception:
                # 非 JSON：忽略
                continue

            if msg.get("type") == "LEGEND_PALETTE":
                predictor: SolarPredictor = ml_models.get("predictor")
                if predictor is None:
                    await websocket.send_text(_json_text({"type": "ERROR", "message": "AI Engine not ready"}))
                    continue

                fmt = msg.get("format", "hex")
                colors = msg.get("colors", [])
                if fmt != "hex" or (not isinstance(colors, list)) or len(colors) != 11:
                    await websocket.send_text(_json_text({
                        "type": "ERROR",
                        "message": "Invalid palette. Expect {type:'LEGEND_PALETTE', format:'hex', colors:[11 items]}."
                    }))
                    continue

                try:
                    predictor.set_palette_from_hex_list(colors)
                    ml_models["palette_ready"] = True
                    await websocket.send_text(_json_text({"type": "PALETTE_OK"}))
                    print("[System] Palette received from frontend.", flush=True)
                except Exception as e:
                    await websocket.send_text(_json_text({"type": "ERROR", "message": f"Palette set failed: {e}"}))

            # 以后加别的 WS 消息类型就在这里 elif

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# --- 2) 预测接口 (接收图片) ---
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    predictor: SolarPredictor = ml_models.get("predictor")
    if predictor is None:
        return {"error": "AI Engine not ready"}

    if not ml_models.get("palette_ready", False):
        return {"error": "Legend palette not ready. Frontend must connect and send LEGEND_PALETTE first."}

    image_bytes = await file.read()

    # predictor.predict 必须返回：analysis_png_bytes, mask_png_bytes, stats
    try:
        analysis_bytes, mask_bytes, stats = predictor.predict(image_bytes)
    except Exception as e:
        return {"error": f"Predict failed: {e}"}

    # --- WebSocket 广播：严格顺序（前端按这个顺序接收） ---
    await manager.broadcast("UPDATE_TEXTURE")

    await manager.broadcast("BLOB_TYPE:ANALYSIS")
    if isinstance(analysis_bytes, bytes) and len(analysis_bytes) > 0:
        await manager.broadcast_bytes(analysis_bytes)
    else:
        print("analysis_bytes invalid, skipped broadcast_bytes", flush=True)

    await manager.broadcast("BLOB_TYPE:MASK")
    if isinstance(mask_bytes, bytes) and len(mask_bytes) > 0:
        await manager.broadcast_bytes(mask_bytes)
    else:
        print("mask_bytes invalid, skipped broadcast_bytes", flush=True)

    await manager.broadcast(_json_text({"type": "ANALYSIS_RESULT", "data": stats}))

    # HTTP 返回：保持你现有行为（返回分析图）
    return Response(content=analysis_bytes, media_type="image/png")


# --- 3) 模型同步接口 ---
@app.post("/sync_model")
async def sync_model_endpoint(payload: GeometryPayload):
    print(f"Received geometry: {len(payload.vertices) // 3} vertices", flush=True)

    message = {"type": "GEOMETRY", "data": payload.model_dump()}
    await manager.broadcast(_json_text(message))
    return {"status": "synced"}


# --- 4) 静态文件挂载 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "../frontend")

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"[Warning] Frontend directory not found at {frontend_dir}", flush=True)
