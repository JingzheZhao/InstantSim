from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
import json
from typing import Any, Dict, Optional

# 尝试导入推理引擎
try:
    from app.core.inference import SolarPredictor
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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
        print(f"[WS] 新客户端连接。当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WS] 客户端断开。剩余连接数: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

    async def broadcast_bytes(self, data: bytes):
        for connection in list(self.active_connections):
            try:
                await connection.send_bytes(data)
            except:
                self.disconnect(connection)


manager = ConnectionManager()


# --- 核心辅助函数：执行推理并广播结果 ---
async def run_inference_and_broadcast(image_bytes: bytes):
    """
    封装推理流程，供 HTTP 请求和 WebSocket 自动重算共用
    """
    predictor = ml_models.get("predictor")
    # 检查系统状态
    if not (predictor and ml_models.get("palette_ready") and ml_models.get("sun_vectors_ready")):
        return None, None, None

    try:
        # 执行推理
        analysis_bytes, mask_bytes, stats = predictor.predict(image_bytes)

        # 按照前端要求的严格顺序广播
        await manager.broadcast("UPDATE_TEXTURE")
        await manager.broadcast("BLOB_TYPE:ANALYSIS")
        await manager.broadcast_bytes(analysis_bytes)
        await manager.broadcast("BLOB_TYPE:MASK")
        await manager.broadcast_bytes(mask_bytes)
        await manager.broadcast(json.dumps({"type": "ANALYSIS_RESULT", "data": stats}, ensure_ascii=False))

        return analysis_bytes, mask_bytes, stats
    except Exception as e:
        print(f"[Error] 推理或广播失败: {e}")
        return None, None, None


# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models/city_sun_model.pth")
    debug_dir = os.path.join(current_dir, "../debug_outputs")
    os.makedirs(debug_dir, exist_ok=True)

    try:
        predictor = SolarPredictor(model_path=model_path)
        predictor.debug_dir = debug_dir  # 传递目录给推理引擎
        ml_models["predictor"] = predictor
        ml_models["palette_ready"] = False
        ml_models["sun_vectors_ready"] = False
        ml_models["last_image"] = None  # 🌟 用于缓存 Grasshopper 发来的最后一张图
        print("[System] Model and Debug environment ready.")
    except Exception as e:
        print(f"[Critical] Initialization failed: {e}")
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GeometryPayload(BaseModel):
    vertices: list
    faces: list


# --- 1) WebSocket 路由 ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            recv = await websocket.receive_text()
            try:
                msg = json.loads(recv)
            except:
                continue

            # 处理调色板
            if msg.get("type") == "LEGEND_PALETTE":
                predictor = ml_models.get("predictor")
                colors = msg.get("colors", [])
                if predictor and len(colors) == 9:  # 适配 9 小时模型
                    predictor.set_palette_from_hex_list(colors)
                    ml_models["palette_ready"] = True
                    await websocket.send_text(json.dumps({"type": "PALETTE_OK"}))
                    print("[System] Palette ready.")

            # 🌟 处理太阳向量更新 (前端拖动滑块)
            elif msg.get("type") == "UPDATE_SIMULATION":
                predictor = ml_models.get("predictor")
                backend_data = msg.get("sun_vectors") or msg.get("backendData")

                if predictor and backend_data and len(backend_data) == 36:
                    predictor.set_sun_vectors(backend_data)
                    ml_models["sun_vectors_ready"] = True
                    await websocket.send_text(json.dumps({"type": "BACKEND_DATA_OK"}))

                    # ✅ 成功更新向量后，再用缓存图片立即重算
                    if ml_models.get("last_image"):
                        print("[System] 收到新向量，正在使用缓存图片实时重算...")
                        await run_inference_and_broadcast(ml_models["last_image"])
                else:
                    await websocket.send_text(json.dumps({
                        "type": "BACKEND_DATA_ERROR",
                        "message": f"sun_vectors must be length 36 (27+9 mask). Got: {len(backend_data) if backend_data else None}"
                    }))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS Error] {e}")
        manager.disconnect(websocket)


# --- 2) 预测接口 (由 Grasshopper 触发) ---
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    print(f"\n======== 收到 Grasshopper 推送: {file.filename} ========")

    # 读取并缓存图片
    contents = await file.read()
    ml_models["last_image"] = contents

    # 状态检查
    if not (ml_models.get("predictor") and ml_models.get("palette_ready") and ml_models.get("sun_vectors_ready")):
        print("❌ [拒绝请求] 系统状态未就绪")
        return JSONResponse(status_code=400, content={"error": "System not ready"})

    # 执行推理并发给前端
    analysis_bytes, _, stats = await run_inference_and_broadcast(contents)

    if analysis_bytes:
        print(f"✅ 推理成功并发回前端. Stats: {stats}")
        return Response(content=analysis_bytes, media_type="image/png")
    else:
        return JSONResponse(status_code=500, content={"error": "Inference failed"})


# --- 3) 模型同步接口 ---
@app.post("/sync_model")
async def sync_model_endpoint(payload: GeometryPayload):
    print(f"[Sync] 几何体同步: {len(payload.vertices) // 3} 顶点")
    await manager.broadcast(json.dumps({"type": "GEOMETRY", "data": payload.model_dump()}))
    return {"status": "synced"}


# --- 4) 静态文件挂载 (修正作用域) ---
current_dir_global = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.normpath(os.path.join(current_dir_global, "../frontend"))
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print(f"[System] 托管前端于: {frontend_path}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)