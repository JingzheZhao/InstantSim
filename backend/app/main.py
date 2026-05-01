from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
import json
from typing import Any, Dict, Optional

# Try importing the inference engine
try:
    from app.core.inference import SolarPredictor
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from app.core.inference import SolarPredictor

# --- Global state ---
ml_models: Dict[str, Any] = {}


# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] New client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WS] Client disconnected. Remaining connections: {len(self.active_connections)}")

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


# --- Core helper: run inference and broadcast results ---
async def run_inference_and_broadcast(image_bytes: bytes):
    """
    Encapsulates the full inference pipeline, shared by HTTP requests
    and WebSocket-triggered re-computations on cached images.
    """
    predictor = ml_models.get("predictor")
    # Check system readiness
    if not (predictor and ml_models.get("palette_ready") and ml_models.get("sun_vectors_ready")):
        return None, None, None

    try:
        # Run inference
        analysis_bytes, mask_bytes, stats = predictor.predict(image_bytes)

        # Broadcast in the strict order required by the frontend
        await manager.broadcast("UPDATE_TEXTURE")
        await manager.broadcast("BLOB_TYPE:ANALYSIS")
        await manager.broadcast_bytes(analysis_bytes)
        await manager.broadcast("BLOB_TYPE:MASK")
        await manager.broadcast_bytes(mask_bytes)
        await manager.broadcast(json.dumps({"type": "ANALYSIS_RESULT", "data": stats}, ensure_ascii=False))

        return analysis_bytes, mask_bytes, stats
    except Exception as e:
        print(f"[Error] Inference or broadcast failed: {e}")
        return None, None, None


# --- Application lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models/city_sun_model.pth")
    debug_dir = os.path.join(current_dir, "../debug_outputs")
    os.makedirs(debug_dir, exist_ok=True)

    try:
        predictor = SolarPredictor(model_path=model_path)
        predictor.debug_dir = debug_dir  # Pass debug directory to inference engine
        ml_models["predictor"] = predictor
        ml_models["palette_ready"] = False
        ml_models["sun_vectors_ready"] = False
        ml_models["last_image"] = None  # Cache for the last image received from Grasshopper
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


# --- 1) WebSocket route ---
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

            # Handle legend palette
            if msg.get("type") == "LEGEND_PALETTE":
                predictor = ml_models.get("predictor")
                colors = msg.get("colors", [])
                if predictor and len(colors) == 9:  # 9-hour model
                    predictor.set_palette_from_hex_list(colors)
                    ml_models["palette_ready"] = True
                    await websocket.send_text(json.dumps({"type": "PALETTE_OK"}))
                    print("[System] Palette ready.")

            # Handle sun vector update (triggered by frontend slider)
            elif msg.get("type") == "UPDATE_SIMULATION":
                predictor = ml_models.get("predictor")
                backend_data = msg.get("sun_vectors") or msg.get("backendData")

                if predictor and backend_data and len(backend_data) == 36:
                    predictor.set_sun_vectors(backend_data)
                    ml_models["sun_vectors_ready"] = True
                    await websocket.send_text(json.dumps({"type": "BACKEND_DATA_OK"}))

                    # Re-run inference with cached image after sun vectors are updated
                    if ml_models.get("last_image"):
                        print("[System] New sun vectors received. Re-computing with cached image...")
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


# --- 2) Predict endpoint (triggered by Grasshopper) ---
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    print(f"\n======== Received Grasshopper push: {file.filename} ========")

    # Read and cache the image
    contents = await file.read()
    ml_models["last_image"] = contents

    # Check system readiness
    if not (ml_models.get("predictor") and ml_models.get("palette_ready") and ml_models.get("sun_vectors_ready")):
        print("[Rejected] System not ready.")
        return JSONResponse(status_code=400, content={"error": "System not ready"})

    # Run inference and broadcast to frontend
    analysis_bytes, _, stats = await run_inference_and_broadcast(contents)

    if analysis_bytes:
        print(f"[OK] Inference succeeded. Stats: {stats}")
        return Response(content=analysis_bytes, media_type="image/png")
    else:
        return JSONResponse(status_code=500, content={"error": "Inference failed"})


# --- 3) Geometry sync endpoint ---
@app.post("/sync_model")
async def sync_model_endpoint(payload: GeometryPayload):
    print(f"[Sync] Geometry sync: {len(payload.vertices) // 3} vertices")
    await manager.broadcast(json.dumps({"type": "GEOMETRY", "data": payload.model_dump()}))
    return {"status": "synced"}


# --- 4) Static file mount ---
current_dir_global = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.normpath(os.path.join(current_dir_global, "../frontend"))
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print(f"[System] Serving frontend from: {frontend_path}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)