# Troubleshooting Guide

Common issues and their solutions for InstantSim.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Model Inference Problems](#model-inference-problems)
3. [Visual Artifacts](#visual-artifacts)
4. [WebSocket Connection Issues](#websocket-connection-issues)
5. [Three.js Rendering Problems](#threejs-rendering-problems)
6. [Performance Issues](#performance-issues)

---

## Installation Issues

### PyTorch Installation Fails

**Problem**: `pip install torch` fails or installs incompatible version.

**Solution**:

```bash
# For CPU-only (recommended for development)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### OpenCV Import Error

**Problem**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### FastAPI Port Already in Use

**Problem**: `Address already in use: Port 8000`

**Solution**:

```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
uvicorn main:app --port 8001
```

---

## Model Inference Problems

### Model File Not Found

**Problem**: `FileNotFoundError: models/city_sun_model.pth`

**Solution**:

1. Check file exists:
```bash
ls -lh backend/models/city_sun_model.pth
```

2. Verify file path in code:
```python
# In inference.py
model_path = os.path.join(os.path.dirname(__file__), "../models/city_sun_model.pth")
print(f"Looking for model at: {os.path.abspath(model_path)}")
```

3. If missing, download/train model or adjust path.

### Model Loading Error

**Problem**: `RuntimeError: Error(s) in loading state_dict`

**Possible Causes**:
1. Model architecture mismatch
2. Corrupted .pth file
3. PyTorch version incompatibility

**Solution**:

```python
# Check model structure
checkpoint = torch.load("models/city_sun_model.pth", map_location="cpu")
print(checkpoint.keys())

# Try loading with weights_only
state_dict = torch.load("models/city_sun_model.pth", 
                        map_location="cpu",
                        weights_only=True)

# Remove "module." prefix if present (from DataParallel)
if list(state_dict.keys())[0].startswith("module."):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
```

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:

1. Use CPU inference:
```python
device = torch.device("cpu")
```

2. Reduce batch size:
```python
# In training
batch_size = 1
```

3. Clear cache:
```python
torch.cuda.empty_cache()
```

4. Use smaller image size:
```python
# Resize to 256×256 instead of 512×512
```

---

## Visual Artifacts

### Red Dot Matrix

**Symptom**: Scattered red pixels across the output image.

**Cause**: Dropout layers active during inference + BatchNorm instability.

**Solution**:

Already implemented in `inference.py`:
```python
# Post-processing pipeline
denoised = cv2.medianBlur(output_img, 3)  # Removes noise
```

If still present, increase kernel size:
```python
denoised = cv2.medianBlur(output_img, 5)  # Stronger filtering
```

### Blue Border Artifacts

**Symptom**: Blue/cyan lines around image edges.

**Cause**: Convolution padding introduces edge artifacts.

**Solution**:

Already implemented:
```python
# Crop outer 2 pixels
cropped = output_img[2:-2, 2:-2]
resized = cv2.resize(cropped, (512, 512))
```

If still visible, increase crop:
```python
cropped = output_img[4:-4, 4:-4]  # Remove 4px border
```

### Missing Low Buildings

**Symptom**: Short buildings (< 20m) don't appear in output.

**Possible Causes**:
1. Input height map contrast too low
2. Model trained on taller buildings only
3. Normalization range mismatch

**Solution**:

1. Check input height mapping:
```python
# Ensure proper normalization
max_height = 100  # meters
pixel_value = (building_height / max_height) * 255

# For shorter buildings, use smaller max_height
max_height = 50  # Better contrast for low-rise
```

2. Adjust brightness:
```python
# Before inference
enhanced = cv2.convertScaleAbs(input_img, alpha=1.2, beta=10)
```

3. Retrain model with low-rise examples in dataset.

### Color Banding

**Symptom**: Visible color steps instead of smooth gradient.

**Cause**: Limited color palette or quantization.

**Solution**:

1. Increase palette size (requires retraining):
```python
# From 11 colors (0-10h) to 21 colors (0-20h half-hour bins)
```

2. Add dithering:
```python
# After color remapping
dithered = cv2.addWeighted(remapped, 0.95, noise, 0.05, 0)
```

3. Use higher bit depth:
```python
# Save as 16-bit PNG instead of 8-bit
cv2.imwrite("output.png", output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
```

---

## WebSocket Connection Issues

### Connection Refused

**Problem**: WebSocket fails to connect.

**Diagnostic**:
```javascript
// Check browser console
ws = new WebSocket("ws://localhost:8000/ws");
ws.onerror = (e) => console.error("WS Error:", e);
```

**Solution**:

1. Verify server is running:
```bash
curl http://localhost:8000/
```

2. Check firewall:
```bash
# Allow port 8000
sudo ufw allow 8000  # Linux
```

3. Use correct protocol:
```javascript
// HTTP site
const ws = new WebSocket("ws://" + window.location.host + "/ws");

// HTTPS site
const ws = new WebSocket("wss://" + window.location.host + "/ws");
```

### Frequent Disconnections

**Problem**: WebSocket drops connection every few seconds.

**Cause**: Proxy timeout, idle timeout, or network issues.

**Solution**:

1. Implement ping/pong keepalive:
```python
# Backend
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.send_text("ping")
            await asyncio.sleep(30)  # Keepalive every 30s
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

```javascript
// Frontend
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send("pong");
    }
}, 30000);
```

2. Handle reconnection:
```javascript
let reconnectInterval;

function connect() {
    ws = new WebSocket("ws://" + window.location.host + "/ws");
    
    ws.onclose = () => {
        console.log("Disconnected, reconnecting...");
        reconnectInterval = setTimeout(connect, 3000);
    };
    
    ws.onopen = () => {
        clearTimeout(reconnectInterval);
    };
}
```

### Binary Data Corruption

**Problem**: Received images are corrupted or display incorrectly.

**Solution**:

1. Ensure correct blob type:
```javascript
ws.binaryType = "blob";  // Not "arraybuffer"
```

2. Check BLOB_TYPE header:
```javascript
ws.onmessage = async (event) => {
    if (event.data instanceof Blob) {
        console.log("Blob received, type:", nextBlobType);
        // Process based on nextBlobType
    }
};
```

3. Verify PNG encoding:
```python
# Backend
ok, encoded = cv2.imencode(".png", image)
if not ok:
    raise RuntimeError("PNG encoding failed")
```

---

## Three.js Rendering Problems

### Texture Not Appearing

**Problem**: 3D model visible but heat map texture missing.

**Diagnostic**:
```javascript
console.log("Texture loaded:", !!analysisTexture);
console.log("Material map:", analysisMesh.material.map);
console.log("Has UV:", !!analysisMesh.geometry.attributes.uv);
```

**Solution**:

1. Check UV attributes exist:
```javascript
if (!geometry.attributes.uv) {
    console.error("Missing UV coordinates!");
    // Generate UVs from bounding box
}
```

2. Verify texture settings:
```javascript
texture.needsUpdate = true;
texture.flipY = false;  // Important!
material.needsUpdate = true;
```

3. Check render order:
```javascript
analysisMesh.renderOrder = 999;  // Render last
```

### Texture Misalignment

**Problem**: Heat map appears shifted or rotated.

**Cause**: UV coordinate mismatch or center offset.

**Solution**:

1. Verify bounding box alignment:
```javascript
const bbox = new THREE.Box3().setFromObject(geometry);
const center = new THREE.Vector3();
bbox.getCenter(center);

// Ensure texture center matches geometry center
```

2. Check texture repeat/offset:
```javascript
console.log("Repeat:", texture.repeat);
console.log("Offset:", texture.offset);

// Should match mask rect UV
texture.repeat.set(uSpan, -vSpan);  // Negative Y for flip
texture.offset.set(u0, v1);
```

3. Debug with test texture:
```javascript
// Load UV checker pattern
const testTexture = new THREE.TextureLoader().load("uv_checker.png");
material.map = testTexture;
```

### Black Screen / Nothing Renders

**Problem**: Canvas is blank.

**Solution**:

1. Check camera position:
```javascript
console.log("Camera:", camera.position);
console.log("Looking at:", controls.target);

// Reset camera
camera.position.set(100, 200, 100);
camera.lookAt(0, 0, 0);
```

2. Verify lights:
```javascript
scene.children.forEach(child => {
    if (child.isLight) console.log("Light:", child.type, child.intensity);
});
```

3. Check renderer:
```javascript
console.log("WebGL supported:", !!document.createElement('canvas').getContext('webgl'));
renderer.info.render.calls  // Should be > 0
```

4. Look for errors:
```javascript
renderer.debug.checkShaderErrors = true;
```

---

## Performance Issues

### Slow Inference (>1 second)

**Problem**: AI prediction takes too long.

**Diagnostic**:
```python
import time

start = time.time()
output = model(input_tensor)
print(f"Inference time: {time.time() - start:.3f}s")
```

**Solution**:

1. Use GPU if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

2. Optimize model:
```python
# Disable gradient computation
with torch.no_grad():
    output = model(input)

# Use half precision (if GPU supports)
model.half()
input_tensor = input_tensor.half()
```

3. Reduce image size:
```python
# Use 256×256 instead of 512×512
resized = cv2.resize(image, (256, 256))
```

### High Memory Usage

**Problem**: Backend uses >2GB RAM.

**Solution**:

1. Clear unused variables:
```python
del output_tensor
torch.cuda.empty_cache()  # If using GPU
import gc
gc.collect()
```

2. Limit debug file storage:
```python
# Don't save debug files in production
if os.environ.get("DEBUG") != "1":
    return  # Skip debug saving
```

### Laggy Frontend

**Problem**: 3D rendering stutters or drops frames.

**Solution**:

1. Reduce render quality:
```javascript
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));  // Cap at 2×
```

2. Disable shadows if not needed:
```javascript
renderer.shadowMap.enabled = false;
```

3. Throttle updates:
```javascript
let updateTimeout;
ws.onmessage = (event) => {
    clearTimeout(updateTimeout);
    updateTimeout = setTimeout(() => {
        processMessage(event.data);
    }, 100);  // Debounce 100ms
};
```

4. Use LOD (Level of Detail):
```javascript
const lod = new THREE.LOD();
lod.addLevel(highDetailMesh, 0);
lod.addLevel(mediumDetailMesh, 50);
lod.addLevel(lowDetailMesh, 100);
```

---

## Getting More Help

If your issue isn't listed here:

1. **Check logs**:
```bash
# Backend: the Uvicorn process prints inference logs to stdout

# Browser
Open DevTools → Console tab
```

2. **Enable debug mode**:
```bash
export DEBUG=1
uvicorn main:app --reload --log-level debug
```

3. **Create an issue** on GitHub with:
   - Error message
   - Steps to reproduce
   - System info (OS, Python version, browser)
   - Relevant logs

4. **Contact**: support@instantsim.com

---

**Last Updated**: 2026-01-20