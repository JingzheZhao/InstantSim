# System Architecture

**Project**: InstantSim  
**Version**: 1.0.0  
**Status**: In Development  
**Last Updated**: 2026-01-20

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Data Flow](#data-flow)
3. [Core Components](#core-components)
4. [Color Remapping System](#color-remapping-system)
5. [Coordinate System Transformations](#coordinate-system-transformations)
6. [Performance Optimization](#performance-optimization)
7. [MLOps Pipeline](#mlops-pipeline)

---

## High-Level Architecture

InstantSim follows a classic **B/S (Browser/Server) + Agent** architecture with three decoupled layers:

```
┌─────────────────────┐
│   Rhino/GH Client   │  Edge Layer (Data Collection)
│  (Geometry Input)   │
└──────────┬──────────┘
           │ HTTP POST / WebSocket
           ▼
┌─────────────────────┐
│   FastAPI Backend   │  Core Layer (Computation)
│  + PyTorch Engine   │
│  + Docker Container │
└──────────┬──────────┘
           │ WebSocket Broadcast
           ▼
┌─────────────────────┐
│  Three.js Frontend  │  View Layer (Visualization)
│  (Web Dashboard)    │
└─────────────────────┘
```

### Layer Responsibilities

**1. Data Collection Layer (The Edge)**
- Rhino/Grasshopper client
- Geometry generation
- Height map extraction via raycasting
- Data transmission

**2. Computation Service Layer (The Core)**
- Python backend in Docker
- Image preprocessing
- AI inference (PyTorch)
- Post-processing & artifact removal
- Business logic dispatch

**3. User Presentation Layer (The View)**
- Web dashboard
- 3D heat map rendering
- Chart visualization (ECharts)
- Version control UI

---

## Data Flow

### Complete Pipeline

```
1. GEOMETRY GENERATION (Rhino)
   ├─> Create/modify 3D model
   ├─> Trigger change event (EndCommand/Idle)
   └─> Generate height map (raycasting)
       ├─> Z=0m (ground) → pixel value 0 (black)
       └─> Z=100m (max height) → pixel value 255 (white)

2. TRANSMISSION (HTTP/WebSocket)
   ├─> Convert bitmap to bytes
   ├─> POST /predict (height map image)
   └─> Establish WebSocket connection

3. AI INFERENCE (Backend)
   ├─> Preprocessing
   │   ├─> Resize to 512×512
   │   ├─> ToTensor conversion
   │   └─> Normalize [-1, 1]
   ├─> Model forward pass (U-Net)
   ├─> Post-processing
   │   ├─> Denoise (median blur)
   │   ├─> Edge crop (remove 2px border)
   │   └─> Color remapping
   └─> Statistical analysis (Lab color space KNN)

4. SYNCHRONIZATION (WebSocket Broadcast)
   ├─> Send "UPDATE_TEXTURE" signal
   ├─> Send analysis image (BLOB_TYPE:ANALYSIS)
   ├─> Send mask image (BLOB_TYPE:MASK)
   └─> Send statistics (JSON)

5. VISUALIZATION (Three.js)
   ├─> Receive geometry stream (vertices + faces)
   ├─> Receive texture stream (analysis PNG)
   ├─> Apply UV mapping (planar projection)
   ├─> Render 3D scene with heat map
   └─> Update KPI dashboard
```

---

## Core Components

### 1. Height Map Generator (Rhino/GH)

**Purpose**: Convert 3D geometry to 2D grayscale image without viewport capture.

**Algorithm**:
```python
# Pseudo-code
grid = create_grid(256, 256, bbox)
for each_point in grid:
    ray = create_ray(point, direction=+Z)
    intersection = ray_cast(ray, geometry)
    if intersection:
        height = intersection.z
        pixel_value = normalize(height, 0, 100) * 255
    else:
        pixel_value = 0  # ground
```

**Advantages**:
- No lighting/shadow interference
- Consistent resolution
- Fast computation (pure geometry calculation)

### 2. AI Inference Engine

**Model**: U-Net Generator (pix2pix architecture)

**Configuration**:
```python
UnetGenerator(
    input_nc=3,      # RGB input (grayscale replicated)
    output_nc=3,     # RGB output (color heat map)
    num_downs=8,     # 8 downsampling layers
    ngf=64,          # Generator filters
    norm_layer=nn.BatchNorm2d,
    use_dropout=True
)
```

**Training Data**:
- **Input (X)**: Normalized height maps
- **Label (Y)**: Ladybug annual radiation heat maps
- **Dataset Size**: 1000+ synthetic pairs
- **Augmentation**: Rotation, flip, brightness jitter

**Inference Optimization**:
- Model: `.eval()` mode
- No gradients: `torch.no_grad()`
- CPU optimization: Single-threaded inference
- GPU option: CUDA support available

### 3. Post-processing Pipeline

**Artifact Removal**:

```python
def remove_artifacts(output_img):
    # 1. Median blur to remove red/blue noise dots
    denoised = cv2.medianBlur(output_img, 3)
    
    # 2. Crop outer 2px to remove blue border
    cropped = denoised[2:-2, 2:-2]
    
    # 3. Resize back to original dimension
    final = cv2.resize(cropped, (512, 512))
    
    return final
```

**Known Edge Cases**:
- **Red dot matrix**: Caused by Dropout during test time
- **Blue border**: Convolution padding artifacts
- **Missing low buildings**: Input contrast too low

### 4. Color Remapping System

See dedicated section below.

### 5. WebSocket Manager

**Connection Pool**:
```python
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
    
    async def broadcast_bytes(self, data: bytes):
        for connection in self.active_connections:
            await connection.send_bytes(data)
```

**Benefits**:
- One-to-many broadcasting
- Automatic cleanup on disconnect
- Error handling per connection

---

## Color Remapping System

**Challenge**: Model was trained with one color palette but needs to support custom frontend palettes.

**Solution**: Two-stage mapping in Lab color space.

### Stage 1: Training Palette

Original gradient used during training:
```python
original_palette = [
    "#0000aa",  # 0h - Dark blue
    "#2a0098",  # 1h
    "#4b0082",  # 2h - Indigo
    "#a00040",  # 3h
    "#d40020",  # 4h
    "#ff0000",  # 5h - Red
    "#ff4500",  # 6h
    "#ff8c00",  # 7h
    "#ffa500",  # 8h - Orange
    "#ffd700",  # 9h - Gold
    "#ffff00"   # 10h - Yellow
]
```

### Stage 2: Frontend Palette

New gradient defined in frontend:
```python
frontend_palette = [
    "#3d5a9e",  # 0h - Deep blue
    "#5b9ff0",  # 2h - Bright blue
    "#90d8ff",  # 4h - Sky blue
    "#fff066",  # 6h - Yellow
    "#ffaa00",  # 8h - Orange
    "#ff5500"   # 10h - Orange-red
]
```

### Remapping Algorithm

```python
def remap_colors(output_bgr, valid_mask):
    # 1. Convert output to Lab color space
    output_lab = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2LAB)
    
    # 2. For each valid pixel, find nearest color in ORIGINAL palette
    valid_pixels = output_lab[valid_mask]
    distances = compute_distances(valid_pixels, original_palette_lab)
    nearest_indices = argmin(distances)  # Range: 0-10
    
    # 3. Map to FRONTEND palette using same index
    new_colors = frontend_palette_bgr[nearest_indices]
    
    # 4. Enhance saturation (1.4× boost)
    hsv = cv2.cvtColor(new_colors, cv2.COLOR_BGR2HSV)
    hsv[:, 1] = clip(hsv[:, 1] * 1.4, 0, 255)
    remapped_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return remapped_bgr
```

**Why Lab Space?**
- Perceptually uniform (Euclidean distance ≈ visual difference)
- More robust than RGB/HSV for color matching
- Industry standard for color science

---

## Coordinate System Transformations

### Rhino vs Three.js Coordinate Systems

| System | Up Axis | Handedness | Origin |
|--------|---------|------------|--------|
| Rhino | Z-up | Right-handed | User-defined |
| Three.js | Y-up | Right-handed | Scene center |

### Transformation Matrix

```javascript
// Rhino Z-up to Three.js Y-up
mainGroup.rotateX(-Math.PI / 2);

// Before: Rhino (X, Y, Z)
// After: Three.js (X, -Z, Y)
```

### UV Mapping Strategy

**Challenge**: Texture coordinates from Rhino don't align with Three.js expectations.

**Solution**: Compute UVs from global bounding box:

```javascript
function computeUVs(geometry) {
    const bbox = geometry.boundingBox;
    const rangeX = bbox.max.x - bbox.min.x;
    const rangeY = bbox.max.y - bbox.min.y;
    
    const uvs = [];
    for (let i = 0; i < vertices.length; i += 3) {
        const x = vertices[i];
        const y = vertices[i + 1];
        
        const u = (x - bbox.min.x) / rangeX;
        const v = (y - bbox.min.y) / rangeY;
        
        uvs.push(u, v);
    }
    
    return uvs;
}
```

**Texture Flipping**:

```javascript
// Handle Y-axis flip in texture coordinate system
analysisTexture.repeat.set(uSpan, -vSpan);  // Negative Y!
analysisTexture.offset.set(u0, v1);         // Start from top
```

---

## Performance Optimization

### Backend Optimizations

**1. Model Loading**
- Load once at startup (lifespan context)
- Keep in memory for sub-100ms inference
- Support both CPU and GPU

**2. Image Processing**
- Use OpenCV for speed (10× faster than PIL)
- Vectorized operations (NumPy)
- Avoid Python loops

**3. WebSocket Efficiency**
- Binary transmission for images (not Base64)
- Text transmission for JSON metadata
- Automatic connection cleanup

### Frontend Optimizations

**1. Three.js Rendering**
- Use `logarithmicDepthBuffer` for large scenes
- Enable shadow map caching
- Limit pixel ratio to 2× (avoid 4K overkill)

**2. Texture Management**
- Use mipmaps for distant views
- Set `generateMipmaps: true`
- Linear filtering for smooth gradients

**3. UI Responsiveness**
- Debounce WebSocket messages (avoid UI thrashing)
- Progressive loading indicators
- Virtual scrolling for large datasets

---

## MLOps Pipeline

### Development Workflow

```
1. DATA COLLECTION
   ├─> Grasshopper automation scripts
   ├─> Generate 1000+ training pairs
   └─> Store in structured folders

2. TRAINING
   ├─> PyTorch training loop
   ├─> Validation split (80/20)
   ├─> Checkpoint saving (every 10 epochs)
   └─> Export final .pth weights

3. DEPLOYMENT
   ├─> Dockerize FastAPI backend
   ├─> Mount model weights as volume
   ├─> Environment variable configuration
   └─> Docker Compose orchestration

4. MONITORING
   ├─> Log inference times
   ├─> Track model accuracy metrics
   └─> User feedback collection
```

### Docker Architecture

**Dockerfile Strategy**:
```dockerfile
# Multi-stage build for smaller image
FROM python:3.9-slim AS base

# Layer 1: System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Layer 2: Python dependencies (cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 3: Application code
COPY . /app
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Volume Mounts**:
```yaml
# docker-compose.yml
volumes:
  - ./models:/app/models          # Model weights
  - ./debug:/app/debug            # Debug outputs
  - ./data/snapshots:/app/data    # Version control
```

### CI/CD Pipeline (Conceptual)

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Lint
        run: flake8 .
      - name: Unit Tests
        run: pytest tests/
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker Image
        run: docker build -t instantsim:latest .
      - name: Push to Registry
        run: docker push instantsim:latest
```

---

## Security Considerations

**1. Input Validation**
- Limit uploaded image size (max 10MB)
- Validate file types (PNG/JPEG only)
- Sanitize filenames

**2. Resource Limits**
- Docker memory limits (e.g., 2GB)
- Request rate limiting (100 req/hour)
- WebSocket connection limits

**3. Data Privacy**
- No persistent storage of user geometry
- Debug files auto-deleted after 24h
- Optional anonymous analytics

---

## Scalability

**Horizontal Scaling**:
- Deploy multiple backend containers
- Use load balancer (Nginx/Traefik)
- Redis for WebSocket session management

**Vertical Scaling**:
- GPU inference (10× speedup)
- Batch processing support
- Model quantization (INT8)

---

## Future Enhancements

**Short-term**:
- [ ] GPU acceleration
- [ ] Multi-model support (different climates)
- [ ] Export to PDF reports

**Long-term**:
- [ ] Real-time collaborative editing
- [ ] Cloud-native deployment (Kubernetes)
- [ ] Mobile app support
- [ ] Integration with BIM tools (Revit API)

---

**Document Version**: 1.0.0  
**Last Updated**: 2026-01-20