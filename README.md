# InstantSim

**Real-time Solar Analysis Powered by AI**

InstantSim is a deep learning-based platform that provides instant daylight analysis for architectural designs. Built with a pix2pix model and microservices architecture, it transforms minute-long building performance simulations into millisecond-level predictions.

---

## Why InstantSim?

Traditional building physics simulations (Ladybug/Honeybee/OpenFOAM) are accurate but slow. InstantSim solves this by:

- **Speed**: Compress minute-level simulations to <100ms
- **Real-time Feedback**: Live sync between Rhino and web dashboard
- **Production Ready**: Containerized deployment with Docker
- **Digital Twin**: Seamless 3D geometry and texture streaming

---

## Features

- **AI-Powered Prediction**: U-Net architecture trained on synthetic Ladybug data
- **Live 3D Visualization**: Three.js-based interactive heat maps
- **Version Control**: Git-like design snapshot management and comparison
- **WebSocket Sync**: Real-time bidirectional communication with Rhino
- **Artifact Removal**: Post-processing pipeline to fix model edge cases
- **Containerized**: Docker-ready for cloud deployment

---

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- Rhino/Grasshopper (for geometry input)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/jingzhezhao/instantsim.git
cd instantsim
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download model weights**

Place your trained model at:
```
backend/models/city_sun_model.pth
```

4. **Run the server**
```bash
cd backend
uvicorn app.main:app --reload
```

5. **Open the web interface**

Navigate to `http://localhost:8000`

### Docker Deployment

```bash
docker-compose up -d
```

The service will be available at `http://localhost:8000`

---

## How It Works

### Data Flow

```
Rhino Geometry → Height Map Generation → AI Inference → Post-processing → Web Visualization
```

### Step-by-Step Process

1. **Input Layer**: Rhino/Grasshopper generates normalized height maps (0-255 grayscale)
   - Background (ground): RGB(0,0,0)
   - Buildings: Grayscale values linearly mapped to height

2. **Processing Layer**: FastAPI backend performs AI inference
   - Preprocessing: Resize to 512×512 → ToTensor → Normalize
   - Inference: U-Net forward pass (PyTorch)
   - Post-processing: Median filter + edge cropping to remove artifacts

3. **Visualization Layer**: Three.js renders 3D heat maps
   - Dual stream sync: geometry (GLTF/OBJ) + texture (PNG)
   - UV mapping: Planar projection from top-down view
   - Real-time texture updates via WebSocket

---

## Project Structure

```
instantsim/
├── backend/
│   ├── main.py                    # FastAPI server & WebSocket
│   ├── app/
│   │   ├── core/
│   │   │   └── inference.py       # AI inference engine
│   │   └── models/
│   │       └── networks.py        # U-Net architecture
│   └── models/
│       └── city_sun_model.pth     # Pre-trained weights
├── frontend/
│   └── index.html                 # Three.js web interface
├── docs/
│   ├── ARCHITECTURE.md            # Detailed system design
│   ├── DEVELOPMENT.md             # Development guide
│   └── TROUBLESHOOTING.md         # Common issues & solutions
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## API Reference

### WebSocket

**Endpoint**: `/ws`

Real-time bidirectional communication channel.

**Client → Server Messages**:
```json
{
  "type": "LEGEND_PALETTE",
  "format": "hex",
  "colors": ["#3d5a9e", "#5b9ff0", ..., "#ff5500"]
}
```

**Server → Client Messages**:
- `UPDATE_TEXTURE`: Signals new analysis started
- `BLOB_TYPE:ANALYSIS`: Next blob is analysis image
- `BLOB_TYPE:MASK`: Next blob is mask image
- Analysis results (JSON with statistics)

### HTTP Endpoints

**POST /predict**

Upload semantic render and get solar analysis.

Request:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@input.png"
```

Response:
- `Content-Type: image/png`
- Body: Color-mapped solar prediction image

**POST /sync_model**

Sync 3D geometry from Rhino to web clients.

Request:
```json
{
  "vertices": [x1, y1, z1, x2, y2, z2, ...],
  "faces": [i1, i2, i3, i4, i5, i6, ...]
}
```

Response:
```json
{
  "status": "synced"
}
```

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI | High-performance async web framework |
| **AI Framework** | PyTorch | Deep learning inference |
| **Model** | U-Net (pix2pix) | Image-to-image translation |
| **Communication** | WebSockets | Real-time bidirectional sync |
| **Frontend** | Three.js | 3D visualization |
| **Deployment** | Docker | Containerization |
| **Database** | SQLite/PostgreSQL | Version control & metadata |

---

## Color System

InstantSim uses a gradient from deep blue (low sun exposure) to orange-red (high sun exposure):

```
0h  → #3d5a9e (Deep Blue)
2h  → #5b9ff0 (Bright Blue)
4h  → #90d8ff (Sky Blue)
6h  → #fff066 (Yellow)
8h  → #ffaa00 (Orange)
10h → #ff5500 (Orange-Red)
```

The palette is fully customizable via the frontend and synchronized with the backend for accurate statistical analysis.

---

## Statistics & Metrics

InstantSim calculates:

- **Average Daylight**: Mean solar hours across valid surfaces
- **Low Sun (<2h)**: Percentage of under-lit areas
- **Solar Potential (>6h)**: Percentage suitable for photovoltaic panels
- **Hourly Distribution**: Breakdown by sun exposure bins (0-10h)

All metrics are computed in Lab color space using KNN classification for improved perceptual accuracy.

---

## Post-processing Pipeline

The system includes automatic artifact removal to handle edge cases:

| Issue | Cause | Solution |
|-------|-------|----------|
| Red dot matrix | Dropout + BatchNorm instability | Median blur filter (3×3 kernel) |
| Blue border artifacts | Convolution padding effects | Crop outer 2px edges |
| Missing low buildings | Input contrast issues | Adjust height mapping range |
| Texture misalignment | Center coordinate mismatch | Bounding box center alignment |

---

## Development

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- Setting up development environment
- Training the model
- Data generation pipeline
- Running tests

---

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for:
- Detailed system design
- MLOps pipeline
- Coordinate system transformations
- Performance optimization strategies

---

## Troubleshooting

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues and solutions.

---

## Roadmap

### Phase 1: Data Engineering
- [x] Grasshopper automated screenshot script
- [x] Generate 1000 training pairs
- [x] Data preprocessing pipeline

### Phase 2: Model Training
- [x] PyTorch training environment
- [x] Data augmentation (rotation, flip)
- [x] Export trained weights (.pth)

### Phase 3: Backend Development
- [x] FastAPI project initialization
- [x] WebSocket broadcast service
- [x] Dockerization

### Phase 4: Integration
- [x] Web dashboard development
- [x] Rhino → Backend → Web pipeline
- [ ] Performance optimization
- [ ] Cloud deployment

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

[Your License Here]

---

## Acknowledgments

- Deep learning model based on [pix2pix](https://phillipi.github.io/pix2pix/)
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [Three.js](https://threejs.org/)
- Solar simulation data generated with [Ladybug Tools](https://www.ladybug.tools/)

---

## Contact

For questions or feedback, please open an issue or contact [zhao1303@yahoo.com]

---

**Made for architects and designers by architects**