# Development Guide

This guide covers setting up your development environment, training the model, and contributing to InstantSim.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Generation Pipeline](#data-generation-pipeline)
3. [Model Training](#model-training)
4. [Running Tests](#running-tests)
5. [Development Workflow](#development-workflow)
6. [Code Style Guide](#code-style-guide)

---

## Environment Setup

### Prerequisites

- Python 3.9 or higher
- pip or conda
- Git
- Docker (optional, for containerized development)
- Rhino 7+ with Grasshopper (for data generation)

### Clone Repository

```bash
git clone https://github.com/yourusername/instantsim.git
cd instantsim
```

### Create Virtual Environment

**Using venv**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda**:
```bash
conda create -n instantsim python=3.9
conda activate instantsim
```

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools
```

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
uvicorn --version
```

---

## Data Generation Pipeline

### Overview

The training data consists of paired images:
- **Input (X)**: Normalized height maps (0-255 grayscale)
- **Label (Y)**: Ladybug solar radiation heat maps (RGB color)

### Grasshopper Automation Script

**Location**: `data_generation/gh_automation.py`

**Key Steps**:

1. **Generate Random Geometry**
```python
# Pseudo-code
def generate_building():
    base_size = random(20, 50)  # meters
    height = random(10, 100)    # meters
    position = random_point_in_grid()
    
    building = create_box(base_size, base_size, height)
    place_at(building, position)
    return building
```

2. **Create Height Map**
```python
def create_height_map(geometry, resolution=512):
    grid = create_grid(resolution, bbox)
    height_map = np.zeros((resolution, resolution), dtype=np.uint8)
    
    for i, point in enumerate(grid):
        ray = Ray(point, Vector3d.ZAxis)
        intersection = ray_shoot(ray, geometry)
        
        if intersection:
            height = intersection.z
            normalized = int((height / max_height) * 255)
            height_map[i] = normalized
    
    return height_map
```

3. **Run Ladybug Simulation**
```python
def run_ladybug_analysis(geometry):
    # Set up Ladybug parameters
    location = EPWLocation("city.epw")
    analysis_period = AnalysisPeriod(1, 1, 0, 12, 31, 23)
    
    # Run cumulative radiation analysis
    analysis = CumulativeRadiation(
        geometry=geometry,
        location=location,
        period=analysis_period,
        grid_size=0.5  # meters
    )
    
    results = analysis.run()
    heat_map = results.to_image()
    return heat_map
```

4. **Save Training Pairs**
```python
def save_pair(height_map, heat_map, index):
    cv2.imwrite(f"data/train/input/{index:04d}.png", height_map)
    cv2.imwrite(f"data/train/label/{index:04d}.png", heat_map)
```

### Running Batch Generation

```bash
python data_generation/batch_generate.py \
    --num_samples 1000 \
    --output_dir data/train \
    --resolution 512
```

### Data Directory Structure

```
data/
├── train/
│   ├── input/           # Height maps
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── label/           # Ladybug outputs
│       ├── 0001.png
│       ├── 0002.png
│       └── ...
├── val/
│   ├── input/
│   └── label/
└── test/
    ├── input/
    └── label/
```

---

## Model Training

### Training Configuration

**Location**: `training/config.yaml`

```yaml
# Model
model:
  type: "pix2pix"
  input_nc: 3
  output_nc: 3
  ngf: 64
  num_downs: 8
  norm: "batch"
  use_dropout: true

# Training
training:
  batch_size: 4
  num_epochs: 200
  lr: 0.0002
  beta1: 0.5
  lambda_L1: 100
  
# Data
data:
  train_dir: "data/train"
  val_dir: "data/val"
  image_size: 512
  
# Augmentation
augmentation:
  random_flip: true
  random_rotation: 15
  color_jitter: 0.1
```

### Training Script

**Location**: `training/train.py`

**Usage**:
```bash
python training/train.py \
    --config training/config.yaml \
    --gpu 0 \
    --output_dir checkpoints/
```

**Key Components**:

```python
# Pseudo-code
def train_epoch(model, dataloader, optimizer, criterion):
    for batch in dataloader:
        # Forward pass
        fake = model(real_input)
        
        # Calculate losses
        loss_GAN = criterion_GAN(fake, real_label)
        loss_L1 = criterion_L1(fake, real_label)
        loss_total = loss_GAN + lambda_L1 * loss_L1
        
        # Backward pass
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
    
    return loss_total.item()
```

### Monitor Training

**TensorBoard**:
```bash
tensorboard --logdir=runs/
```

**Metrics to Track**:
- Generator loss (GAN + L1)
- Discriminator loss
- Validation PSNR/SSIM
- Sample predictions

### Export Trained Model

```bash
python training/export_model.py \
    --checkpoint checkpoints/epoch_200.pth \
    --output backend/models/city_sun_model.pth
```

---

## Running Tests

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_inference.py

# Run with coverage
pytest --cov=app tests/
```

### Test Structure

```
tests/
├── test_api.py          # API endpoint tests
├── test_inference.py    # Model inference tests
├── test_utils.py        # Utility function tests
└── fixtures/            # Test data
    ├── sample_input.png
    └── sample_label.png
```

### Example Test Case

```python
# tests/test_inference.py
import pytest
from app.core.inference import SolarPredictor

def test_model_loading():
    predictor = SolarPredictor(model_path="models/city_sun_model.pth")
    assert predictor.model is not None

def test_prediction():
    predictor = SolarPredictor()
    with open("tests/fixtures/sample_input.png", "rb") as f:
        image_bytes = f.read()
    
    analysis, mask, stats = predictor.predict(image_bytes)
    
    assert len(analysis) > 0
    assert len(mask) > 0
    assert "avg" in stats
```

### API Tests

```bash
# Start test server
pytest tests/test_api.py --run-server

# Or use test client directly
pytest tests/test_api.py
```

---

## Development Workflow

### Branch Strategy

```
main (production)
  ├── develop (staging)
      ├── feature/new-ui
      ├── feature/gpu-support
      └── bugfix/color-mapping
```

### Making Changes

1. **Create feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes and commit**
```bash
git add .
git commit -m "Add: description of changes"
```

3. **Push and create PR**
```bash
git push origin feature/your-feature-name
```

4. **Code review and merge**

### Commit Message Format

```
Type: Brief description

Detailed explanation if needed

Examples:
- Add: new color remapping algorithm
- Fix: texture alignment issue in Three.js
- Update: documentation for API endpoints
- Refactor: inference engine code structure
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install
```

**`.pre-commit-config.yaml`**:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

---

## Code Style Guide

### Python (PEP 8)

**Formatting**:
```bash
# Auto-format code
black app/ tests/

# Check style
flake8 app/ tests/
```

**Type Hints**:
```python
def predict(self, image_bytes: bytes) -> Tuple[bytes, bytes, Dict[str, Any]]:
    """
    Run solar prediction on input image.
    
    Args:
        image_bytes: Input image as bytes
        
    Returns:
        Tuple of (analysis_png, mask_png, statistics_dict)
    """
    pass
```

### JavaScript

**Formatting**:
- Use 2-space indentation
- Semicolons required
- ES6+ features encouraged

```javascript
// Good
const camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 10000);

// Bad
var camera = new THREE.PerspectiveCamera(45,aspect,0.1,10000)
```

### Documentation

**Docstrings**:
```python
def calculate_stats(self, output_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Calculate solar statistics from predicted heat map.
    
    Uses KNN classification in Lab color space to bin pixels
    into 11 hour categories (0-10h).
    
    Args:
        output_bgr: BGR image array from model prediction
        
    Returns:
        Dictionary containing:
            - avg: Average sun hours
            - low_sun: Percentage of pixels <2h
            - high_sun: Percentage of pixels >6h
            - distribution: List of percentages for each hour bin
    
    Raises:
        ValueError: If image is empty or invalid
    """
    pass
```

---

## Debugging

### Enable Debug Mode

**Backend**:
```bash
# Set environment variable
export DEBUG=1

# Or in Python
import os
os.environ["DEBUG"] = "1"
```

**Frontend**:
```javascript
// In index.html
console.log("FRONTEND VERSION: shader-live-1");
```

### Debug Outputs

Debug images are saved to `backend/debug/`:
- `debug_ai_input.png` - Original input
- `debug_ai_output_original.png` - Raw model output
- `debug_ai_output_remapped.png` - After color remapping
- `debug_mask.png` - Valid area mask

### Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.debug("Detailed debug message")
```

---

## Performance Profiling

### Backend Profiling

```python
import cProfile
import pstats

def profile_inference():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run your code
    predictor.predict(image_bytes)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.print_stats(20)
```

### Frontend Profiling

```javascript
// Chrome DevTools
// Performance tab -> Record -> Stop

// Or programmatically
console.time('render');
renderer.render(scene, camera);
console.timeEnd('render');
```

---

## Docker Development

### Build Development Image

```bash
docker build -t instantsim:dev -f Dockerfile.dev .
```

### Run with Volume Mounts

```bash
docker run -it --rm \
    -v $(pwd)/app:/app/app \
    -v $(pwd)/models:/app/models \
    -p 8000:8000 \
    instantsim:dev
```

This enables hot-reloading during development.

---

## Contributing Checklist

Before submitting a PR:

- [ ] Code follows style guide (black + flake8)
- [ ] All tests pass (`pytest`)
- [ ] New features have tests
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No debug code or print statements
- [ ] Performance impact considered

---

## Getting Help

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Open an issue on GitHub
- Join our Discord/Slack (if applicable)
- Email: dev@instantsim.com

---

**Happy coding!**