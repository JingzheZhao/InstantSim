from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
# CRITICAL FIX: Use relative import (.) to avoid conflict with system 'app' library
from .inference import SunPredictor
import io
import os

# Initialize the FastAPI app
app = FastAPI(title="InstantSun Backend", version="1.0")

# Global variable to store the predictor instance
predictor = None
# Path relative to the root 'InstantSim_Backend' folder
MODEL_PATH = "app/models/city_sun_model.pth"


@app.on_event("startup")
async def startup_event():
    """
    This function runs once when the server starts.
    It initializes the AI model and loads it into memory/GPU.
    """
    global predictor
    # Check if model exists before loading
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Starting up... Loading model from {MODEL_PATH}")
            predictor = SunPredictor(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print(f"Current working directory: {os.getcwd()}")


@app.get("/")
def health_check():
    """
    Simple health check endpoint.
    Use this to verify if the server is running.
    """
    status = "active" if predictor else "model_not_loaded"
    return {"status": status, "service": "InstantSun API"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    The main prediction endpoint.
    1. Receives an uploaded image file (bytes).
    2. Passes it to the predictor for inference and artifact removal.
    3. Returns the processed image as a PNG stream.
    """
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not initialized. Check server logs.")

    try:
        # 1. Read the uploaded file bytes
        contents = await file.read()
        image_stream = io.BytesIO(contents)

        # 2. Run inference (Predict -> Remove Artifacts -> Resize)
        result_image = predictor.predict(image_stream)

        # 3. Save the result into a byte stream (in-memory)
        output_stream = io.BytesIO()
        result_image.save(output_stream, format="PNG")
        output_stream.seek(0)  # Reset stream position to the beginning

        # 4. Return the image directly as a response
        return StreamingResponse(output_stream, media_type="image/png")

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    # For debugging purposes only
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)