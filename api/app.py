from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import shutil

from orchestrator.orchestrator import Orchestrator

app = FastAPI(title="Auto Data Scientist API", description="API for Auto ML Platform")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and pipeline
model = None
pipeline = None
target_encoder = None

class PredictRequest(BaseModel):
    data: list[dict]

@app.on_event("startup")
def load_artifacts():
    global model, pipeline, target_encoder
    
    model_path = "models/latest/model.pkl"
    pipeline_path = "models/latest/pipeline.pkl"
    encoder_path = "models/latest/target_encoder.pkl"
    
    if os.path.exists(model_path) and os.path.exists(pipeline_path):
        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        
        if os.path.exists(encoder_path):
            target_encoder = joblib.load(encoder_path)
        else:
            target_encoder = None
            
        print("✅ Successfully loaded model and pipeline artifacts.")
    else:
        print("⚠️ Warning: Model or pipeline artifacts not found. Please run the training pipeline first.")

@app.post("/upload-and-train")
async def upload_and_train(
    file: UploadFile = File(...), 
    target_column: str = Form(...),
    task_type: str = Form(...)
):
    print(f"📥 Received file: {file.filename} with target: {target_column} ({task_type})")
    
    # Save the uploaded file temporarily
    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Initialize Orchestrator and run pipeline
        orchestrator = Orchestrator(task_type=task_type)
        result = orchestrator.run_training_pipeline(
            raw_data_path=file_path,
            target_column=target_column
        )
        
        # Reload artifacts to memory to generate predictions
        load_artifacts()
        
        # Generate predictions for the uploaded dataset
        df = pd.read_csv(file_path)
        
        # If the target column is in the dataframe, drop it before prediction, or keep it depending on pipeline logic.
        # The pipeline was trained on data without the target column.
        if target_column in df.columns:
            X_pred = df.drop(columns=[target_column])
        else:
            X_pred = df

        transformed_data = pipeline.transform(X_pred)
        predictions = model.predict(transformed_data)
        
        # Inverse transform predictions if it's classification and we have an encoder
        if target_encoder is not None and task_type == "classification":
            predictions = target_encoder.inverse_transform(predictions)
        
        # Append predictions
        df[f"Predicted_{target_column}"] = predictions
        
        # Save predictions CSV
        os.makedirs("data/processed", exist_ok=True)
        predictions_path = "data/processed/predictions.csv"
        df.to_csv(predictions_path, index=False)
        
        # Prepare response
        return {
            "status": "success",
            "message": "Pipeline completed successfully",
            "data": result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-predictions")
def download_predictions():
    file_path = "data/processed/predictions.csv"
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename="predictions.csv", media_type='text/csv')
    else:
        raise HTTPException(status_code=404, detail="Predictions file not found.")

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
        
    try:
        input_data = pd.DataFrame(request.data)
        transformed_data = pipeline.transform(input_data)
        predictions = model.predict(transformed_data)
        
        if target_encoder is not None:
             predictions = target_encoder.inverse_transform(predictions)
             
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
        
    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/batch_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        df = pd.read_csv(file_path)
        transformed_data = pipeline.transform(df)
        predictions = model.predict(transformed_data)
        
        if target_encoder is not None:
             predictions = target_encoder.inverse_transform(predictions)
             
        df["AI_Prediction"] = predictions
        
        os.makedirs("data/processed", exist_ok=True)
        out_path = f"data/processed/batch_predictions.csv"
        df.to_csv(out_path, index=False)
        
        return FileResponse(path=out_path, filename="batch_predictions.csv", media_type='text/csv')
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

# Mount directories
os.makedirs("frontend", exist_ok=True)
os.makedirs("reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
