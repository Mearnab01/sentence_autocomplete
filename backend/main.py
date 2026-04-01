from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from services.inference import predict_llm, predict_dl
from services.training import train_dl_model
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Autocomplete System API")

# Setup CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str
    mode: str = "llm_mode" # "llm_mode" or "dl_mode"
    top_k: int = 3

class TrainRequest(BaseModel):
    texts: List[str]
    epochs: int = 5
    batch_size: int = 32

@app.on_event("startup")
def startup_event():
    logger.info("Starting up FastAPI application")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: PredictRequest):
    if request.mode == "llm_mode":
        return await predict_llm(request.text)
    elif request.mode == "dl_mode":
        return predict_dl(request.text, top_k=request.top_k)
    else:
        return {"error": f"Invalid mode: {request.mode}"}

@app.post("/train")
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    if not request.texts:
        return {"error": "No training texts provided"}
        
    # Run training in background to not block the API
    background_tasks.add_task(
        train_dl_model,
        texts=request.texts,
        epochs=request.epochs,
        batch_size=request.batch_size
    )
    return {"status": "Training started in background"}
