import logging

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from training import TrainingError, build_job, run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Model Maker API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "model-maker-api", "status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/train")
async def train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    features: str = Form(...),
    target: str = Form(...),
    config: str = Form(...),
    model_id: str = Form(...),
) -> dict[str, str | bool]:
    """Accept a CSV and train a regression network in the background."""
    payload = await file.read()
    try:
        job = build_job(
            file_bytes=payload,
            features_raw=features,
            target=target,
            config_raw=config,
            model_id=model_id,
            nextjs_url=settings.nextjs_url,
            callback_secret=settings.callback_secret,
        )
    except TrainingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(run_training, job)
    return {"success": True, "status": "training"}
