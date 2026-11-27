

https://github.com/user-attachments/assets/654862ef-56f4-4a7a-b21f-77b276287453

# Model Maker API

Training service for the Model Maker web app. It fits a small feed-forward network to a CSV and reports progress back to that app.

The web app stores the job. This service does the training.

## What a training run does

1. Read the uploaded CSV and check that the chosen columns are numeric.
2. Split the rows 60% train, 20% validation, 20% test.
3. Standardise inputs using the training split. Constant columns are left unscaled rather than divided by zero.
4. Train a regression network with ReLU hidden layers, a linear output, Adam, and mean squared error.
5. After each epoch, post progress to the web app. When training finishes, post the test loss, R², loss history, and a few sample predictions.

## Requirements

- Python 3.11
- The Model Maker web app, because progress is written through its callback

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `CALLBACK_SECRET` to the same value used by the web app.

```bash
python main.py
```

The service listens on `http://localhost:8001`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness check used by the web app |
| `POST` | `/train` | Start a training job and return immediately |

`POST /train` is multipart form data:

| Field | Value |
| --- | --- |
| `file` | CSV dataset |
| `features` | JSON array of column names |
| `target` | Target column name |
| `config` | JSON object: `learningRate`, `epochs`, `hiddenLayers`, `neuronsPerLayer` |
| `model_id` | Id already created by the web app |

Progress is sent to `{NEXTJS_URL}/api/update-model` with the header `X-Callback-Secret`.

## Configuration

| Variable | Purpose |
| --- | --- |
| `NEXTJS_URL` | Web app origin, without a trailing slash |
| `CALLBACK_SECRET` | Shared secret for the progress callback |
| `ALLOWED_ORIGINS` | Comma-separated browser origins allowed by CORS |
| `PORT` | Port to listen on. Defaults to `8001` |
