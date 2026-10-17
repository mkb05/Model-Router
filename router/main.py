import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from tokenizers import Tokenizer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Edge Model Router API", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "onnx_router", "model_quantized.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "onnx_router", "tokenizer.json")

# Allow requests from localhost during development and your deployed frontend
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "*",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    raise RuntimeError(f"Missing artifacts in onnx_router. Checked: {MODEL_PATH}")

# Optimize for low-RAM single CPU environments
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)

# Fast Rust tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
tokenizer.enable_truncation(max_length=128)
tokenizer.enable_padding(length=128)

class RouteRequest(BaseModel):
    prompt: str

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

@app.get("/health")
def health_check():
    return {"status": "healthy", "engine": "onnxruntime-cpu"}

@app.post("/api/v1/route")
def route_prompt(payload: RouteRequest):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        # Tokenize directly to NumPy arrays
        encoded = tokenizer.encode(payload.prompt)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # Sub-25ms CPU inference
        logits = session.run(None, ort_inputs)[0]
        probs = softmax(logits)[0]
        label_id = int(np.argmax(probs))

        tier = "heavy" if label_id == 1 else "small"
        target_model = "openai/gpt-oss-120b" if tier == "heavy" else "groq/compound-mini"

        return {
            "decision": {
                "tier": tier,
                "confidence": round(float(probs[label_id]), 4)
            },
            "scores": {
                "small_probability": round(float(probs[0]), 4),
                "heavy_probability": round(float(probs[1]), 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))