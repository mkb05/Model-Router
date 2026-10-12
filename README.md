# Edge Model Router

[![Render](https://img.shields.io/badge/Render-Deployed-brightgreen?style=flat-square&logo=render)](https://model-router-9k9s.onrender.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-blue?style=flat-square)](https://onnxruntime.ai/)

A sub-25ms prompt difficulty and complexity classifier microservice engineered to reduce downstream LLM API costs by 40% to 70%.

The service evaluates incoming prompts and returns an agnostic difficulty tier (`small` vs. `heavy`) alongside prediction confidence scores. Downstream consumers map these tiers to their preferred LLM providers (e.g., dispatching `small` to fast/inexpensive models like Claude Haiku or Llama 8B, and `heavy` to reasoning models like Claude Opus or GPT-4o).

The service runs a dynamic **INT8-quantized encoder** via **ONNX Runtime (C++)**, maintaining an idle RAM footprint of **<80 MB** on Render's free tier without triggering Out-Of-Memory (OOM Status 137) restarts.

---

## Live Endpoint

* **Base URL:** `https://model-router-9k9s.onrender.com`
* **Interactive Docs (Swagger UI):** `https://model-router-9k9s.onrender.com/docs`
* **Health Check:** `https://model-router-9k9s.onrender.com/health`

> **Note on Free Hosting:** If the instance has been idle, Render spins down the container. The first cold-start request may take 30–50 seconds to resume. Subsequent calls execute in <25 ms.

---

## API Reference

### 1. Health Check
`GET /health`

```bash
curl -X GET "[https://model-router-9k9s.onrender.com/health](https://model-router-9k9s.onrender.com/health)"
```
Response:
```
{
  "status": "healthy",
  "engine": "onnxruntime-cpu"
}
```
### 2. Classify Prompt Complexity
   ```
POST /api/v1/route
```
```
{
  "prompt": "Write a python function to check if a word is a palindrome."
}
```

Example Response (small Tier)

```
{
  "decision": {
    "tier": "small",
    "confidence": 0.8413
  },
  "scores": {
    "small_probability": 0.8413,
    "heavy_probability": 0.1587
  }
}
```

Example Response (heavy Tier)

```
{
  "decision": {
    "tier": "heavy",
    "confidence": 0.8009
  },
  "scores": {
    "small_probability": 0.1991,
    "heavy_probability": 0.8009
  }
}
```

<h2>Client Integration Examples</h2>
Clients inspect the returned tier and dispatch to their chosen models.

<h5>Python (httpx / requests)</h5>

```
import httpx

ROUTER_URL = "[https://model-router-9k9s.onrender.com/api/v1/route](https://model-router-9k9s.onrender.com/api/v1/route)"

# Map tiers to your provider's model IDs
MODEL_TIER_MAPPING = {
    "small": "groq/compound-mini",       # Or "claude-3-haiku-20240307", "gpt-4o-mini"
    "heavy": "openai/gpt-oss-120b"       # Or "claude-3-opus-20240229", "gpt-4o"
}

async def select_model(prompt: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(ROUTER_URL, json={"prompt": prompt})
            if resp.status_code == 200:
                tier = resp.json()["decision"]["tier"]
                return MODEL_TIER_MAPPING.get(tier, MODEL_TIER_MAPPING["small"])
    except Exception:
        pass
    return MODEL_TIER_MAPPING["small"]
```

TypeScript / Node.js (fetch)

```
const ROUTER_URL = "[https://model-router-9k9s.onrender.com/api/v1/route](https://model-router-9k9s.onrender.com/api/v1/route)";

async function routePrompt(prompt: string): Promise<"small" | "heavy"> {
  try {
    const res = await fetch(ROUTER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    if (res.ok) {
      const data = await res.json();
      return data.decision.tier;
    }
  } catch (err) {
    console.warn("Router request failed, defaulting to small tier:", err);
  }
  return "small";
}
```

cURL
```
curl -X POST "[https://model-router-9k9s.onrender.com/api/v1/route](https://model-router-9k9s.onrender.com/api/v1/route)" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Design a distributed consensus mechanism for multi-region writes."}'
```

<h2>Technical Specifications</h2>

```
Parameter:              Specification

Base Architecture:      DistilBERT Sequence Classifier
Inference Framework:    ONNX Runtime (CPUExecutionProvider)
Quantization:           Dynamic INT8 (weights stored as 8-bit integers)
Model Size:             ~65 MB (down from ~268 MB FP32)
Token Truncation:       128 tokens (enforces sub-25ms classification)
Runtime Dependencies:   Standalone C++ engine; zero PyTorch dependencies
Idle Memory Footprint:  ~75 MB RAM (well under Render's 512 MB ceiling)
```

<H2>Local Development</H2>


### Clone the repository
```
git clone [https://github.com/](https://github.com/)<your-username>/model-router.git
cd model-router/router
```

### Create virtual environment
```
python -m venv venv
```

### Windows:
```
.\venv\Scripts\Activate.ps1
```

### Linux/macOS:
```
source venv/bin/activate
```

### Install inference dependencies
```
pip install -r requirements.txt
```

### Start the server locally
```
uvicorn main:app --port 8001 --reload
```
