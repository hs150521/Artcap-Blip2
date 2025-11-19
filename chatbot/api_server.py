#!/usr/bin/env python3
"""
FastAPI server for BLIP-2 image captioning.
Provides API endpoint to generate captions from images and prompts.
"""

import base64
import io
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
except ImportError as exc:
    raise SystemExit(
        "transformers is required. Install it with `pip install transformers`."
    ) from exc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = REPO_ROOT / "models" / "blip2-opt-2.7b" / "snapshots" / "59a1ef6c1e5117b3f65523d1c6066825bcf315e3"

# Global model variables
_processor: Optional[Blip2Processor] = None
_model: Optional[Blip2ForConditionalGeneration] = None
_device: Optional[torch.device] = None


def load_model():
    """Load BLIP-2 model and processor."""
    global _processor, _model, _device

    if _processor is not None and _model is not None:
        return

    logger.info(f"Loading BLIP-2 model from {MODEL_PATH}...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"BLIP-2 model path not found: {MODEL_PATH}")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {_device}")

    dtype = torch.float16 if _device.type == "cuda" else torch.float32

    try:
        # Try fast tokenizer first
        _processor = Blip2Processor.from_pretrained(
            str(MODEL_PATH), local_files_only=True, use_fast=True
        )
    except Exception as exc:
        # Fall back to slow tokenizer if fast tokenizer fails
        logger.warning(
            f"Fast tokenizer failed ({exc}), falling back to slow tokenizer."
        )
        try:
            _processor = Blip2Processor.from_pretrained(
                str(MODEL_PATH), local_files_only=True, use_fast=False
            )
        except Exception as exc2:
            logger.error(f"Failed to load processor: {exc2}")
            raise

    _model = Blip2ForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=dtype,
    )
    _model.to(_device)
    _model.eval()

    logger.info("BLIP-2 model loaded successfully.")


def generate_caption(image: Image.Image, prompt: str) -> str:
    """Generate caption from image and prompt."""
    if _processor is None or _model is None or _device is None:
        load_model()

    # Process inputs - ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Process inputs
    inputs = _processor(images=image, text=prompt, return_tensors="pt")
    
    # Move inputs to device
    processed_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k == "pixel_values" and _model.dtype == torch.float16:
                processed_inputs[k] = v.to(device=_device, dtype=torch.float16)
            elif v.dtype.is_floating_point:
                processed_inputs[k] = v.to(device=_device, dtype=_model.dtype if hasattr(_model, 'dtype') else torch.float32)
            else:
                processed_inputs[k] = v.to(_device)
        else:
            processed_inputs[k] = v
    
    inputs = processed_inputs

    # Generate
    with torch.no_grad():
        generated_ids = _model.generate(**inputs, max_new_tokens=200)

    # Decode
    generated_text = _processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    return generated_text


from contextlib import asynccontextmanager

class GenerateRequest(BaseModel):
    image: str  # base64 encoded image
    prompt: str


class GenerateResponse(BaseModel):
    caption: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Startup
    load_model()
    yield
    # Shutdown (if needed)

# FastAPI app
app = FastAPI(title="BLIP-2 Image Captioning API", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate caption from image and prompt."""
    try:
        logger.info(f"Received request with prompt: {request.prompt[:50]}...")
        
        # Decode base64 image
        image_data = base64.b64decode(request.image.split(",")[-1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        logger.info(f"Image decoded successfully, size: {image.size}")

        # Generate caption
        logger.info("Generating caption...")
        caption = generate_caption(image, request.prompt)
        logger.info(f"Generated caption: {caption[:100]}...")

        return GenerateResponse(caption=caption)

    except Exception as e:
        logger.error(f"Error generating caption: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

