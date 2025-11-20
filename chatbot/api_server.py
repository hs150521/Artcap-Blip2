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
from typing import Optional, Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import yaml

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "blip2" / "LAVIS"))

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

# KV model variables
_kv_model: Optional[Any] = None
_kv_vis_processor: Optional[Any] = None


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


def load_kv_model():
    """Load KV-modulated BLIP-2 model."""
    global _kv_model, _kv_vis_processor, _device
    
    if _kv_model is not None:
        return
    
    logger.info("Loading KV-modulated BLIP-2 model...")
    
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    config_path = REPO_ROOT / "models" / "KV" / "config" / "artquest.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"KV model config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import KV model loader
    from models.KV.utils.model_loader import load_kv_model as load_kv_model_func
    
    # Load model
    _kv_model = load_kv_model_func(config, device=_device)
    _kv_model.eval()
    
    # Load checkpoint
    checkpoint_path = REPO_ROOT / "models" / "KV" / "runs" / "best.pt"
    if checkpoint_path.exists():
        logger.info(f"Loading KV model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=_device)
        model_state = checkpoint.get("model_state_dict", checkpoint)
        _kv_model.load_state_dict(model_state, strict=False)
        logger.info("KV model checkpoint loaded successfully.")
    else:
        logger.warning(f"KV model checkpoint not found at {checkpoint_path}, using untrained model.")
    
    # Load vision processor (same as BLIP-2)
    from lavis.processors import load_processor
    _kv_vis_processor = load_processor("blip_image_eval").build(image_size=224)
    
    logger.info("KV-modulated BLIP-2 model loaded successfully.")


def generate_caption_kv(image: Image.Image, prompt: str) -> str:
    """Generate caption using KV-modulated model."""
    if _kv_model is None or _kv_vis_processor is None or _device is None:
        load_kv_model()
    
    # Process image
    image_tensor = _kv_vis_processor(image).unsqueeze(0).to(_device)
    
    # Prepare samples dict
    samples = {
        "image": image_tensor,
        "prompt": prompt
    }
    
    # Generate
    with torch.no_grad():
        generated_texts = _kv_model.generate(
            samples,
            num_beams=5,
            max_new_tokens=200,
            min_new_tokens=1,
        )
    
    # Extract answer (remove prompt prefix if present)
    generated_text = generated_texts[0] if generated_texts else ""
    
    # Remove prompt prefix if it's in the generated text
    if prompt in generated_text:
        generated_text = generated_text.replace(prompt, "").strip()
    
    return generated_text.strip()


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
    try:
        load_kv_model()
    except Exception as e:
        logger.warning(f"Failed to load KV model: {e}. KV model endpoints will not be available.")
    yield
    # Shutdown (if needed)

# FastAPI app
app = FastAPI(
    title="BLIP-2 Image Captioning API", 
    lifespan=lifespan
)

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


@app.post("/api/generate-kv", response_model=GenerateResponse)
async def generate_kv(request: GenerateRequest):
    """Generate caption using KV-modulated model."""
    if _kv_model is None:
        raise HTTPException(status_code=503, detail="KV model not loaded")
    
    try:
        logger.info(f"KV: Received request with prompt: {request.prompt[:50]}...")
        
        # Decode base64 image
        image_data = base64.b64decode(request.image.split(",")[-1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        logger.info(f"KV: Image decoded successfully, size: {image.size}")

        # Generate caption
        logger.info("KV: Generating caption...")
        caption = generate_caption_kv(image, request.prompt)
        logger.info(f"KV: Generated caption: {caption[:100]}...")

        return GenerateResponse(caption=caption)

    except Exception as e:
        logger.error(f"KV: Error generating caption: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "blip2_loaded": _model is not None,
        "kv_loaded": _kv_model is not None
    }


if __name__ == "__main__":
    import uvicorn
    # Configure uvicorn to handle large requests (50MB)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        limit_concurrency=10,
        limit_max_requests=1000,
        timeout_keep_alive=30,
    )

