"""
BLIP2-OPT model variant with EfficientNet-guided gating on Q-Former cross-attention.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Add LAVIS to path
# Handle symlink case: try both resolved and original paths
_lavis_path = Path(__file__).resolve().parents[3] / "blip2" / "LAVIS"
if not _lavis_path.exists():
    # Try original path (before symlink resolution)
    _lavis_path = Path(__file__).parent.parent.parent.parent.resolve() / "blip2" / "LAVIS"
if not _lavis_path.exists():
    # Fallback: try workspace root from current working directory
    import os
    workspace_root = Path(os.getcwd()).resolve()
    _lavis_path = workspace_root / "blip2" / "LAVIS"
if _lavis_path.exists():
    sys.path.insert(0, str(_lavis_path))

from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train  # type: ignore
from transformers import AutoTokenizer, OPTForCausalLM
from transformers.models.bert.configuration_bert import BertConfig

from .efficientnet_adapter import EfficientNetAdapter
from .prompt_mapper import PromptMapper
from .qformer_gated import (
    BertLMHeadModelGated,
    GatingConfig,
    LoRAConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class PromptMapperConfig:
    num_tokens: int = 4
    hidden_dim: int = 1024
    dropout: float = 0.1
    use_layer_norm: bool = True


class Blip2OPTGated(Blip2Base):
    """
    BLIP2-OPT model that freezes visual encoder and LLM while learning gating modules
    for Q-Former cross-attention using EfficientNet-B3 features.
    """

    def __init__(
        self,
        vit_model: str = "eva_clip_g",
        img_size: int = 224,
        drop_path_rate: float = 0.0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,
        num_query_token: int = 32,
        opt_model: str = "facebook/opt-2.7b",
        prompt: str = "",
        max_txt_len: int = 32,
        apply_lemmatizer: bool = False,
        efficientnet_checkpoint: Optional[str] = None,
        efficientnet_output_dim: int = 768,
        convert_from_blip_norm: bool = True,
        gating_config: Optional[Dict] = None,
        prompt_mapper_cfg: Optional[Dict] = None,
        lora_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        # Initialize vision encoder (frozen)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logger.info("Visual encoder frozen.")

        # Initialize Qformer with gated cross-attention
        qformer_config = BertConfig.from_pretrained("bert-base-uncased")
        qformer_config.encoder_width = self.visual_encoder.num_features
        qformer_config.add_cross_attention = True
        qformer_config.cross_attention_freq = 2
        qformer_config.query_length = num_query_token

        gating_cfg = self._build_gating_config(
            gating_config or {},
            controller_dim=qformer_config.hidden_size,
            num_heads=qformer_config.num_attention_heads,
        )
        lora_cfg = self._build_lora_config(lora_config or {})

        self.Qformer, self.query_tokens = self._init_qformer(
            qformer_config,
            gating_cfg,
            lora_cfg,
        )

        # Initialize EfficientNet adapter
        repo_root = Path(__file__).resolve().parents[3]
        eff_ckpt = Path(efficientnet_checkpoint or (repo_root / "runs" / "efficientnet-28" / "best.pt"))
        self.efficientnet_adapter = EfficientNetAdapter(
            checkpoint_path=str(eff_ckpt),
            output_dim=efficientnet_output_dim,
            convert_from_blip_norm=convert_from_blip_norm,
        )

        # Initialize PromptMapper
        mapper_cfg = self._build_prompt_mapper_config(prompt_mapper_cfg or {})
        self.prompt_mapper = PromptMapper(
            image_dim=efficientnet_output_dim,
            output_dim=self.Qformer.config.hidden_size,
            hidden_dim=mapper_cfg.hidden_dim,
            num_tokens=mapper_cfg.num_tokens,
            text_dim=None,  # Will be set dynamically after OPT init
            dropout=mapper_cfg.dropout,
            use_layer_norm=mapper_cfg.use_layer_norm,
        )

        # Initialize OPT model (frozen)
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16 if vit_precision == "fp16" else None
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer("\n", add_special_tokens=False).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size,
            self.opt_model.config.hidden_size,
        )

        # Update prompt mapper text projection now that OPT is initialized
        self.prompt_mapper.text_dim = self.opt_model.config.hidden_size
        self.prompt_mapper.text_proj = nn.Linear(
            self.opt_model.config.hidden_size,
            self.Qformer.config.hidden_size,
            bias=True,
        )
        nn.init.trunc_normal_(self.prompt_mapper.text_proj.weight, std=0.02)
        nn.init.zeros_(self.prompt_mapper.text_proj.bias)

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    @staticmethod
    def _build_gating_config(cfg: Dict, controller_dim: int, num_heads: int) -> GatingConfig:
        gating_type = cfg.get("type", "film")
        per_head = cfg.get("per_head", True)
        hidden_dim = cfg.get("hidden_dim", controller_dim)
        init_scale = cfg.get("init_scale", 1e-2)
        use_layer_norm = cfg.get("use_layer_norm", True)
        use_bias = cfg.get("use_bias", True)

        if per_head and controller_dim % num_heads != 0 and gating_type == "film":
            logger.warning(
                "Controller dim %s not divisible by num_heads %s; consider setting per_head=False.",
                controller_dim,
                num_heads,
            )

        return GatingConfig(
            controller_dim=controller_dim,
            hidden_dim=hidden_dim,
            per_head=per_head,
            init_scale=init_scale,
            gating_type=gating_type,
            use_layer_norm=use_layer_norm,
            use_bias=use_bias,
        )

    @staticmethod
    def _build_lora_config(cfg: Dict) -> LoRAConfig:
        enabled = cfg.get("enabled", False)
        rank = cfg.get("rank", 8)
        alpha = cfg.get("alpha", 16)
        dropout = cfg.get("dropout", 0.0)
        return LoRAConfig(enabled=enabled, rank=rank, alpha=alpha, dropout=dropout)

    @staticmethod
    def _build_prompt_mapper_config(cfg: Dict) -> PromptMapperConfig:
        return PromptMapperConfig(
            num_tokens=cfg.get("num_tokens", 4),
            hidden_dim=cfg.get("hidden_dim", 1024),
            dropout=cfg.get("dropout", 0.1),
            use_layer_norm=cfg.get("use_layer_norm", True),
        )

    def _init_qformer(
        self,
        config: BertConfig,
        gating_config: GatingConfig,
        lora_config: LoRAConfig,
    ) -> Tuple[BertLMHeadModelGated, nn.Parameter]:
        qformer = BertLMHeadModelGated.from_pretrained(
            "bert-base-uncased",
            config=config,
            gating_config=gating_config,
            lora_config=lora_config if lora_config.enabled else None,
        )
        query_tokens = nn.Parameter(torch.zeros(1, config.query_length, config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=config.initializer_range)

        qformer.cls = None
        qformer.bert.embeddings.word_embeddings = None
        qformer.bert.embeddings.position_embeddings = None

        return qformer, query_tokens

    def _prepare_controller(
        self,
        efficientnet_embeddings: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        controller_tokens, _ = self.prompt_mapper(
            efficientnet_embeddings,
            text_embeddings=text_embeddings,
        )
        return controller_tokens

    def forward(self, samples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = samples["image"]
        device = images.device

        # EfficientNet features (remain in fp32)
        with torch.no_grad():
            efficientnet_embeddings, _ = self.efficientnet_adapter(images)
        efficientnet_embeddings = efficientnet_embeddings.to(device=device, dtype=self.query_tokens.dtype)

        text_inputs: List[str] = samples.get("text_input", [])
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        self.opt_tokenizer.padding_side = "right"
        opt_tokens = self.opt_tokenizer(
            [t + "\n" for t in text_inputs],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        text_embeddings = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids).detach()
        text_embeddings = text_embeddings.to(dtype=self.query_tokens.dtype)

        controller_tokens = self._prepare_controller(efficientnet_embeddings, text_embeddings)
        controller_tokens = controller_tokens.to(device=device, dtype=self.query_tokens.dtype)
        self.Qformer.set_gating_controller(controller_tokens)

        # Vision encoder forward with autocast
        vision_param = next(self.visual_encoder.parameters(), None)
        vision_dtype = vision_param.dtype if vision_param is not None else images.dtype
        vision_images = images.to(dtype=vision_dtype)

        autocast_dtype = vision_dtype if vision_dtype in (torch.float16, torch.bfloat16) else None
        with self.maybe_autocast(dtype=autocast_dtype):
            image_embeds = self.ln_vision(self.visual_encoder(vision_images))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        self.Qformer.set_gating_controller(None)

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long, device=device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id,
            -100,
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100

        empty_targets = torch.ones(atts_opt.size(), dtype=torch.long, device=device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = torch.cat([inputs_opt, text_embeddings], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "controller_tokens": controller_tokens.detach(),
            "efficientnet_embeddings": efficientnet_embeddings.detach(),
            "opt_tokens": opt_tokens,
            "attention_mask": attention_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        samples: Dict[str, torch.Tensor],
        use_nucleus_sampling: bool = False,
        num_beams: int = 5,
        max_length: int = 30,
        min_length: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        num_captions: int = 1,
        temperature: float = 1.0,
    ) -> List[str]:
        images = samples["image"]
        device = images.device

        with torch.no_grad():
            efficientnet_embeddings, _ = self.efficientnet_adapter(images)
        efficientnet_embeddings = efficientnet_embeddings.to(device=device, dtype=self.query_tokens.dtype)

        prompt_texts: List[str]
        if "text_input" in samples:
            text_input = samples["text_input"]
            if isinstance(text_input, str):
                prompt_texts = [text_input]
            else:
                prompt_texts = list(text_input)
        else:
            prompt_texts = [self.prompt] * images.size(0)

        self.opt_tokenizer.padding_side = "left"
        opt_tokens = self.opt_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        text_embeddings = self.opt_model.get_input_embeddings()(opt_tokens.input_ids).to(dtype=self.query_tokens.dtype)
        controller_tokens = self._prepare_controller(efficientnet_embeddings, text_embeddings)
        controller_tokens = controller_tokens.to(device=device, dtype=self.query_tokens.dtype)
        self.Qformer.set_gating_controller(controller_tokens)

        vision_param = next(self.visual_encoder.parameters(), None)
        vision_dtype = vision_param.dtype if vision_param is not None else images.dtype
        vision_images = images.to(dtype=vision_dtype)
        autocast_dtype = vision_dtype if vision_dtype in (torch.float16, torch.bfloat16) else None

        with self.maybe_autocast(dtype=autocast_dtype):
            image_embeds = self.ln_vision(self.visual_encoder(vision_images))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        self.Qformer.set_gating_controller(None)

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long, device=device)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        inputs_embeds = torch.cat([inputs_opt, text_embeddings], dim=1)
        
        # Ensure consistent dtype for generation
        inputs_embeds = inputs_embeds.to(dtype=self.opt_model.dtype)

        outputs = self.opt_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )

        decoded = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [text.strip() for text in decoded]

    @torch.no_grad()
    def predict_answers(
        self,
        samples: Dict[str, torch.Tensor],
        num_beams: int = 5,
        inference_method: str = "generate",
        max_len: int = 10,
        min_len: int = 1,
        length_penalty: float = 0.0,
        **kwargs,
    ) -> List[str]:
        if inference_method != "generate":
            raise ValueError(f"Unsupported inference_method {inference_method}")

        images = samples["image"]
        device = images.device

        with torch.no_grad():
            efficientnet_embeddings, _ = self.efficientnet_adapter(images)
        efficientnet_embeddings = efficientnet_embeddings.to(device=device, dtype=self.query_tokens.dtype)

        text_input = samples["text_input"]
        if isinstance(text_input, str):
            text_input = [text_input]

        self.opt_tokenizer.padding_side = "left"
        opt_tokens = self.opt_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        text_embeddings = self.opt_model.get_input_embeddings()(opt_tokens.input_ids).to(dtype=self.query_tokens.dtype)
        controller_tokens = self._prepare_controller(efficientnet_embeddings, text_embeddings)
        controller_tokens = controller_tokens.to(device=device, dtype=self.query_tokens.dtype)
        self.Qformer.set_gating_controller(controller_tokens)

        vision_param = next(self.visual_encoder.parameters(), None)
        vision_dtype = vision_param.dtype if vision_param is not None else images.dtype
        vision_images = images.to(dtype=vision_dtype)
        autocast_dtype = vision_dtype if vision_dtype in (torch.float16, torch.bfloat16) else None
        with self.maybe_autocast(dtype=autocast_dtype):
            image_embeds = self.ln_vision(self.visual_encoder(vision_images))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        self.Qformer.set_gating_controller(None)

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long, device=device)

        inputs_embeds = torch.cat([inputs_opt, text_embeddings], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        # Ensure consistent dtype for generation
        inputs_embeds = inputs_embeds.to(dtype=self.opt_model.dtype)

        outputs = self.opt_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=num_beams,
            max_new_tokens=max_len,
            min_length=min_len,
            eos_token_id=self.eos_token_id,
            length_penalty=length_penalty,
        )
        decoded = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded = [text.strip() for text in decoded]

        if self._apply_lemmatizer or samples.get("apply_lemmatizer"):
            decoded = self._lemmatize(decoded)

        return decoded


