"""
BLIP-2 OPT model augmented with EfficientNet-guided KV modulation. This module
freezes the heavy vision and language backbones while training lightweight
controllers that inject style priors into the Q-Former cross-attention layers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM

# Add LAVIS to path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "blip2" / "LAVIS"))

from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

from .qformer_kv import BertLMHeadModelKVModulated
from .efficientnet_adapter import EfficientNetAdapter
from .prompt_mapper import PromptMapper


class Blip2OPTKV(Blip2Base):
    """BLIP-2 OPT variant with EfficientNet-guided KV modulation."""

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
        max_txt_len: int = 64,
        apply_lemmatizer: bool = False,
        efficientnet_checkpoint: Optional[str] = None,
        efficientnet_output_dim: int = 768,
        enable_efficientnet_grad: bool = False,
        prompt_mapper_cfg: Optional[Dict] = None,
        kv_modulation_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        prompt_mapper_cfg = prompt_mapper_cfg or {}
        kv_modulation_cfg = kv_modulation_cfg or {}

        self.tokenizer = self.init_tokenizer()

        # Vision encoder -----------------------------------------------------
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model,
            img_size,
            drop_path_rate,
            use_grad_checkpoint,
            vit_precision,
        )
        if freeze_vit:
            for _, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("Frozen visual encoder (%s)", vit_model)

        # Q-Former with KV modulation ----------------------------------------
        self.Qformer, self.query_tokens = self.init_Qformer_kv(
            num_query_token=num_query_token,
            vision_width=self.visual_encoder.num_features,
            kv_config=kv_modulation_cfg,
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        qformer_hidden = self.Qformer.config.hidden_size

        # OPT language model -------------------------------------------------
        opt_model_path = self._resolve_opt_path(opt_model)
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model_path, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model_path,
            torch_dtype=torch.float16,
        )
        for param in self.opt_model.parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer("\n", add_special_tokens=False).input_ids[0]
        self.opt_proj = nn.Linear(qformer_hidden, self.opt_model.config.hidden_size)

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        self.prompt_length = self.opt_tokenizer(prompt, return_tensors="pt").attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        # EfficientNet controller --------------------------------------------
        self.efficientnet_adapter = EfficientNetAdapter(
            checkpoint_path=efficientnet_checkpoint,
            output_dim=efficientnet_output_dim,
            enable_feature_grad=enable_efficientnet_grad,
        )
        self.prompt_mapper = PromptMapper(
            image_dim=efficientnet_output_dim,
            output_dim=qformer_hidden,
            hidden_dim=prompt_mapper_cfg.get("hidden_dim"),
            num_tokens=prompt_mapper_cfg.get("num_tokens", 4),
            dropout=prompt_mapper_cfg.get("dropout", 0.1),
            use_layer_norm=prompt_mapper_cfg.get("use_layer_norm", True),
        )

    # --------------------------------------------------------------------- #
    # Initialization helpers
    # --------------------------------------------------------------------- #
    def init_Qformer_kv(
        self,
        num_query_token: int,
        vision_width: int,
        cross_attention_freq: int = 2,
        kv_config: Optional[Dict] = None,
    ):
        """Initialize the KV-modulated Q-Former."""
        from transformers.models.bert.configuration_bert import BertConfig

        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        Qformer = BertLMHeadModelKVModulated.from_pretrained(
            "bert-base-uncased",
            config=encoder_config,
            kv_config=kv_config,
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def _resolve_opt_path(self, opt_model: str) -> str:
        opt_model_path = opt_model
        if not Path(opt_model_path).exists():
            repo_root = Path(__file__).resolve().parents[3]
            potential_path = repo_root / opt_model_path
            if potential_path.exists():
                opt_model_path = str(potential_path)
        if Path(opt_model_path).exists():
            snapshots_dir = Path(opt_model_path) / "snapshots"
            if snapshots_dir.exists():
                snapshot_dirs = sorted(snapshots_dir.iterdir())
                if snapshot_dirs:
                    opt_model_path = str(snapshot_dirs[0])
                    logging.info("Using OPT snapshot at %s", opt_model_path)
        return opt_model_path

    # --------------------------------------------------------------------- #
    # Core logic
    # --------------------------------------------------------------------- #
    def _prepare_controller(self, images: torch.Tensor) -> torch.Tensor:
        """Extract EfficientNet embeddings and map them to controller tokens."""
        embeddings, _ = self.efficientnet_adapter(images)
        controller_tokens, _ = self.prompt_mapper(embeddings)
        return controller_tokens

    def _encode_vision(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images with the (frozen) vision tower."""
        vision_param = next(self.visual_encoder.parameters(), None)
        vision_dtype = vision_param.dtype if vision_param is not None else images.dtype
        if images.dtype != vision_dtype:
            images = images.to(dtype=vision_dtype)

        autocast_dtype = vision_dtype if vision_dtype in (torch.float16, torch.bfloat16) else None
        with self.maybe_autocast(dtype=autocast_dtype):
            image_embeds = self.ln_vision(self.visual_encoder(images))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        return {"embeds": image_embeds, "atts": image_atts}

    def forward(self, samples: Dict[str, torch.Tensor]):
        images: torch.Tensor = samples["image"]

        controller_tokens = self._prepare_controller(images)
        self.Qformer.set_controller_state(controller_tokens)

        vision = self._encode_vision(images)

        query_tokens = self.query_tokens.expand(vision["embeds"].shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=vision["embeds"],
            encoder_attention_mask=vision["atts"],
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long, device=inputs_opt.device)

        questions = samples["text_input"]
        answers = samples.get("answers")
        prompt_prefix = [f"Question: {q} Answer:" for q in questions]

        if answers is not None:
            text_sequences = [f"{p} {a}".strip() for p, a in zip(prompt_prefix, answers)]
        else:
            text_sequences = [f"{p} " for p in prompt_prefix]

        self.opt_tokenizer.padding_side = "right"
        opt_tokens = self.opt_tokenizer(
            text_sequences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(inputs_opt.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id,
            -100,
        )
        if answers is not None:
            prompt_tokens = self.opt_tokenizer(
                [f"{p} " for p in prompt_prefix],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False,
            ).to(inputs_opt.device)
            prefix_lengths = prompt_tokens.attention_mask.sum(dim=1)
            for idx, prefix_len in enumerate(prefix_lengths):
                mask_len = min(1 + int(prefix_len.item()), targets.size(1))
                targets[idx, :mask_len] = -100
        elif self.prompt:
            targets[:, : self.prompt_length] = -100

        empty_targets = torch.ones_like(atts_opt, dtype=torch.long).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=targets,
                return_dict=True,
            )
        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate(
        self,
        samples: Dict[str, torch.Tensor],
        num_beams: int = 5,
        max_new_tokens: int = 30,
        min_new_tokens: int = 1,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        use_nucleus_sampling: bool = False,
    ):
        images: torch.Tensor = samples["image"]

        controller_tokens = self._prepare_controller(images)
        self.Qformer.set_controller_state(controller_tokens)

        vision = self._encode_vision(images)
        query_tokens = self.query_tokens.expand(vision["embeds"].shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=vision["embeds"],
            encoder_attention_mask=vision["atts"],
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long, device=inputs_opt.device)

        prompt_text = samples["prompt"] if "prompt" in samples else self.prompt
        if isinstance(prompt_text, str):
            prompt_text = [prompt_text] * inputs_opt.size(0)

        opt_tokens = self.opt_tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
        ).to(inputs_opt.device)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        
        # 计算输入序列长度（用于后续提取新生成的token）
        input_length = inputs_embeds.shape[1]

        with self.maybe_autocast():
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                eos_token_id=self.eos_token_id,
            )
        
        # 提取新生成的token（排除输入部分）
        # 当使用inputs_embeds时，outputs可能包含输入，需要手动提取新token
        generated_token_ids = outputs[:, input_length:]

        generated_text = self.opt_tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        return generated_text


