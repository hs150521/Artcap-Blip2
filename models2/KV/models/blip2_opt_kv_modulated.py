"""
Modified BLIP2-OPT model with KV modulation support.

This module extends the original BLIP2-OPT model to support KV-Prefix injection
from EfficientNet features in the cross-attention layers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add LAVIS to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "blip2" / "LAVIS"))

from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import AutoTokenizer, OPTForCausalLM

# Import our modified Qformer
from .qformer_kv_modulated import BertLMHeadModelKVModulated

# Import KV-Prefix generator
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_modulation.kv_prefix_generator import KVPrefixGenerator
from utils.efficientnet_loader import load_efficientnet_model, extract_efficientnet_features


class Blip2OPTKVModulated(Blip2Base):
    """
    BLIP2-OPT model with KV modulation support.
    
    This model extends Blip2OPT to support KV-Prefix injection from EfficientNet
    features in the Qformer cross-attention layers.
    """
    
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        # KV modulation specific parameters
        efficientnet_checkpoint: Optional[str] = None,
        num_prefix_tokens: int = 8,
        use_kv_modulation: bool = True,
    ):
        """
        Args:
            efficientnet_checkpoint: Path to EfficientNet checkpoint
            num_prefix_tokens: Number of prefix tokens for KV modulation
            use_kv_modulation: Whether to use KV modulation
        """
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
            logging.info("freeze vision encoder")
        
        # Initialize Qformer with KV modulation support
        self.Qformer, self.query_tokens = self.init_Qformer_kv_modulated(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        # Initialize OPT model (frozen)
        # Check if opt_model is a local path
        opt_model_path = opt_model
        if not Path(opt_model_path).exists():
            # If not a valid local path, try to resolve relative to repo root
            repo_root = Path(__file__).resolve().parents[3]
            potential_path = repo_root / opt_model_path
            if potential_path.exists():
                opt_model_path = str(potential_path)
            else:
                # Fall back to original behavior for remote models
                logging.warning(f"Local OPT model not found at {opt_model_path}, falling back to Hugging Face download")
        
        # For git-lfs format models, try to find the snapshot directory
        if Path(opt_model_path).exists():
            # Check if this is a git-lfs repository structure
            snapshots_dir = Path(opt_model_path) / "snapshots"
            if snapshots_dir.exists():
                # Get the first snapshot directory
                snapshot_dirs = list(snapshots_dir.iterdir())
                if snapshot_dirs:
                    opt_model_path = str(snapshot_dirs[0])
                    logging.info(f"Using snapshot directory: {opt_model_path}")
        
        # If we're using a local path but it's a BLIP2 model, fall back to the original model name
        # This is because the local directory might contain BLIP2 config instead of pure OPT config
        if Path(opt_model_path).exists():
            try:
                # Try to load the config to check if it's a BLIP2 model
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(opt_model_path)
                if hasattr(config, 'model_type') and config.model_type == 'blip-2':
                    logging.warning(f"Local path {opt_model_path} contains BLIP2 model, falling back to original OPT model name: facebook/opt-2.7b")
                    opt_model_path = "facebook/opt-2.7b"
            except Exception as e:
                logging.warning(f"Failed to check config type: {e}, using path as is")
        
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model_path, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model_path, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        
        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        
        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        
        # Initialize EfficientNet and KV-Prefix generator
        self.use_kv_modulation = use_kv_modulation
        if self.use_kv_modulation:
            repo_root = Path(__file__).resolve().parents[3]
            self.efficientnet_model, self.efficientnet_preprocess, _ = load_efficientnet_model(
                checkpoint_path=efficientnet_checkpoint,
                repo_root=str(repo_root),
            )
            
            # Freeze EfficientNet
            for param in self.efficientnet_model.parameters():
                param.requires_grad = False
            
            # Initialize KV-Prefix generator
            efficientnet_feat_dim = 1536  # EfficientNet-B3 feature dimension
            qformer_hidden_size = self.Qformer.config.hidden_size
            num_heads = self.Qformer.config.num_attention_heads
            
            self.kv_prefix_generator = KVPrefixGenerator(
                efficientnet_feat_dim=efficientnet_feat_dim,
                qformer_hidden_size=qformer_hidden_size,
                num_prefix_tokens=num_prefix_tokens,
                num_heads=num_heads,
            )
        else:
            self.efficientnet_model = None
            self.efficientnet_preprocess = None
            self.kv_prefix_generator = None
    
    def init_Qformer_kv_modulated(self, num_query_token, vision_width, cross_attention_freq=2):
        """Initialize Qformer with KV modulation support."""
        from transformers.models.bert.configuration_bert import BertConfig
        
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        Qformer = BertLMHeadModelKVModulated.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def _convert_to_efficientnet_preprocessing(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convert BLIP2 preprocessed images to EfficientNet preprocessing.
        
        Args:
            image: Tensor of shape (batch_size, 3, H, W) preprocessed with BLIP2 normalization
        
        Returns:
            Tensor preprocessed with EfficientNet normalization
        """
        # BLIP2 uses: mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        # EfficientNet uses ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        blip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        blip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        eff_mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        eff_std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        
        # Denormalize BLIP2 preprocessing
        denormalized = image * blip_std + blip_mean
        # Clamp to valid range [0, 1]
        denormalized = torch.clamp(denormalized, 0.0, 1.0)
        # Normalize with EfficientNet preprocessing
        eff_images = (denormalized - eff_mean) / eff_std
        
        return eff_images
    
    def forward(self, samples):
        original_image = samples["image"]
        image_device = original_image.device
        
        # Extract EfficientNet features and generate KV prefixes
        k_prefix = None
        v_prefix = None
        if self.use_kv_modulation and self.efficientnet_model is not None:
            # EfficientNet is in float32, so ensure input is float32 and run outside autocast
            # Disable autocast for EfficientNet to avoid dtype mismatches
            with torch.no_grad():
                # Temporarily disable any active autocast context
                eff_input = original_image
                # Ensure EfficientNet input is float32 (model is in float32)
                if eff_input.dtype != torch.float32:
                    eff_input = eff_input.to(dtype=torch.float32)
                # Convert BLIP2 preprocessed images to EfficientNet preprocessing
                eff_images = self._convert_to_efficientnet_preprocessing(eff_input)
                
                # Extract features - ensure EfficientNet model and input are both float32
                # by explicitly disabling autocast if it's active
                device_type = eff_images.device.type
                with torch.amp.autocast(enabled=False, device_type=device_type):
                    eff_features = extract_efficientnet_features(
                        self.efficientnet_model,
                        eff_images,
                        return_pooled=True,
                    )
            
            # Generate KV prefixes
            k_prefix, v_prefix = self.kv_prefix_generator(eff_features)
            
            # Set KV prefixes in Qformer
            self.Qformer.set_kv_prefix(k_prefix, v_prefix)
        
        # Handle visual encoder dtype - ensure input matches model dtype
        vision_image = original_image
        vision_param = next(self.visual_encoder.parameters(), None)
        if vision_param is not None:
            vision_dtype = vision_param.dtype
            # Convert input to match visual encoder dtype (likely fp16)
            if vision_image.dtype != vision_dtype:
                vision_image = vision_image.to(dtype=vision_dtype)
        else:
            vision_dtype = vision_image.dtype
        
        # Use autocast only if model is in fp16/bf16, otherwise disable it for visual encoder
        autocast_dtype = vision_dtype if vision_dtype in (torch.float16, torch.bfloat16) else None
        with self.maybe_autocast(dtype=autocast_dtype):
            image_embeds = self.ln_vision(self.visual_encoder(vision_image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            vision_image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image_device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image_device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image_device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """Generate text from images."""
        original_image = samples["image"]
        image_device = original_image.device
        
        # Extract EfficientNet features and generate KV prefixes
        if self.use_kv_modulation and self.efficientnet_model is not None:
            with torch.no_grad():
                eff_input = original_image
                if eff_input.dtype != torch.float32:
                    eff_input = eff_input.to(dtype=torch.float32)
                eff_images = self._convert_to_efficientnet_preprocessing(eff_input)
                eff_features = extract_efficientnet_features(
                    self.efficientnet_model,
                    eff_images,
                    return_pooled=True,
                )
            k_prefix, v_prefix = self.kv_prefix_generator(eff_features)
            self.Qformer.set_kv_prefix(k_prefix, v_prefix)
        
        vision_image = original_image
        vision_param = next(self.visual_encoder.parameters(), None)
        vision_dtype = vision_param.dtype if vision_param is not None else vision_image.dtype
        if vision_param is not None and vision_image.dtype != vision_dtype:
            vision_image = vision_image.to(dtype=vision_dtype)
        autocast_dtype = vision_dtype if vision_dtype in (torch.float16, torch.bfloat16) else None
        
        with self.maybe_autocast(dtype=autocast_dtype):
            image_embeds = self.ln_vision(self.visual_encoder(vision_image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                vision_image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                vision_image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * original_image.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image_device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
            
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
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text
    
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        """Predict answers for VQA tasks."""
        image = samples["image"]
        
        # Extract EfficientNet features and generate KV prefixes
        if self.use_kv_modulation and self.efficientnet_model is not None:
            with torch.no_grad():
                eff_input = image
                if eff_input.dtype != torch.float32:
                    eff_input = eff_input.to(dtype=torch.float32)

                device_type = eff_input.device.type
                with torch.amp.autocast(enabled=False, device_type=device_type):
                    eff_images = self._convert_to_efficientnet_preprocessing(eff_input)
                    eff_features = extract_efficientnet_features(
                        self.efficientnet_model,
                        eff_images,
                        return_pooled=True,
                    )

            target_dtype = next(self.kv_prefix_generator.parameters()).dtype
            if eff_features.dtype != target_dtype:
                eff_features = eff_features.to(dtype=target_dtype)
            k_prefix, v_prefix = self.kv_prefix_generator(eff_features)
            self.Qformer.set_kv_prefix(k_prefix, v_prefix)
        
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
            
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
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
