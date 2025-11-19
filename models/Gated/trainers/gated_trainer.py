"""High-level trainer for the gated BLIP-2 project."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.Gated.modules import Blip2OPTGated
from models.Gated.utils import (
    build_dataloaders,
    build_style_token_ids,
    compute_consistency_kl,
    compute_style_hit_rate,
    save_best_checkpoint,
)


logger = logging.getLogger(__name__)


class GatedTrainer:
    """Orchestrates training with frozen encoders and gated cross-attention."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_metric = float("-inf")

        self.train_loader, self.val_loader, self.test_loader, self.metadata = build_dataloaders(config)
        self.model = self._build_model().to(self.device)
        self.image_style_head = self._build_style_head().to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=self._use_amp)

        self.style_token_ids = self._prepare_style_tokens()
        self.style_token_to_label_map = self._build_style_token_to_label_map()

    @property
    def _training_cfg(self) -> Dict:
        return self.config.get("training", {})

    @property
    def _loss_cfg(self) -> Dict:
        return self.config.get("loss", {})

    @property
    def _use_amp(self) -> bool:
        return bool(self._training_cfg.get("use_amp", True)) and self.device.type == "cuda"

    def _build_model(self) -> Blip2OPTGated:
        model_cfg = self.config.get("model", {})
        paths_cfg = self.config.get("paths", {})

        qformer_cfg = model_cfg.get("qformer", {})
        gating_cfg = model_cfg.get("gating", {})
        prompt_mapper_cfg = model_cfg.get("prompt_mapper", {})
        lora_cfg = model_cfg.get("lora", {})

        return Blip2OPTGated(
            vit_model=model_cfg.get("visual_encoder", "eva_clip_g"),
            img_size=model_cfg.get("img_size", 224),
            freeze_vit=model_cfg.get("freeze_visual_encoder", True),
            num_query_token=qformer_cfg.get("num_query_tokens", 32),
            opt_model=model_cfg.get("opt_model", "facebook/opt-2.7b"),
            prompt=model_cfg.get("prompt", ""),
            max_txt_len=qformer_cfg.get("max_txt_len", 32),
            efficientnet_checkpoint=paths_cfg.get("efficientnet_checkpoint"),
            efficientnet_output_dim=model_cfg.get("efficientnet_output_dim", 768),
            convert_from_blip_norm=model_cfg.get("convert_from_blip_norm", True),
            gating_config=gating_cfg,
            prompt_mapper_cfg=prompt_mapper_cfg,
            lora_config=lora_cfg,
        )

    def _build_style_head(self) -> torch.nn.Module:
        output_dim = self.config.get("model", {}).get("efficientnet_output_dim", 768)
        style_vocab_size = len(self.metadata.get("style_vocab", []))
        if style_vocab_size == 0:
            style_vocab_size = 1
        head = torch.nn.Linear(output_dim, style_vocab_size)
        torch.nn.init.trunc_normal_(head.weight, std=0.02)
        torch.nn.init.zeros_(head.bias)
        return head

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [
            p
            for name, p in self.model.named_parameters()
            if p.requires_grad
        ]
        params.extend(p for p in self.image_style_head.parameters() if p.requires_grad)

        if not params:
            raise ValueError("No trainable parameters found for optimizer.")

        return torch.optim.AdamW(
            params,
            lr=float(self._training_cfg.get("lr", 1e-4)),
            weight_decay=float(self._training_cfg.get("weight_decay", 0.01)),
        )

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
        scheduler_type = self._training_cfg.get("lr_scheduler", "cosine")
        if scheduler_type == "none":
            return None

        total_steps = len(self.train_loader) * int(self._training_cfg.get("max_epochs", 5))
        warmup_steps = int(self._training_cfg.get("warmup_steps", 100))

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            if scheduler_type == "cosine":
                return 0.5 * (1.0 + math.cos(progress * math.pi))
            return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _prepare_style_tokens(self) -> torch.Tensor:
        styles = self.metadata.get("style_vocab", [])
        if not styles:
            return torch.empty(0, dtype=torch.long, device=self.device)
        token_ids = build_style_token_ids(styles, self.model.opt_tokenizer)
        return token_ids.to(self.device)

    def _build_style_token_to_label_map(self) -> Dict[int, int]:
        """Map style token IDs to style label indices for KL computation."""
        styles = self.metadata.get("style_vocab", [])
        style_to_id = self.metadata.get("style_to_id", {})
        if not styles or not style_to_id:
            return {}
        
        token_to_label = {}
        for style, label_id in style_to_id.items():
            tokens = self.model.opt_tokenizer(style, add_special_tokens=False).input_ids
            for token_id in tokens:
                token_to_label[token_id] = label_id
        return token_to_label

    def train(self) -> None:
        max_epochs = int(self._training_cfg.get("max_epochs", 5))
        for epoch in range(max_epochs):
            train_metrics = self._run_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            metric = -val_metrics["loss"]
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "style_head_state_dict": self.image_style_head.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if self._use_amp else None,
                "metrics": {"train": train_metrics, "val": val_metrics},
                "config": self.config,
            }
            checkpoint_path = self._training_cfg.get("checkpoint_path", "models/Gated/checkpoints/best.pt")
            checkpoint_path = str(Path(checkpoint_path).expanduser())
            self.best_metric = save_best_checkpoint(
                checkpoint_state,
                checkpoint_path,
                metric,
                self.best_metric,
            )

            logger.info(
                "Epoch %d | train_loss %.4f | val_loss %.4f | best_metric %.4f",
                epoch + 1,
                train_metrics["loss"],
                val_metrics["loss"],
                self.best_metric,
            )

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.image_style_head.train()

        grad_accum = int(self._training_cfg.get("gradient_accumulation", 1))
        main_weight = float(self._loss_cfg.get("main_weight", 1.0))
        style_weight = float(self._loss_cfg.get("style_hit_weight", 0.0))
        kl_weight = float(self._loss_cfg.get("consistency_kl_weight", 0.0))

        total_loss = 0.0
        total_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
            samples = self._move_batch_to_device(batch)

            with autocast(enabled=self._use_amp):
                outputs = self.model(samples)
                loss = outputs["loss"] * main_weight

                if style_weight > 0 and self.style_token_ids.numel() > 0:
                    style_rate = compute_style_hit_rate(outputs["logits"], self.style_token_ids)
                    loss = loss + style_weight * (1.0 - style_rate)

                if kl_weight > 0 and self.style_token_ids.numel() > 0 and self.style_token_to_label_map:
                    image_style_logits = self.image_style_head(outputs["efficientnet_embeddings"])
                    text_style_logits = self._compute_text_style_logits(outputs["logits"])
                    if text_style_logits is not None:
                        kl_loss = compute_consistency_kl(image_style_logits, text_style_logits)
                        loss = loss + kl_weight * kl_loss

            if self._use_amp:
                self.scaler.scale(loss / grad_accum).backward()
            else:
                (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                if self._use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), self._training_cfg.get("max_grad_norm", 1.0))

                if self._use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler:
                    self.scheduler.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        return {"loss": avg_loss}

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        self.image_style_head.eval()

        main_weight = float(self._loss_cfg.get("main_weight", 1.0))
        style_weight = float(self._loss_cfg.get("style_hit_weight", 0.0))
        kl_weight = float(self._loss_cfg.get("consistency_kl_weight", 0.0))

        total_loss = 0.0
        total_style = 0.0
        total_kl = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                samples = self._move_batch_to_device(batch)
                outputs = self.model(samples)

                loss = outputs["loss"] * main_weight
                if style_weight > 0 and self.style_token_ids.numel() > 0:
                    style_rate = compute_style_hit_rate(outputs["logits"], self.style_token_ids)
                    loss = loss + style_weight * (1.0 - style_rate)
                    total_style += style_rate.item()

                if kl_weight > 0 and self.style_token_ids.numel() > 0 and self.style_token_to_label_map:
                    image_style_logits = self.image_style_head(outputs["efficientnet_embeddings"])
                    text_style_logits = self._compute_text_style_logits(outputs["logits"])
                    if text_style_logits is not None:
                        kl_loss = compute_consistency_kl(image_style_logits, text_style_logits)
                        loss = loss + kl_weight * kl_loss
                        total_kl += kl_loss.item()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        avg_style = total_style / max(1, total_batches) if style_weight > 0 else 0.0
        avg_kl = total_kl / max(1, total_batches) if kl_weight > 0 else 0.0

        return {
            "loss": avg_loss,
            "style_hit_rate": avg_style,
            "kl_loss": avg_kl,
        }

    def _compute_text_style_logits(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute text style logits by aggregating token logits per style label."""
        if not self.style_token_to_label_map:
            return None
        
        style_vocab_size = len(self.metadata.get("style_vocab", []))
        if style_vocab_size == 0:
            return None
        
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Aggregate logits per style label
        text_style_logits = torch.zeros(batch_size, style_vocab_size, device=device, dtype=logits.dtype)
        counts = torch.zeros(style_vocab_size, device=device)
        
        for token_id, label_id in self.style_token_to_label_map.items():
            if token_id < vocab_size:
                token_logits = logits[..., token_id]  # (batch, seq_len)
                text_style_logits[:, label_id] += token_logits.mean(dim=-1)
                counts[label_id] += 1
        
        # Normalize by count (avoid division by zero)
        counts = torch.clamp(counts, min=1.0)
        text_style_logits = text_style_logits / counts.unsqueeze(0)
        
        return text_style_logits

    def _trainable_parameters(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p
        for p in self.image_style_head.parameters():
            if p.requires_grad:
                yield p

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        images = batch["image"].to(self.device, non_blocking=True)
        text_input = batch["text_input"]
        return {
            "image": images,
            "text_input": text_input,
        }

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Dict:
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        model_state = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint
        self.model.load_state_dict(model_state, strict=strict)
        style_state = checkpoint.get("style_head_state_dict")
        if style_state:
            self.image_style_head.load_state_dict(style_state, strict=strict)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except RuntimeError:
                logger.warning("Optimizer state mismatch; starting with fresh optimizer.")
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except RuntimeError:
                logger.warning("Scheduler state mismatch; starting with new scheduler.")
        if self._use_amp and checkpoint.get("scaler_state_dict"):
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except RuntimeError:
                logger.warning("Grad scaler state mismatch; resetting scaler.")
        return checkpoint

