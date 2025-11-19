"""
High-level trainer orchestrating KV-modulated BLIP-2 fine-tuning.
"""

from __future__ import annotations

import json
import math
import random
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, List

import numpy as np

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.KV.datasets import build_artquest_dataloaders
from models.KV.utils.metrics import exact_match, average
from models.KV.utils.model_loader import load_kv_model


logger = logging.getLogger(__name__)


class KVTrainer:
    """Encapsulates the training/eval loop for the KV model."""

    def __init__(self, config: Dict, resume_path: Optional[str] = None) -> None:
        self.config = config
        self.project_cfg = config.get("project", {})
        self.training_cfg = config.get("training", {})
        self.model_cfg = config.get("model", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resume_from_cfg = self.training_cfg.get("resume_from")
        resume_path = resume_path or resume_from_cfg
        self.resume_path = Path(resume_path).expanduser() if resume_path else None

        self._set_seed(int(self.project_cfg.get("seed", 1337)))

        output_root = Path(self.project_cfg.get("output_dir", "models/KV/runs"))
        if self.resume_path:
            if not self.resume_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {self.resume_path}")
            self.run_dir = self.resume_path.parent
            logger.info("Resuming training from %s", self.resume_path)
        else:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.run_dir = output_root / timestamp
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "training_log.json"
        self.log_records: List[Dict] = self._load_existing_logs()

        self.train_loader, self.val_loader, self.test_loader, self.metadata = self._build_dataloaders()
        self.model = load_kv_model(config, device=self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters detected.")

        self.optimizer = AdamW(
            trainable_params,
            lr=float(self.training_cfg.get("lr", 5e-5)),
            weight_decay=float(self.training_cfg.get("weight_decay", 0.01)),
        )
        total_steps = len(self.train_loader) * int(self.training_cfg.get("epochs", 1))
        warmup_steps = int(self.training_cfg.get("warmup_steps", 0))
        self.scheduler = LambdaLR(
            self.optimizer,
            lambda step: self._lr_lambda(step, warmup_steps, total_steps),
        )

        self.use_amp = bool(self.training_cfg.get("use_amp", True)) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_accum = int(self.training_cfg.get("grad_accumulation", 1))
        self.max_grad_norm = float(self.training_cfg.get("max_grad_norm", 1.0))
        self.start_epoch = 1
        if self.log_records:
            self.best_val = min(
                entry.get("val_loss", float("inf")) for entry in self.log_records
            )
        else:
            self.best_val = float("inf")

        self._maybe_resume()

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------ #
    def _build_dataloaders(self):
        data_cfg = self.config.get("data", {})
        artquest_cfg = data_cfg.get("artquest", {})
        dataset_cfg = artquest_cfg.get("dataset_cfg", {})
        batch_size = int(data_cfg.get("batch_size", 4))
        num_workers = int(data_cfg.get("num_workers", 4))
        limit = data_cfg.get("limit")

        train_csv = Path(artquest_cfg["train_csv"])
        val_csv = Path(artquest_cfg["val_csv"])
        test_csv = Path(artquest_cfg.get("test_csv")) if artquest_cfg.get("test_csv") else None
        image_root = Path(artquest_cfg["image_root"])

        return build_artquest_dataloaders(
            train_csv,
            val_csv,
            test_csv,
            image_root,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_cfg=dataset_cfg,
            limit=limit,
        )

    # ------------------------------------------------------------------ #
    def _lr_lambda(self, step: int, warmup: int, total: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = float(step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(progress * math.pi)))

    # ------------------------------------------------------------------ #
    def train(self) -> None:
        epochs = int(self.training_cfg.get("epochs", 1))
        if self.start_epoch > epochs:
            logger.warning(
                "Configured epochs=%d but checkpoint already at epoch=%d. Nothing to train.",
                epochs,
                self.start_epoch - 1,
            )
            return

        for epoch in range(self.start_epoch, epochs + 1):
            train_metrics = self._run_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["exact_match"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["exact_match"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._append_log(record)

            if val_metrics["loss"] < self.best_val:
                self.best_val = val_metrics["loss"]
                self._save_checkpoint(epoch, record)

    # ------------------------------------------------------------------ #
    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss: list[float] = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            loss = self._forward_batch(batch)
            loss = loss / self.grad_accum
            self.scaler.scale(loss).backward()

            if step % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            running_loss.append(loss.item())
            pbar.set_postfix(loss=sum(running_loss) / len(running_loss))

        metrics = {
            "loss": average(running_loss),
            "exact_match": self._quick_eval_batches(self.train_loader, max_batches=5),
        }
        return metrics

    # ------------------------------------------------------------------ #
    def _forward_batch(self, batch: Dict) -> torch.Tensor:
        samples = {
            "image": batch["image"].to(self.device, non_blocking=True),
            "text_input": batch["text_input"],
            "answers": batch.get("answers") or batch.get("answer"),
        }
        with autocast(enabled=self.use_amp):
            output = self.model(samples)
            loss = output["loss"]
        return loss

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        loader: DataLoader,
        max_batches: Optional[int] = None,
        desc: str = "Evaluating",
    ) -> Dict[str, float]:
        self.model.eval()
        losses = []
        exact_matches = []
        eval_loader = tqdm(loader, desc=desc, dynamic_ncols=True)
        with torch.no_grad():
            for idx, batch in enumerate(eval_loader):
                samples = {
                    "image": batch["image"].to(self.device, non_blocking=True),
                    "text_input": batch["text_input"],
                    "answers": batch.get("answers") or batch.get("answer"),
                }
                output = self.model(samples)
                losses.append(output["loss"].item())

                preds = self._generate_answers(batch)
                refs = batch.get("answers") or batch.get("answer")
                exact_matches.append(exact_match(preds, refs))

                if max_batches and (idx + 1) >= max_batches:
                    break
        self.model.train()
        return {
            "loss": average(losses),
            "exact_match": average(exact_matches),
        }

    # ------------------------------------------------------------------ #
    def _quick_eval_batches(self, loader: DataLoader, max_batches: int) -> float:
        self.model.eval()
        matches = []
        eval_loader = tqdm(loader, desc="Quick Eval", dynamic_ncols=True, leave=False)
        with torch.no_grad():
            for idx, batch in enumerate(eval_loader):
                preds = self._generate_answers(batch)
                refs = batch.get("answers") or batch.get("answer")
                matches.append(exact_match(preds, refs))
                if (idx + 1) >= max_batches:
                    break
        self.model.train()
        return average(matches)

    # ------------------------------------------------------------------ #
    def _generate_answers(self, batch: Dict) -> Iterable[str]:
        prompts = [f"Question: {q} Answer:" for q in batch["text_input"]]
        gen_samples = {
            "image": batch["image"].to(self.device, non_blocking=True),
            "prompt": prompts,
        }
        max_new_tokens = int(self.model_cfg.get("max_new_tokens", 10))
        outputs = self.model.generate(gen_samples, num_beams=2, max_new_tokens=max_new_tokens)
        return outputs

    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, epoch: int, record: Dict) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "record": record,
            "config": self.config,
        }
        path = self.run_dir / f"checkpoint_epoch_{epoch:02d}.pt"
        torch.save(ckpt, path)

    def _load_existing_logs(self) -> List[Dict]:
        if not self.log_path.exists():
            return []
        text = self.log_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        # Try direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Attempt to convert concatenated objects into JSON array
        try:
            normalized = "[" + re.sub(r"}\s*{", "},{", text) + "]"
            parsed = json.loads(normalized)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Fallback: incremental brace matching
        logs: List[Dict] = []
        buffer = ""
        balance = 0
        for char in text:
            buffer += char
            if char == "{":
                balance += 1
            elif char == "}":
                balance -= 1
            if balance == 0 and buffer.strip():
                try:
                    logs.append(json.loads(buffer))
                except json.JSONDecodeError:
                    pass
                buffer = ""
        return logs

    def _append_log(self, record: Dict) -> None:
        self.log_records.append(record)
        with self.log_path.open("w", encoding="utf-8") as f:
            json.dump(self.log_records, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def _maybe_resume(self) -> None:
        if not self.resume_path:
            return
        checkpoint = torch.load(self.resume_path, map_location=self.device)
        model_state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(model_state, strict=False)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        record = checkpoint.get("record") or {}
        self.best_val = record.get("val_loss", record.get("loss", self.best_val))
        logger.info(
            "Checkpoint loaded. Resuming from epoch %d (best val loss %.4f).",
            self.start_epoch,
            self.best_val,
        )

