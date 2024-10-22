# Code taken and modified from: https://lightning.ai/docs/pytorch/LTS/notebooks/lightning_examples/text-transformers.html

from datetime import datetime
from typing import Optional

import evaluate
import torch
import lightning as L
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        model: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adamw_epsilon: float = 1e-8,
        adamw_betas: tuple = (0.9, 0.98),
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.log("train/loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs['loss'], outputs['logits']

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        output = {"loss": val_loss, "preds": preds, "labels": labels}
        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val/loss_{split}", loss, prog_bar=True, sync_dist=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True, sync_dist=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adamw_epsilon, betas=self.hparams.adamw_betas)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]