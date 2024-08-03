import torch
import lightning as L
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class MLMTransformer(L.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 10000,
        weight_decay: float = 0.0
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs['loss']
        self.validation_step_outputs.append({'loss': loss})
        return loss
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        loss = torch.tensor([x['loss'] for x in outputs], device=self.device).mean()
        self.log("val/loss", loss, sync_dist=True)
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]