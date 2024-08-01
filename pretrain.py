import os
import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import BertConfig, BertForMaskedLM
from bert import BERTConfig, BERT
from mlm.data import WikipediaMLMDataModule
from mlm.model import MLMTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", help="Number of epochs.", type=int, required=True)
    parser.add_argument("--nb_devices", help="Number of GPUs per node.", type=int, required=True)
    parser.add_argument("--nb_nodes", help="Number of nodes.", type=int, required=True)
    parser.add_argument("--run_name", help="Name of the run.", type=str, required=True)
    parser.add_argument("--nb_workers", help="Number of workers.", type=int, default=None)
    parser.add_argument("--ckpt_path", help="Path to a checkpoint to resume from.", type=str, default=None)
    return parser.parse_args()

    
def main():
    L.seed_everything(42)
    
    # Parse arguments:
    args = parse_args()
    nb_epochs = args.nb_epochs
    nb_devices = args.nb_devices
    nb_nodes = args.nb_nodes
    run_name = args.run_name
    ckpt_path = args.ckpt_path
    nb_workers = args.nb_workers
    
    # Create tokenizer and data module: 
    dm = WikipediaMLMDataModule(
        batch_size=32,
        num_workers=nb_workers if nb_workers else os.cpu_count(), 
        train_val_split=0.95,
        mlm_probability=0.15,
        pin_memory=True
    )
    dm.setup("fit")
    
    # Create model:
    # config = BertConfig.from_pretrained('bert-base-uncased')
    # bert = BertForMaskedLM(config)
    config = BERTConfig()
    bert = BERT(config)
    model = MLMTransformer(
        model=bert,
        learning_rate=1e-4,
        adam_epsilon=1e-8,
        warmup_steps=10000,
        weight_decay=0.01
    )
    
    # Set callbacks + logger + trainer:
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'ckpts/{run_name}',
        filename='{epoch}-{step}',
        save_on_train_epoch_end=True,
    )
    logger = WandbLogger(project='lightningBERT', name=run_name)
    logger.watch(model)
    
    trainer = L.Trainer(
        max_epochs=nb_epochs,
        devices=nb_devices,
        num_nodes=nb_nodes,
        log_every_n_steps=10,
        strategy="ddp",
        logger=logger, 
        callbacks=[lr_monitor, checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar=True,
        fast_dev_run=False
    )

    # Fit model:
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()