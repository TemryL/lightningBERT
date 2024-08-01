import os
import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import BertTokenizer
from transformers import AutoConfig, BertForSequenceClassification
from glue.data import GLUEDataModule
from glue.model import GLUETransformer
from transformers.utils import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="Name of the GLUE task.", type=str, required=True)
    parser.add_argument("--nb_epochs", help="Number of epochs.", type=int, required=True)
    parser.add_argument("--nb_devices", help="Number of GPUs per node.", type=int, required=True)
    parser.add_argument("--nb_nodes", help="Number of nodes.", type=int, required=True)
    parser.add_argument("--run_name", help="Name of the run.", type=str, required=True)
    parser.add_argument("--nb_workers", help="Number of workers.", type=int, default=None)
    parser.add_argument("--ckpt_path", help="Path to a checkpoint to resume from.", type=str, default=None)
    return parser.parse_args()

    
def main():
    L.seed_everything(42)
    logging.set_verbosity_error()
    
    # Parse arguments:
    args = parse_args()
    task_name = args.task_name
    nb_epochs = args.nb_epochs
    nb_devices = args.nb_devices
    nb_nodes = args.nb_nodes
    run_name = args.run_name
    ckpt_path = args.ckpt_path
    nb_workers = args.nb_workers
    
    # Create tokenizer and data module:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')      
    dm = GLUEDataModule(
        tokenizer,
        task_name=task_name,
        pin_memory=True,
        num_workers=nb_workers if nb_workers else os.cpu_count()
    )
    dm.setup("fit")
    
    # Create model:
    config = AutoConfig.from_pretrained('bert-base-uncased', num_labels=dm.num_labels)
    bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    model = GLUETransformer(
        model=bert,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
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
        fast_dev_run=True
    )

    # Fit model:
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()