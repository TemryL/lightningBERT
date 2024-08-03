import os
import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import BertTokenizer
from importlib.machinery import SourceFileLoader
from bert import CLSBERT
from glue.data import GLUEDataModule
from glue.model import GLUETransformer
from transformers.utils import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the config file.", type=str, required=True)
    parser.add_argument("--task_name", help="Name of the GLUE task.", type=str, required=True)
    parser.add_argument("--nb_epochs", help="Number of epochs.", type=int, required=True)
    parser.add_argument("--nb_gpus", help="Number of GPUs per node.", type=int, required=True)
    parser.add_argument("--nb_nodes", help="Number of nodes.", type=int, required=True)
    parser.add_argument("--run_name", help="Name of the run.", type=str, required=True)
    parser.add_argument("--nb_workers", help="Number of workers.", type=int, default=None)
    parser.add_argument("--ckpt_path", help="Path to a pre-trained BERT model checkpoint", type=str, required=True)
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory for data loading", default=False)
    return parser.parse_args()

    
def main():
    L.seed_everything(42)
    logging.set_verbosity_error()
    
    # Parse arguments:
    args = parse_args()
    cfg = SourceFileLoader("config", args.config).load_module()
    task_name = args.task_name
    nb_epochs = args.nb_epochs
    nb_gpus = args.nb_gpus
    nb_nodes = args.nb_nodes
    run_name = args.run_name
    ckpt_path = args.ckpt_path
    nb_workers = args.nb_workers
    pin_memory = args.pin_memory
    
    # Create tokenizer and data module:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')      
    dm = GLUEDataModule(
        tokenizer,
        task_name = task_name,
        max_seq_length = cfg.max_seq_length,
        train_batch_size = cfg.train_batch_size,
        eval_batch_size = cfg.eval_batch_size,
        pin_memory = pin_memory,
        num_workers = nb_workers if nb_workers else os.cpu_count()
    )
    dm.setup("fit")
    
    # Create model:    
    cls_bert = CLSBERT.from_pretrained_bert(ckpt_path, dm.num_labels)
    model = GLUETransformer(
        model = cls_bert,
        num_labels = dm.num_labels,
        eval_splits = dm.eval_splits,
        task_name = dm.task_name,
        learning_rate = cfg.learning_rate,
        adamw_epsilon = cfg.adamw_epsilon,
        adamw_betas = cfg.adamw_betas,
        warmup_steps = cfg.warmup_steps,
        weight_decay = cfg.weight_decay
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
        max_epochs = nb_epochs,
        devices = nb_gpus,
        num_nodes = nb_nodes,
        log_every_n_steps = 10,
        strategy = "ddp",
        logger = logger, 
        callbacks = [lr_monitor, checkpoint_callback],
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar = True,
        fast_dev_run = False
    )

    # Fit model:
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()