import os
import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from importlib.machinery import SourceFileLoader
from bert import BERTConfig, BERT
from mlm.data import WikipediaMLMDataModule
from mlm.model import MLMTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the config file.", type=str, required=True)
    parser.add_argument("--nb_epochs", help="Number of epochs.", type=int, required=True)
    parser.add_argument("--nb_gpus", help="Number of GPUs per node.", type=int, required=True)
    parser.add_argument("--nb_nodes", help="Number of nodes.", type=int, required=True)
    parser.add_argument("--run_name", help="Name of the run.", type=str, required=True)
    parser.add_argument("--nb_workers", help="Number of workers.", type=int, default=None)
    parser.add_argument("--ckpt_path", help="Path to a checkpoint to resume from.", type=str, default=None)
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned memory for data loading", default=False)
    return parser.parse_args()

    
def main():
    L.seed_everything(42)
    
    # Parse arguments:
    args = parse_args()
    cfg = SourceFileLoader("config", args.config).load_module()
    nb_epochs = args.nb_epochs
    nb_gpus = args.nb_gpus
    nb_nodes = args.nb_nodes
    run_name = args.run_name
    ckpt_path = args.ckpt_path
    nb_workers = args.nb_workers
    pin_memory = args.pin_memory
    
    # Create tokenizer and data module: 
    dm = WikipediaMLMDataModule(
        batch_size = cfg.batch_size,
        num_workers = nb_workers if nb_workers else os.cpu_count(), 
        train_val_split = cfg.train_val_split,
        mlm_probability = cfg.mlm_probability,
        pin_memory = pin_memory
    )
    dm.setup("fit")
    
    # Create model:
    bert_config = BERTConfig()
    bert_config.update(**cfg.bert_config)
    bert = BERT(bert_config)
    model = MLMTransformer(
        model = bert,
        learning_rate = cfg.learning_rate,
        adamw_epsilon = cfg.adamw_epsilon,
        adamw_betas = cfg.adamw_betas,
        warmup_steps = cfg.warmup_steps,
        weight_decay = cfg.weight_decay
    )
    
    # Set callbacks:
    lr_monitor = LearningRateMonitor(logging_interval='step')
    interval_ckpt = ModelCheckpoint(
        dirpath = f'ckpts/{run_name}',
        filename = 'last-{epoch}-{step}',
        every_n_train_steps = 2000,
    )
    val_ckpt = ModelCheckpoint(
        dirpath = f'ckpts/{run_name}',
        filename = 'best-{epoch}-{step}',
        monitor = 'val/loss',
        mode = 'min',
        save_top_k = 1,
        save_on_train_epoch_end = False,
    )
    epoch_ckpt = ModelCheckpoint(
        dirpath = f'ckpts/{run_name}/epochs',
        filename = '{epoch}',
        save_on_train_epoch_end = True,
        save_top_k = -1
    )

    # Set logger:
    logger = WandbLogger(project='lightningBERT', name=run_name)
    logger.watch(model)
    
    # Set trainer:
    trainer = L.Trainer(
        max_epochs = nb_epochs,
        devices = nb_gpus,
        num_nodes = nb_nodes,
        log_every_n_steps = 10,
        val_check_interval = 2000,
        strategy = "ddp",
        logger = logger, 
        callbacks = [lr_monitor, interval_ckpt, val_ckpt, epoch_ckpt],
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar = True,
        fast_dev_run = False
    )

    # Fit model:
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()