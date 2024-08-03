# TO BE NAMED

```bash
export HF_HOME=...
export HF_TOKEN=...
python mlm/prepare_data.py
```
## Pretrain on Wikipedia
```
python pretrain.py \
    --nb_epochs=5 \
    --nb_gpus=1 \
    --nb_nodes=1 \
    --nb_workers=20 \
    --pin_memory \
    --config='configs/pretrain_cfg.py' \
    --run_name='pretrain_base'
```

## Fine-tuning on GLUE
```
python finetune.py \
    --nb_epochs=5 \
    --nb_gpus=1 \
    --nb_nodes=1 \
    --nb_workers=20 \
    --pin_memory \
    --config='configs/finetune_cfg.py' \
    --ckpt_path='ckpts/v3/best-epoch=0-step=10.ckpt' \
    --task_name='cola' \
    --run_name='finetune_base'
```