# TO BE NAMED

```bash
export HF_HOME=...
export HF_TOKEN=...
python mlm/prepare_data.py
```

## Pretrain on Wikipedia

## Fine-tuning on GLUE

### Train
```bash
python train.py \
--task_name='cola' \
--nb_epochs=2 \
--nb_devices=1 \
--nb_nodes=1 \
--run_name='v0' \
--save_top_k=1
```

### Test
```bash
```