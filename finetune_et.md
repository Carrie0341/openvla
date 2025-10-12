Fine tune command

```sh
# Fine tune on et_vla 
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir data/et_vla \
  --dataset_name et_vla \
  --run_root_dir logs \
  --adapter_tmp_dir /tmp \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project "openvla-finetune" \
  --wandb_entity "miislab-ntust-national-taiwan-university-of-science-and-" \
  --save_steps 100
```

```sh
# test on LIBERO 
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir data/modified_libero_rlds \
  --dataset_name libero_10_no_noops \
  --run_root_dir logs \
  --adapter_tmp_dir /tmp \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project "openvla-finetune" \
  --wandb_entity "miislab-ntust-national-taiwan-university-of-science-and-" \
  --save_steps 3
```