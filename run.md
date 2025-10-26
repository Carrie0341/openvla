Run Deploy Server (Unity)

```sh
python vla-scripts/deploy_unity.py --openvla_path="openvla/openvla-7b" --lora_path="logs/openvla-7b+et_vla+b4+lr-0.0005+lora-r32+dropout-0.0/" --host="0.0.0.0" --port=8000
```
python vla-scripts/deploy_unity.py --openvla_path="checkpoint/" --host="0.0.0.0" --port=8000