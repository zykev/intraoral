# modelscope download --model="Qwen/Qwen2.5-0.5B-Instruct" --local_dir /home/peiliu/Checkpoints


from modelscope import snapshot_download
model_dir = snapshot_download("facebook/sam3", local_dir="/home/peiliu/Checkpoints")