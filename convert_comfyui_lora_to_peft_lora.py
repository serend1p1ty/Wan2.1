import safetensors.torch

lora_path = "epoch10_inflate/adapter_model.safetensors"
ckpt = safetensors.torch.load_file(lora_path)
ckpt = {k.replace("diffusion_model", "base_model.model"):v for k, v in ckpt.items()}
safetensors.torch.save_file(ckpt, lora_path)
