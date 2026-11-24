import torch

print("--- GPU Diagnostic Tool ---")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("\n[!] PyTorch cannot see your GPU. It is running on the CPU.")
