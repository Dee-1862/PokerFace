import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("Success! Your RTX 4060 is linked.")
else:
    print("Error: Still using CPU. Did the install command finish?")