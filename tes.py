import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory allocated:", torch.cuda.memory_allocated(0))
    print("GPU memory cached:", torch.cuda.memory_reserved(0))
else:
    print("PyTorch is running on CPU.")