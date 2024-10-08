import torch
import sys

# Checks if you have access to any GPUs
def check_gpus():
    print("\n\nPython version:", sys.version)
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available. GPUs detected.")
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")

        for i in range(gpu_count):
            gpu = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {gpu.name}")
            print(f"  Compute Capability: {gpu.major}.{gpu.minor}")
            print(f"  Total Memory: {gpu.total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. No GPUs detected.")

    print("\nCUDA Device:")
    if torch.cuda.is_available():
        print(f"  Current Device: {torch.cuda.current_device()}")
        print(f"  Device Name: {torch.cuda.get_device_name(0)}\n\n")
    else:
        print("  No CUDA device available\n\n")

if __name__ == "__main__":
    check_gpus()