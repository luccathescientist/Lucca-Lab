import torch

def check_gpu():
    print("--- Lucca's GPU Health Check ---")
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected.")
        return

    device_count = torch.cuda.device_count()
    print(f"✅ Found {device_count} GPU(s).")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Check current utilization
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  Memory Reserved: {reserved:.2f} GB")
        print(f"  Memory Allocated: {allocated:.2f} GB")

if __name__ == "__main__":
    check_gpu()
