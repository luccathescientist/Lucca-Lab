import torch
import triton
import triton.language as tl
import time
import subprocess
import json
import os

# --- The Self-Healing Watchdog Logic ---

class KernelWatchdog:
    def __init__(self, model_name="DeepSeek-R1"):
        self.model_name = model_name
        self.history = []

    def get_gpu_status(self):
        # Simulated check for Blackwell RTX 6000 stats
        try:
            res = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,nounits,noheader'])
            used, free, util = map(int, res.decode().split(','))
            return {"used": used, "free": free, "util": util}
        except:
            return {"used": 0, "free": 96000, "util": 0}

    def consult_reasoning_model(self, error_msg, current_config):
        print(f"DEBUG: Consulting {self.model_name} for error: {error_msg}")
        # In a real scenario, this would call the R1 API or local model.
        # We'll simulate the "Reasoning" outcome for the experiment.
        
        if "out of memory" in error_msg.lower() or "illegal memory access" in error_msg.lower():
            new_config = {
                "BLOCK_SIZE_M": max(16, current_config["BLOCK_SIZE_M"] // 2),
                "BLOCK_SIZE_N": max(16, current_config["BLOCK_SIZE_N"] // 2),
                "num_warps": current_config["num_warps"],
                "num_stages": max(1, current_config["num_stages"] - 1)
            }
            return new_config, "Heuristic: Reduced block sizes to mitigate memory pressure/alignment issues."
        return current_config, "No changes suggested."

# --- The "Healing" Triton Kernel (Matrix Multiplication) ---

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def run_matmul(M, N, K, config):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"]
    )
    return c

# --- Simulation of Failure and Healing ---

if __name__ == "__main__":
    watchdog = KernelWatchdog()
    
    # Intentionally aggressive config that might fail on smaller tensors or cause issues
    current_config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "num_warps": 8,
        "num_stages": 4
    }
    
    M, N, K = 2048, 2048, 2048
    
    print(f"Initial Config: {current_config}")
    
    try:
        # Simulate a run that fails (we'll catch a simulated or real error)
        # For this test, let's pretend it OOMed if we were pushing 16k+
        # But we'll force a "Healing" cycle by catching a hypothetical error.
        raise RuntimeError("CUDA error: out of memory during kernel launch")
        
    except Exception as e:
        print(f"Caught Error: {e}")
        new_config, reason = watchdog.consult_reasoning_model(str(e), current_config)
        print(f"Reasoning: {reason}")
        print(f"Healed Config: {new_config}")
        
        # Verify it runs with the healed config
        print("Running with healed config...")
        output = run_matmul(M, N, K, new_config)
        print("Kernel executed successfully.")
        
        with open("healing_log.json", "w") as f:
            json.dump({
                "error": str(e),
                "reasoning": reason,
                "original_config": current_config,
                "healed_config": new_config,
                "status": "SUCCESS"
            }, f, indent=2)
