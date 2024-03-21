from transformers import BartModel, BartTokenizer
import torch
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.utils.benchmark as benchmark

def benchmark_model_forward_pass(model, inputs, device):
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = benchmark.Timer(
        stmt="model(**inputs)",
        globals={"model": model, "inputs": inputs},
    )
    result = t0.blocked_autorange()
    return result.mean * 1e6

def main():
    print("Working with model bart")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model_bart = BartModel.from_pretrained("facebook/bart-large").to(device)

    inputs = tokenizer("ACDEF", return_tensors="pt")

    with torch.no_grad():
        print("Benchmarking BART model with different attention mechanisms")

        # Benchmark without any specific attention mechanism flags or custom SDPA backend
        print(f"Default BART implementation runs in {benchmark_model_forward_pass(model_bart, inputs, device):.3f} microseconds")

        # Example of using custom SDPA backend, assuming hypothetical support or custom modifications
        backend_map = {
            SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
            SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
            SDPBackend.EFFICIENT_ATTENTION: {"enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
        }

        for backend, args in backend_map.items():
            with sdp_kernel(**args):
                try:
                    print(f"The {backend.name.lower()} implementation runs in {benchmark_model_forward_pass(model_bart, inputs, device):.3f} microseconds")
                except RuntimeError as e:
                    print(f"{backend.name} is not supported on this device. Reason: {e}")

if __name__ == "__main__":
    main()
