from transformers import EsmModel, EsmTokenizer
import torch
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.utils.benchmark as benchmark

def benchmark_model_forward_pass(model, inputs):
    t0 = benchmark.Timer(
        stmt="model(**inputs)",
        globals={"model": model, "inputs": inputs},
    )
    result = t0.blocked_autorange()
    return result.mean * 1e6

def main():
    device = torch.device("cuda")
    print(f"Working with model esm on {device}")

    # Initialize tokenizer and model
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_esm_sdpa = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device).half()  # Convert model to float16
    model_esm_eager = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device).half()  # Convert model to float16

    # Prepare inputs
    
    sequences = ["ACDEF", "GHJKL", "MNOPQ"]  # Replace with your actual sequences
    inputs = tokenizer(sequences, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device

    # Convert model and inputs to float16


    with torch.no_grad():
        # the eager way
        print(f"Using implementation of attention {model_esm_eager.config._attn_implementation}")
        print(f"The default non-sdpa implementation runs in {benchmark_model_forward_pass(model_esm_eager, inputs):.3f} microseconds")

        outputs_eager = model_esm_eager(**inputs)

        # the sdpa way
        print(f"Using implementation of attention {model_esm_sdpa.config._attn_implementation}")

        # Helpful arguments mapper
        backend_map = {
            SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
            SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
            SDPBackend.EFFICIENT_ATTENTION: {
                "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
        }

        for backend, args in backend_map.items():
            with sdp_kernel(**args):
                try:
                    print(f"The {backend.name.lower()} implementation runs in {benchmark_model_forward_pass(model_esm_sdpa, inputs):.3f} microseconds")
                except RuntimeError as e:
                    print(f"{backend.name} is not supported on this device. Reason: {e}")

        outputs_sdpa = model_esm_sdpa(**inputs)

        check_last_hidden_state = torch.allclose(outputs_sdpa.last_hidden_state, outputs_eager.last_hidden_state, atol=1e-5)
        check_pooler_output = torch.allclose(outputs_sdpa.pooler_output, outputs_eager.pooler_output, atol=1e-5)
        print(f"Check last hidden state: {check_last_hidden_state}")
        print(f"Check pooler output: {check_pooler_output}")


if __name__ == "__main__":
    main()

