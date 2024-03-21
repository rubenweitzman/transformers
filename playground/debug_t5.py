from transformers import T5Model, T5Tokenizer
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
    print("Working with model T5")
    device = torch.device("cuda")
    print(f"Running on device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    # Example input
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model_T5_sdpa = T5Model.from_pretrained("google-t5/t5-small").to(device).half()
    model_T5_eager = T5Model.from_pretrained("google-t5/t5-small").to(device).half()

    model_T5_eager.config._attn_implementation = "eager"

    # Prepare inputs
    encoder_inputs_list = ["Translate English to German: The house is wonderful.", "Translate English to German: The car is fast."]
    decoder_inputs_list = ["Das Haus ist wunderbar.", "Das Auto ist schnell."]

    encoder_inputs = tokenizer(encoder_inputs_list, return_tensors="pt")
    decoder_inputs = tokenizer(decoder_inputs_list, return_tensors="pt")

    inputs = {
        "input_ids": encoder_inputs["input_ids"].to(device),
        "attention_mask": encoder_inputs["attention_mask"].to(device),
        "decoder_input_ids": decoder_inputs["input_ids"].to(device),
        "decoder_attention_mask": decoder_inputs["attention_mask"].to(device),
    }

    # inputs = tokenizer("ACDEF", return_tensors="pt").to(device)
    # inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # the eager way
        print(f"Using implementation of attention {model_T5_eager.config._attn_implementation}")
        print(f"The default non-sdpa implementation runs in {benchmark_model_forward_pass(model_T5_eager, inputs):.3f} microseconds")

        outputs_eager = model_T5_eager(**inputs)

        # the sdpa way
        print(f"Using implementation of attention {model_T5_sdpa.config._attn_implementation}")

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
                    print(f"The {backend.name.lower()} implementation runs in {benchmark_model_forward_pass(model_T5_sdpa, inputs):.3f} microseconds")
                except RuntimeError as e:
                    print(f"{backend.name} is not supported on this device. Reason: {e}")

        outputs_sdpa = model_T5_sdpa(**inputs)

        check_last_hidden_state = torch.allclose(outputs_sdpa.last_hidden_state, outputs_eager.last_hidden_state, atol=1e-5)
        # check_pooler_output = torch.allclose(outputs_sdpa.pooler_output, outputs_eager.pooler_output, atol=1e-5)
        print(f"Check last hidden state: {check_last_hidden_state}")
        # print(f"Check pooler output: {check_pooler_output}")

if __name__ == "__main__":
    main()
