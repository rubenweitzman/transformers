from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import matplotlib.pyplot as plt

def generate_batch_inputs(tokenizer, text, correct_text, batch_size, device):
    # Tokenize input text with <mask>
    inputs = tokenizer([text] * batch_size, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Tokenize correct text (with the actual word instead of <mask>)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer([correct_text] * batch_size, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
    
    # Ensure labels match the size of inputs, setting labels for non-mask tokens to -100
    labels = torch.where(inputs["input_ids"] == tokenizer.mask_token_id, labels, torch.tensor(-100).to(device))
    
    inputs["labels"] = labels
    return inputs


def benchmark_model_forward_backward(model, inputs):
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    end_time.record()
    torch.cuda.synchronize()

    model.zero_grad()  # Clear gradients

    execution_time = start_time.elapsed_time(end_time)  # Time in milliseconds
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    torch.cuda.reset_peak_memory_stats()  # Reset memory stats for next measurement

    return execution_time, peak_memory

def plot_results(batch_sizes, execution_times, memory_usages):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, execution_times, '-o')
    plt.title('Execution Time vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Execution Time (ms)')

    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, memory_usages, '-o')
    plt.title('Peak Memory Usage vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Peak Memory (MB)')

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)

    # Original text with a <mask> token and its corrected version
    text = "The capital of France is <mask>."
    correct_text = "The capital of France is Paris."
    batch_sizes = [8, 16, 32, 64]  # Example batch sizes

    execution_times, memory_usages = [], []

    for batch_size in batch_sizes:
        inputs = generate_batch_inputs(tokenizer, text, correct_text, batch_size, device)
        execution_time, peak_memory = benchmark_model_forward_backward(model, inputs)
        execution_times.append(execution_time)
        memory_usages.append(peak_memory)
        print(f"Batch size: {batch_size}, Execution time: {execution_time} ms, Peak memory: {peak_memory} MB")

    plot_results(batch_sizes, execution_times, memory_usages)
    print("Benchmarking completed. Results saved to 'benchmark_results.png'.")

if __name__ == "__main__":
    main()
