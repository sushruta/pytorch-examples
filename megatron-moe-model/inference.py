"""
Inference script for MoE model.
Supports text generation and evaluation.
"""

import argparse
import time
import yaml
import torch
from typing import Dict, Any, List

from models import MyModel, ModelConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config_path: str, checkpoint_path: str = None, device: str = "cuda") -> MyModel:
    """
    Load model from configuration and optional checkpoint.

    Args:
        config_path: Path to model configuration YAML
        checkpoint_path: Optional path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    # Load config
    config = load_config(config_path)
    model_config = ModelConfig(**config['model'])

    # Create model
    print(f"Creating model: {model_config.name}")
    model = MyModel(model_config)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"Model loaded with {model.get_num_params() / 1e9:.2f}B parameters")

    return model


def tokenize_text(text: str, vocab_size: int = 32000) -> torch.Tensor:
    """
    Simple character-level tokenization for demonstration.
    Replace with proper tokenizer (e.g., SentencePiece, tiktoken).

    Args:
        text: Input text
        vocab_size: Vocabulary size

    Returns:
        Token IDs tensor
    """
    # For demo: use character codes modulo vocab_size
    token_ids = [ord(c) % vocab_size for c in text]
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


def detokenize_ids(token_ids: torch.Tensor) -> str:
    """
    Simple character-level detokenization for demonstration.
    Replace with proper detokenizer.

    Args:
        token_ids: Token IDs tensor [batch_size, seq_len]

    Returns:
        Decoded text
    """
    # For demo: convert back to characters
    ids = token_ids[0].tolist()
    text = ''.join([chr(min(max(0, id), 127)) for id in ids])
    return text


def generate_text(
    model: MyModel,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: The MoE model
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device to run on

    Returns:
        Generated text
    """
    # Tokenize prompt
    input_ids = tokenize_text(prompt, vocab_size=model.vocab_size).to(device)

    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...")

    # Generate
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    elapsed = time.time() - start_time
    tokens_per_sec = max_new_tokens / elapsed

    # Detokenize
    generated_text = detokenize_ids(output_ids)

    print(f"\nGenerated text:\n{generated_text}")
    print(f"\nGeneration speed: {tokens_per_sec:.2f} tokens/sec")

    return generated_text


def evaluate_perplexity(
    model: MyModel,
    texts: List[str],
    device: str = "cuda",
) -> float:
    """
    Evaluate perplexity on a list of texts.

    Args:
        model: The MoE model
        texts: List of text samples
        device: Device to run on

    Returns:
        Average perplexity
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    print(f"Evaluating perplexity on {len(texts)} samples...")

    with torch.no_grad():
        for text in texts:
            # Tokenize
            input_ids = tokenize_text(text, vocab_size=model.vocab_size).to(device)

            # Forward pass
            logits, loss, _ = model(input_ids, labels=input_ids)

            if loss is not None:
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)

    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"Perplexity: {perplexity:.2f}")

    return perplexity


def benchmark_throughput(
    model: MyModel,
    batch_size: int = 1,
    seq_len: int = 512,
    num_iterations: int = 100,
    device: str = "cuda",
):
    """
    Benchmark model throughput.

    Args:
        model: The MoE model
        batch_size: Batch size for benchmarking
        seq_len: Sequence length
        num_iterations: Number of iterations to run
        device: Device to run on
    """
    model.eval()

    # Warmup
    print("Warming up...")
    dummy_input = torch.randint(0, model.vocab_size, (batch_size, seq_len)).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Benchmark
    print(f"Benchmarking with batch_size={batch_size}, seq_len={seq_len}...")

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # Compute metrics
    total_tokens = batch_size * seq_len * num_iterations
    tokens_per_sec = total_tokens / elapsed
    latency_ms = (elapsed / num_iterations) * 1000

    print(f"\nBenchmark results:")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    print(f"  Latency: {latency_ms:.2f} ms/batch")
    print(f"  Time per iteration: {elapsed / num_iterations * 1000:.2f} ms")

    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Peak memory: {memory_allocated:.2f} GB")


def analyze_expert_usage(
    model: MyModel,
    input_ids: torch.Tensor,
    device: str = "cuda",
):
    """
    Analyze which experts are being used for a given input.
    Requires model modification to return expert indices.
    """
    # This is a placeholder for expert analysis
    # You would need to modify the model to track expert usage
    print("Expert usage analysis not yet implemented.")
    print("To implement: modify MoELayer to return selected expert indices")


def main():
    parser = argparse.ArgumentParser(description="Inference with MoE model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="generate",
                       choices=["generate", "evaluate", "benchmark"],
                       help="Inference mode")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for benchmarking")
    parser.add_argument("--seq_len", type=int, default=512,
                       help="Sequence length for benchmarking")

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model
    model = load_model(args.config, args.checkpoint, args.device)

    # Run inference based on mode
    if args.mode == "generate":
        generate_text(
            model=model,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )

    elif args.mode == "evaluate":
        # Example evaluation texts
        eval_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Deep learning models require large amounts of data.",
        ]
        evaluate_perplexity(model, eval_texts, args.device)

    elif args.mode == "benchmark":
        benchmark_throughput(
            model=model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_iterations=100,
            device=args.device,
        )


if __name__ == "__main__":
    main()
