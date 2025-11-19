"""
Simple test script to verify model implementation.
Tests model creation, forward pass, and basic functionality.
"""

import torch
from models import MyModel, ModelConfig


def test_model_creation():
    """Test that model can be created successfully."""
    print("Testing model creation...")

    config = ModelConfig(
        vocab_size=1000,
        hidden_size=512,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=1376,
        num_experts=8,
        num_experts_per_tok=2,
        moe_layer_indices=[1, 3],
        max_position_embeddings=512,
    )

    model = MyModel(config)
    print(f"✓ Model created with {model.get_num_params() / 1e6:.2f}M parameters")

    return model, config


def test_forward_pass(model, config):
    """Test forward pass."""
    print("\nTesting forward pass...")

    batch_size = 2
    seq_len = 64

    # Create dummy input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, loss, aux_loss = model(input_ids, labels=input_ids)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    if aux_loss is not None:
        print(f"  Aux loss: {aux_loss.item():.4f}")


def test_generation(model, config):
    """Test text generation."""
    print("\nTesting generation...")

    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    print(f"  Input length: {input_ids.shape[1]}")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            temperature=1.0,
            top_k=50,
        )

    print(f"✓ Generation successful")
    print(f"  Output length: {output_ids.shape[1]}")
    print(f"  Generated {output_ids.shape[1] - input_ids.shape[1]} new tokens")


def test_moe_layers(model):
    """Test that MoE layers are correctly identified."""
    print("\nTesting MoE layer identification...")

    moe_count = 0
    dense_count = 0

    for i, layer in enumerate(model.layers):
        if layer.is_moe_layer:
            moe_count += 1
            print(f"  Layer {i}: MoE")
        else:
            dense_count += 1
            print(f"  Layer {i}: Dense")

    print(f"✓ Found {moe_count} MoE layers and {dense_count} dense layers")


def test_gpu_support():
    """Test GPU support if available."""
    print("\nTesting GPU support...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

        # Test model on GPU
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=688,
            num_experts=4,
            num_experts_per_tok=2,
            moe_layer_indices=[1],
        )

        model = MyModel(config).to(device)
        input_ids = torch.randint(0, config.vocab_size, (1, 32)).to(device)

        with torch.no_grad():
            logits, _, _ = model(input_ids)

        print(f"✓ GPU forward pass successful")
        print(f"  Peak memory: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

    else:
        print("⚠ CUDA not available")
        print("  Note: Model will run on CPU (slower but functional)")
        print("  For H100 GPU training, run on a GPU-enabled machine")


def test_attention_mechanisms():
    """Test different attention configurations."""
    print("\nTesting attention mechanisms...")

    # Test MHA (Multi-Head Attention)
    print("  Testing MHA...")
    config_mha = ModelConfig(
        vocab_size=1000,
        hidden_size=512,
        num_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,  # Same as num_attention_heads
        intermediate_size=1376,
    )
    model_mha = MyModel(config_mha)
    print("  ✓ MHA model created")

    # Test GQA (Grouped Query Attention)
    print("  Testing GQA...")
    config_gqa = ModelConfig(
        vocab_size=1000,
        hidden_size=512,
        num_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,  # Fewer than num_attention_heads
        intermediate_size=1376,
    )
    model_gqa = MyModel(config_gqa)
    print("  ✓ GQA model created")

    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 32))

    with torch.no_grad():
        _, _, _ = model_mha(input_ids)
        _, _, _ = model_gqa(input_ids)

    print("✓ Both MHA and GQA work correctly")


def test_various_configs():
    """Test various model configurations."""
    print("\nTesting various configurations...")

    configs = [
        ("Tiny", 128, 2, 2, 2),
        ("Small", 512, 4, 8, 4),
        ("Medium", 1024, 8, 16, 8),
    ]

    for name, hidden, layers, heads, experts in configs:
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=hidden,
            num_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            intermediate_size=int(hidden * 2.688),
            num_experts=experts,
            moe_layer_indices=list(range(1, layers, 2)),
        )

        model = MyModel(config)
        params = model.get_num_params() / 1e6

        print(f"  {name}: {params:.1f}M parameters ✓")


def main():
    """Run all tests."""
    print("=" * 50)
    print("MoE Model Test Suite")
    print("=" * 50)

    try:
        # Basic tests
        model, config = test_model_creation()
        test_forward_pass(model, config)
        test_generation(model, config)
        test_moe_layers(model)

        # Advanced tests
        test_attention_mechanisms()
        test_various_configs()
        test_gpu_support()

        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
