import torch
import torch.nn as nn
from fp4tk.linear import FP4Linear
from fp4tk.recipe import FP4_RECIPES
import pytest

@pytest.fixture
def linear_configs():
    """Test configurations for linear layers"""
    return [
        (32, 64, True),   # with bias
        (128, 32, False),  # without bias
        (512, 2048, True),  # larger dimensions with bias
        (2048, 256, False) # larger dimensions without bias
    ]
@pytest.fixture
def test_inputs():
    """Different input tensor shapes for testing"""
    def _test_inputs(in_features):
        return [
            torch.randn(5, in_features),           # 2D: batch_size x in_features
            torch.randn(3, 7, in_features),        # 3D: batch_size x seq_len x in_features
        ]
    return _test_inputs

class TestTetraJetLinear:
    def test_forward_pass(self, linear_configs, test_inputs):
        for in_features, out_features, has_bias in linear_configs:
            ref_linear = nn.Linear(in_features, out_features, bias=has_bias)

            tetrajet_linear = FP4Linear.from_linear(ref_linear, FP4_RECIPES["tetrajet"])

            # Test with different batch sizes and sequence lengths
            for i, input_tensor in enumerate(test_inputs(in_features)):
                ref_output = ref_linear(input_tensor)
                tetrajet_linear_output = tetrajet_linear(input_tensor)
                
                # Check shapes match
                assert ref_output.shape == tetrajet_linear_output.shape, \
                    f"Output Shape mismatch for config {in_features}x{out_features}, bias={has_bias}, input {i}"  

    def test_backward_pass(self, linear_configs):
        """Test that CustomLinear produces the same gradients as nn.Linear"""
        for in_features, out_features, has_bias in linear_configs:

            ref_linear = nn.Linear(in_features, out_features, bias=has_bias)
            tetrajet_linear = FP4Linear.from_linear(ref_linear, FP4_RECIPES["tetrajet"])

            input_tensor = torch.randn(64, in_features, requires_grad=True)
            input_tensor_copy = input_tensor.clone().detach().requires_grad_(True)

            # Forward pass
            ref_output = ref_linear(input_tensor)
            tetrajet_output = tetrajet_linear(input_tensor_copy)
            
            # Create dummy loss function (sum of outputs)
            ref_loss = ref_output.sum()
            tetrajet_loss = tetrajet_output.sum()
            
            # Backward pass
            ref_loss.backward()
            tetrajet_loss.backward()
            
            # Check shape for now
            # TODO find delta tolerance
            assert input_tensor.grad.shape == input_tensor_copy.grad.shape
            assert ref_linear.weight.grad.shape == tetrajet_linear.weight.grad.shape
            
            if has_bias:
                assert ref_linear.bias.grad.shape == tetrajet_linear.bias.grad.shape