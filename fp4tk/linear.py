import torch
from collections import OrderedDict

from fp4tk.recipe import RecipeConfig
from fp4tk.matmul import FP4MatMul
from fp4tk.quantizer import QuantizerRegistry


class FP4Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None, recipe: RecipeConfig | None = None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self._setup_quantized_training(recipe)
    
    @classmethod
    def from_linear(cls, linear_layer, recipe: RecipeConfig | None = None):
        """Create a FP4Linear from an existing nn.Linear layer by reusing parameters."""
        # Instantiate without running nn.Linear.__init__ of the class
        fp4_linear = cls.__new__(cls)
        # or super(cls, custom_linear).__init__()
        
        # We only want to run __init__ of Module base class 
        # which nn.Linear is based from
        torch.nn.Module.__init__(fp4_linear)

        fp4_linear.in_features = linear_layer.in_features
        fp4_linear.out_features = linear_layer.out_features

        # Reuse the same parameter tensors (no copying considering large model, just reference)
        # We need to register them as parameters
        fp4_linear.register_parameter('weight', linear_layer.weight)
        fp4_linear.register_parameter('bias', linear_layer.bias)

        fp4_linear._setup_quantized_training(recipe)
        return fp4_linear

    def _setup_quantized_training(self, recipe: RecipeConfig): 
        if recipe is None:
            raise ValueError(f"recipe {RecipeConfig} must be provided for {__class__.__name__}")
        
        self.recipe = recipe
        self.quantizers = OrderedDict()
        for q_name in (
            "fwd_x", "fwd_wt", 
            "bwd_grad_y", "bwd_w", 
            "bwd_grad_yt", "bwd_x"):
            q_config = getattr(recipe, q_name)
            self.quantizers[q_name] = QuantizerRegistry.create(q_config)
        
    def forward(self, input):
        n_dim = len(input.shape)
        if n_dim == 3:
            B, L, E = input.shape
            Y =  FP4MatMul.apply(input.view(-1, E), self.weight, self.bias, list(self.quantizers.values()), self.recipe)
            return Y.view(B, L, -1)
        elif n_dim == 2:
            return FP4MatMul.apply(input, self.weight, self.bias, list(self.quantizers.values()), self.recipe)

        else:
            raise NotImplementedError
        
    def extra_repr(self) -> str:
        repr_str = super().extra_repr()
        repr_str += f"\nFP4 Recipe: {self.recipe.name}"
        for i, (qname, q) in enumerate(self.quantizers.items()):
            repr_str += f"\n  Q{i}( {qname:<12}): {q}"
        return repr_str

        # + f",\n{self.quantizers}"
        # return f"IC={self.in_features}, OC={self.out_features}, bias={self.bias is not None}\n{self.recipe}"

if __name__ == "__main__":
    from recipe import mx_baseline_recipe, tetrajet_recipe, fp4_all_the_way_recipe

    tetrajet_linear = FP4Linear(128, 64, recipe=tetrajet_recipe)
    # habanalab_linear = FP4Linear(128, 64, recipe=fp4_all_the_way_recipe)

    torch_linear = torch.nn.Linear(31, 256)
    
    mx_linear = FP4Linear.from_linear(torch_linear, recipe=mx_baseline_recipe)

    print(tetrajet_linear)
