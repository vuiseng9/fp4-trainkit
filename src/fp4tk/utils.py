
import torch.nn as nn
from fp4tk.recipe import RecipeConfig
from fp4tk.linear import FP4Linear

class FP4LinearConverter:
    @staticmethod
    def apply(model: nn.Module, recipe: RecipeConfig, keywords: list[str] = None, verbose: bool = False):
        def replace_linear_with_FP4Linear(model, recipe, keywords):
            """Recursively replace nn.Linear with FP4Linear in the model."""
            for name, module in model.named_children():
                if isinstance(module, nn.Linear) and any(keyword in name for keyword in keywords):
                    if verbose:
                        print(f"[fp4tk]: Replacing {name} with FP4Linear")
                    setattr(model, name, FP4Linear.from_linear(module, recipe))
                else:
                    replace_linear_with_FP4Linear(module, recipe, keywords)

        # Always in-place considering large model.
        replace_linear_with_FP4Linear(model, recipe, keywords or [])