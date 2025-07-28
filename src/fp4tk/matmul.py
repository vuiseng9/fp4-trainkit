import torch
from fp4tk.recipe import RecipeConfig
from collections import OrderedDict

class FP4MatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, Quant: OrderedDict , recipe: RecipeConfig):
        # using X, W instead of generic A & B for easy correspondence to Linear layer
        # assume W following layout of nn.Linear, i.e. OCxIC

        if recipe.quant_fwd_matmul:
            fq_X  = Quant[0](X)
            fq_Wt = Quant[1](W.T)
            Y = torch.matmul(fq_X, fq_Wt)
        else:
            Y = torch.matmul(X, W.T)

        if recipe.double_quantization:
            ctx.save_for_backward(X, fq_Wt.T) # fq_Wt has to be transposed back
        else:
            ctx.save_for_backward(X, W)

        ctx.Quant = Quant
        ctx.recipe = recipe
        ctx.has_bias = b is not None

        return Y
    
    @staticmethod
    def backward(ctx, grad_Y):
        X, W = ctx.saved_tensors

        grad_X = grad_W = grad_b = None

        if ctx.needs_input_grad[0] is True:
            if ctx.recipe.quant_bwd_grad_x_matmul:
                grad_X = torch.matmul(
                            ctx.Quant[2](grad_Y), 
                            ctx.Quant[3](W)
                            )
            else:
                grad_X = torch.matmul(grad_Y, W)
        
        if ctx.needs_input_grad[1] is True:
            if ctx.recipe.quant_bwd_grad_w_matmul:
                grad_W = torch.matmul(
                            ctx.Quant[4](grad_Y.view(-1, grad_Y.shape[-1]).T), 
                            ctx.Quant[5](X.view(-1, X.shape[-1]))
                            )
            else:
                grad_W = torch.matmul(
                            grad_Y.view(-1, grad_Y.shape[-1]).T, 
                            X.view(-1, X.shape[-1])
                            )
                
        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0)

        return grad_X, grad_W, grad_b, None, None