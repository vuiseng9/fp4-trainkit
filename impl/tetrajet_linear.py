import torch
from impl.simulate_mxfp4 import MXFP4Simulator, ScalerImpl, Blocking

class TetraJetMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, CastMXFP4):
        # using X, W instead of generic A & B for easy correspondence to Linear layer
        # assume W following layout of nn.Linear, i.e. OCxIC

        fq_X = CastMXFP4(X, Blocking.ROWWISE)
        fq_Wt = CastMXFP4(W.t(), Blocking.COLWISE)

        fq_Y = torch.matmul(fq_X, fq_Wt)

        ctx.save_for_backward(X, W)
        ctx.CastMXFP4 = CastMXFP4
        # we can't save b if it is none 
        # but b is also not needed when it is not None grad_b = 1 *
        ctx.has_bias = b is not None

        return fq_Y
    
    @staticmethod
    def backward(ctx, grad_Y):
        X, W = ctx.saved_tensors

        fq_grad_X = fq_grad_W = grad_b = None

        if ctx.needs_input_grad[0] is True:
            fq_grad_X = torch.matmul(
                            ctx.CastMXFP4(grad_Y, Blocking.ROWWISE),  
                            ctx.CastMXFP4(W, Blocking.COLWISE))
        
        if ctx.needs_input_grad[1] is True:
            fq_grad_W = torch.matmul(
                            ctx.CastMXFP4(grad_Y.view(-1, grad_Y.shape[-1]).t(), Blocking.ROWWISE), 
                            ctx.CastMXFP4(X.view(-1, X.shape[-1]), Blocking.COLWISE))

        if ctx.has_bias and ctx.needs_input_grad[2] is True:
            grad_b = grad_Y.sum(dim=0)

        return fq_grad_X, fq_grad_W, grad_b, None
    

class TetraJetLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.CastMXFP4 = MXFP4Simulator(ScalerImpl.TetraJet)

    def forward(self, input):
        n_dim = len(input.shape)
        if n_dim == 3:
            B, L, E = input.shape
            Y =  TetraJetMatMul.apply(input.view(-1, E), self.weight, self.bias, self.CastMXFP4)
            return Y.view(B, L, -1)

        elif n_dim == 2:
            return TetraJetMatMul.apply(input, self.weight, self.bias, self.CastMXFP4)

        else:
            raise NotImplementedError
        
    def extra_repr(self) -> str:
        b_str = "T" if self.bias is not None else "F"
        return f"IC={self.in_features}, OC={self.out_features}, b={b_str}"