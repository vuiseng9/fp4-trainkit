import torch
from dataclasses import dataclass
from typing import ClassVar
from fp4tk.recipe import QuantFormat, ScaleImpl, BlockAxis, QuantConfig, Rounding
from fp4tk.quantizer import QuantizerRegistry

#TODO
# two level scaling as described in
# https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
# Current implementation is only single level of scaling, i.e. one E4M3 scale per 16-element block

@QuantizerRegistry.register(QuantFormat.NVFP4.value)
@dataclass
class NVFP4Simulator:
    # Class‑level constant (not an __init__ field)
    K: ClassVar[int] = 16 # block size
    E_MIN: ClassVar[int] = 0 # post exponent biasing
    E_MAX: ClassVar[int] = 2 # post exponent biasing
    MAX_POWER_OF_TWO: ClassVar[int] = 2 ** E_MAX # 2^(3-1)
    MIN_VAL: ClassVar[float] = -6.0 # -1 x 2^E_MAX × base2(1.1)
    MAX_VAL: ClassVar[float] =  6.0 # +1 x 2^E_MAX × base2(1.1)
    M_MIN: ClassVar[float] = 0.0
    M_MAX: ClassVar[float] = 1.5

    SCALE_MIN_VAL: ClassVar[int] = 2**-9
    SCALE_MAX_VAL: ClassVar[int] = 448

    quant_config: QuantConfig
    
    def __post_init__(self):
        self.block_axis = self.quant_config.block_axis
        self.scale_impl = self.quant_config.scale_impl
        self.rounding = self.quant_config.element_rounding

    def __call__(self, tensor):
        return self.emulate(tensor)

    def __repr__(self):
        return f"{__class__.__name__}(scale impl.: {self.scale_impl}, {self.block_axis.name} blocking, {self.rounding.name} rounding)"

    def get_e4m3_scale(self, tensor):
        # calculate E4M3-emulated scale
        if tensor.shape[-1] != self.K:
            raise ValueError(f"Last dimension of tensor must be equal to block size K={self.K}")

        if tensor.dtype not in [torch.float32, torch.bfloat16]:
            raise ValueError("Tensor must be of type FP32 or BF16")

        eps = torch.finfo(tensor.dtype).eps
        # we use clamp_(min=eps) to achive in-place effect, 
        # instead of torch.max( tensor.abs(), eps )

        absmax = tensor.abs().max(-1, keepdim=True).values

        match self.scale_impl:
            case ScaleImpl.FP4_All_The_Way:
                # PyTorch doesn’t support E8M0 (used in MXFP4), but does support E4M3. We use torch casting here.
                # Downcast the scale to float8 (E4M3), then convert it back to the original dtype.
                scale = (absmax/self.MAX_VAL).clamp(self.SCALE_MIN_VAL, self.SCALE_MAX_VAL) 
                scale = scale.to(torch.float8_e4m3fn).to(scale.dtype)

            case ScaleImpl.MX | ScaleImpl.TetraJet | ScaleImpl.Nvidia_To_Infinity:
                raise NotImplementedError(f"{self.scale_impl} scale impl. is intended for MXFP4, not NVFP4. Use MXFP4Simulator instead.")
            
            case _:
                raise NotImplementedError(f"No default impl: {impl}")
        
        return scale
        

    def emulate(self, tensor):        
        if tensor.dtype not in [torch.float32, torch.bfloat16]:
            raise ValueError("Tensor must be of type FP32 or BF16")

        if len(tensor.shape) != 2:
            raise ValueError(f"Tensor must be 2D for NVFP4 emulation, current shape: {tensor.shape}")
        
        if self.block_axis.value not in [0, 1]:
            raise ValueError(f"channel axis for blocking must be 0 (rows) or 1 (columns), current axis: {self.block_axis}")
        
        if tensor.shape[self.block_axis.value] % self.K != 0:
            raise ValueError(f"{self.block_axis}, Tensor size along axis {self.block_axis.value} must be divisible by block size K={self.K}, current shape: {tensor.shape}")
        
        # standardize to block by the last axis
        if self.block_axis == BlockAxis.ColWise:
            tensor = tensor.T        

        # reshape tensor to (r, c/K, K) for blocking
        r, c = tensor.shape
        tensor = tensor.view(r, -1, self.K)
        scale = self.get_e4m3_scale(tensor)
        
        # scaled tensor 
        scaled_tensor = tensor / scale
        # (-1)^s
        sign = scaled_tensor.sign()
        # remove sign from scaled_tensor 
        scaled_tensor = scaled_tensor.abs()

        eps = torch.finfo(scaled_tensor.dtype).eps
        
        # find E2 (note that we avoid redundant bias by baking in the bias)
        e = torch.floor(torch.log2(scaled_tensor.clamp_(min=eps))).clamp(self.E_MIN, self.E_MAX)

        # find M1
        m = (scaled_tensor / (2**e))
        
        match self.rounding:
            case Rounding.Nearest:
                # Do nothing, round to nearest is common final step for all rounding modes,
                # putting round at the end
                pass
            case Rounding.Stochastic:
                noise = torch.rand_like(m) - 0.5
                m += noise

        # because 1 bit fractional, rotate right by multiplying 2^1, round to nearest integer, then rotate left by dividing 2^1, we get final significand
        m_quantized = ( (m*2).round()/2 ).clamp(self.M_MIN, self.M_MAX)
        # mantissa can only be 0, 0.5, 1.0, 1.5 for FP4-E2M1

        # Reconstruct value in input datatype
        # FP4-E2M1 = (-1)^sign * 2**e * m_quantized
        emulated_tensor = scale * (sign * 2**e * m_quantized)
        emulated_tensor = emulated_tensor.view(r, -1) # relayout as original

        if self.block_axis == BlockAxis.ColWise:
            emulated_tensor = emulated_tensor.T

        return emulated_tensor    


if __name__ == "__main__":
    from fp4tk.recipe import QuantFormat, Rounding, ScaleImpl
    qcfg = QuantConfig(QuantFormat.NVFP4, BlockAxis.RowWise, Rounding.Nearest, ScaleImpl.FP4_All_The_Way)
    nvfp4 = NVFP4Simulator(qcfg)
    tensor = torch.ones(4, 64)*25
    res_mx = nvfp4(tensor)
    print("end.")