
import torch
from dataclasses import dataclass
from typing import ClassVar
from enum import Enum, StrEnum, auto

# TODO: how to disable autodiff?
# TODO: EPS?

class ScalerImpl(StrEnum):
    MX = auto()
    TetraJet = auto()
    NV_MXFP8_RECIPE = auto()

class Blocking(Enum):
    COLWISE = 0 
    ROWWISE = 1

@dataclass
class MXFP4Simulator:
    # Class‑level constant (not an __init__ field)
    K: ClassVar[int] = 32 # block size
    E_MIN: ClassVar[int] = 0 # post biasing
    E_MAX: ClassVar[int] = 2 # post biasing
    MAX_POWER_OF_TWO: ClassVar[int] = 2 ** E_MAX # 2^(3-1)
    MIN_VAL: ClassVar[float] = -6.0 # -1 x 2^E_MAX × base2(1.1)
    MAX_VAL: ClassVar[float] =  6.0 # +1 x 2^E_MAX × base2(1.1)
    M_MIN: ClassVar[float] = 0.0
    M_MAX: ClassVar[float] = 1.5

    SCALER_E_MIN: ClassVar[float] = -127 # post biasing
    SCALER_E_MAX: ClassVar[float] =  127 # post biasing

    scaler_impl: ScalerImpl
    
    def __post_init__(self):
        pass

    def __call__(self, tensor, blocking_dim):
        return self.emulate(tensor, blocking_dim)

    def __repr__(self):
        return f"{__class__.__name__}(impl:{self.scaler_impl})"

    def get_e8m0_scaler(self, tensor):
        # calculate E8M0-emulated scaler
        if tensor.shape[-1] != self.K:
            raise ValueError(f"Last dimension of tensor must be equal to block size K={self.K}")

        if tensor.dtype not in [torch.float32, torch.bfloat16]:
            raise ValueError("Tensor must be of type FP32 or BF16")

        amax = tensor.abs().max(-1, keepdim=True).values

        match self.scaler_impl:
            case ScalerImpl.MX:
                # OCP MX baseline
                scaler_exponent = torch.floor( torch.log2(amax/self.MAX_POWER_OF_TWO) )
            
            case ScalerImpl.TetraJet:
                scaler_exponent = torch.ceil( torch.log2( 2*amax/(self.MAX_VAL-self.MIN_VAL) ) ) 

            case ScalerImpl.NV_MXFP8_RECIPE:
                # Mishra's MXFP8 Recipe
                scaler_exponent = torch.ceil( torch.log2(amax/self.MAX_VAL) )

            case _:
                raise NotImplementedError(f"No default impl: {impl}")
        
        scaler_exponent = scaler_exponent.clamp(self.SCALER_E_MIN, self.SCALER_E_MAX)
        scaler = 2**scaler_exponent
        return scaler #power of two value

    def emulate(self, tensor,  blocking_dim):        
        if tensor.dtype not in [torch.float32, torch.bfloat16]:
            raise ValueError("Tensor must be of type FP32 or BF16")

        if len(tensor.shape) != 2:
            raise ValueError(f"Tensor must be 2D for MXFP4 emulation, current shape: {tensor.shape}")
        
        if blocking_dim.value not in [0, 1]:
            raise ValueError(f"channel axis for blocking must be 0 (rows) or 1 (columns), current axis: {blocking_dim}")
        
        if tensor.shape[blocking_dim.value] % self.K != 0:
            raise ValueError(f"{blocking_dim}, Tensor size along axis {blocking_dim.value} must be divisible by block size K={self.K}, current shape: {tensor.shape}")
        
        # standardize to block by the last axis
        if blocking_dim == Blocking.COLWISE:
            tensor = tensor.t()        

        # reshape tensor to (r, c/K, K) for blocking
        r, c = tensor.shape
        tensor = tensor.reshape(r, -1, self.K)
        scaler = self.get_e8m0_scaler(tensor)
        
        # scaled tensor 
        scaled_tensor = tensor / scaler
        # (-1)^s
        sign = scaled_tensor.sign()
        # remove sign from scaled_tensor 
        scaled_tensor = scaled_tensor.abs()

        # find E2 (note we that we avoid redundant bias by baking in bias beforehand)
        e = torch.floor(torch.log2(scaled_tensor)).clamp(self.SCALER_E_MIN, self.SCALER_E_MAX) 
        # find M1
        m = (scaled_tensor / (2**e)).clamp(self.M_MIN, self.M_MAX) # mantissa can only be 0, 0.5, 1.0, 1.5 for FP4-E2M1

        # because 1 bit fractional, rotate right by multiplying 2^1, round to nearest integer, then rotate left by dividing 2^1, we get final significand
        m_quantized = (m*2).round()/2

        # Reconstruct value in input datatype
        # FP4-E2M1 = (-1)^sign * 2**e * m_quantized
        emulated_tensor = scaler * (sign * 2**e * m_quantized)
        emulated_tensor = emulated_tensor.reshape(r, -1) # relayout as original

        if blocking_dim == Blocking.COLWISE:
            emulated_tensor = emulated_tensor.t()

        return emulated_tensor    


if __name__ == "__main__":
    # Example usage
    mxfp4 = MXFP4Simulator(ScalerImpl.MX)
    ttj_mxfp4 = MXFP4Simulator(ScalerImpl.TetraJet)
    tensor = torch.ones(4, 64)*33
    res_mx = mxfp4(tensor, Blocking.ROWWISE)
    res_ttj = ttj_mxfp4(tensor, Blocking.ROWWISE)
    print("end.")