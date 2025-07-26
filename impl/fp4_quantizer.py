import torch
from dataclasses import dataclass
from typing import ClassVar
from impl.recipe import QuantFormat, ScaleImpl, BlockAxis, QuantConfig

# TODO: how to disable autodiff? do we need?
# TODO: EPS?

# quantizer_factory.py
from typing import Type

class QuantizerRegistry:
    _registry: dict[str, Type] = {}

    @classmethod
    def register(cls, key: str):
        def decorator(qcls: Type):
            cls._registry[key] = qcls
            return qcls
        return decorator

    @classmethod
    def create(cls, qcfg):
        """
        Look up qformat in the registry
        and instantiate with cfg.
        """
        try:
            QConstructor = cls._registry[qcfg.quant_format.value]
        except KeyError:
            raise ValueError(f"Unknown quantizer type {qcfg.quant_format.value!r}")
        return QConstructor(qcfg)


@QuantizerRegistry.register(QuantFormat.MXFP4.value)
@dataclass
class MXFP4Simulator:
    # Class‑level constant (not an __init__ field)
    K: ClassVar[int] = 32 # block size
    E_MIN: ClassVar[int] = 0 # post exponent biasing
    E_MAX: ClassVar[int] = 2 # post exponent biasing
    MAX_POWER_OF_TWO: ClassVar[int] = 2 ** E_MAX # 2^(3-1)
    MIN_VAL: ClassVar[float] = -6.0 # -1 x 2^E_MAX × base2(1.1)
    MAX_VAL: ClassVar[float] =  6.0 # +1 x 2^E_MAX × base2(1.1)
    M_MIN: ClassVar[float] = 0.0
    M_MAX: ClassVar[float] = 1.5

    SCALE_E_MIN: ClassVar[float] = -127 # post exponent biasing
    SCALE_E_MAX: ClassVar[float] =  127 # post exponent biasing

    quant_config: QuantConfig
    
    def __post_init__(self):
        self.block_axis = self.quant_config.block_axis
        self.scale_impl = self.quant_config.scale_impl
        self.rounding = self.quant_config.element_rounding

    def __call__(self, tensor):
        return self.emulate(tensor)

    def __repr__(self):
        return f"{__class__.__name__}(scale impl.: {self.scale_impl}, {self.block_axis.name} blocking, {self.rounding.name} rounding)"

    def get_e8m0_scale(self, tensor):
        # calculate E8M0-emulated scale
        if tensor.shape[-1] != self.K:
            raise ValueError(f"Last dimension of tensor must be equal to block size K={self.K}")

        if tensor.dtype not in [torch.float32, torch.bfloat16]:
            raise ValueError("Tensor must be of type FP32 or BF16")

        absmax = tensor.abs().max(-1, keepdim=True).values

        match self.scale_impl:
            case ScaleImpl.MX:
                # OCP MX baseline
                scale_exponent = torch.floor_( 
                                    torch.log2_( 
                                        absmax.div_(self.MAX_POWER_OF_TWO)))
            
            case ScaleImpl.TetraJet:
                scale_exponent = torch.ceil_(
                                    torch.log2_( 
                                        absmax.mul_(2).div_(self.MAX_VAL-self.MIN_VAL)))

            case ScaleImpl.FP4_All_The_Way:
                raise NotImplementedError
            
            # case ScaleImpl.Mishraetal:
            #     # Mishra's MXFP8 Recipe
            #     scale_exponent = torch.ceil_(
            #                         torch.log2_(
            #                             absmax.div_(self.MAX_VAL)))

            case _:
                raise NotImplementedError(f"No default impl: {impl}")
        
        scale_exponent.clamp_(self.SCALE_E_MIN, self.SCALE_E_MAX)
        
        scale = scale_exponent.exp2_()
        return scale #power of two value

    def emulate(self, tensor):        
        if tensor.dtype not in [torch.float32, torch.bfloat16]:
            raise ValueError("Tensor must be of type FP32 or BF16")

        if len(tensor.shape) != 2:
            raise ValueError(f"Tensor must be 2D for MXFP4 emulation, current shape: {tensor.shape}")
        
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
        scale = self.get_e8m0_scale(tensor)
        
        # scaled tensor 
        scaled_tensor = tensor / scale
        # (-1)^s
        sign = scaled_tensor.sign()
        # remove sign from scaled_tensor 
        scaled_tensor = scaled_tensor.abs()

        # find E2 (note we that we avoid redundant bias by baking in bias beforehand)
        e = torch.floor(torch.log2(scaled_tensor)).clamp(self.SCALE_E_MIN, self.SCALE_E_MAX)

        # find M1
        m = (scaled_tensor / (2**e)).clamp(self.M_MIN, self.M_MAX) # mantissa can only be 0, 0.5, 1.0, 1.5 for FP4-E2M1

        # because 1 bit fractional, rotate right by multiplying 2^1, round to nearest integer, then rotate left by dividing 2^1, we get final significand
        m_quantized = (m*2).round()/2

        # Reconstruct value in input datatype
        # FP4-E2M1 = (-1)^sign * 2**e * m_quantized
        emulated_tensor = scale * (sign * 2**e * m_quantized)
        emulated_tensor = emulated_tensor.view(r, -1) # relayout as original

        if self.block_axis == BlockAxis.ColWise:
            emulated_tensor = emulated_tensor.T

        return emulated_tensor    


if __name__ == "__main__":
    from recipe import QuantFormat, Rounding, ScaleImpl
    qcfg = QuantConfig(QuantFormat.MXFP4, BlockAxis.RowWise, Rounding.Nearest, ScaleImpl.MX)
    mxfp4 = MXFP4Simulator(qcfg)
    tensor = torch.ones(4, 64)*33
    res_mx = mxfp4(tensor)
    print("end.")