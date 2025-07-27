from dataclasses import dataclass, fields
from enum import Enum, IntEnum, StrEnum, auto
import warnings


class QuantFormat(StrEnum):
    MXFP4 = auto()
    NVFP4 = auto()

class BlockSize(IntEnum):
    NV = 16
    MX = 32

# Unused for now
class FP(StrEnum):
    E2M1 = "fp4_e2m1"
    E2M3 = "fp6_e2m3"
    E3M2 = "fp6_e3m2"
    E4M3 = "fp8_e4m3"
    E5M2 = "fp8_e5m2"
    E8M0 = "fp8_e8m0"

class ScaleImpl(StrEnum):
    MX = auto()
    TetraJet = auto()
    FP4_All_The_Way = auto()
    Nvidia_To_Infinity = auto()

class Rounding(StrEnum):
    Nearest = auto()
    Stochastic = auto()

class BlockAxis(Enum):
    ColWise = 0 
    RowWise = 1

@dataclass
class QuantConfig:
    quant_format: QuantFormat
    block_axis: BlockAxis
    element_rounding: Rounding
    scale_impl: ScaleImpl

    def __repr__(self) -> str:
        return (
            f"{__class__.__name__}("
            f"{self.quant_format.name}, "
            f"{self.block_axis.name}, "
            f"{self.element_rounding.name} Round, "
            f"{self.scale_impl.name} Scale Impl.)")


@dataclass
class RecipeConfig:
    name: str
    quant_format: QuantFormat

    # Y = X @ Wt  
    quant_fwd_matmul: bool = False 
    fwd_x: QuantConfig | None = None 
    fwd_wt: QuantConfig | None = None
    
    # dL/dx = dL/dy @ dy/dx
    # dL/dx = dL/dy @ w
    quant_bwd_grad_x_matmul: bool = False
    bwd_grad_y: QuantConfig | None = None
    bwd_w: QuantConfig | None = None
    
    # dL/dw = dL/dy.t() @ dy/dw
    # dL/dw = dL/dy.t() @ x
    quant_bwd_grad_w_matmul: bool = False
    bwd_grad_yt: QuantConfig | None = None
    bwd_x: QuantConfig | None = None

    # x and w blocking axis differs between fwd and bwd
    # True: during bwd, x and w is requantized with quantized ones in fwd path
    # False: source x and w is quantized independently during fwd and bwd path
    double_quantization: bool = False

    def __post_init__(self):
        pass

    def __repr__(self) -> str:
        lines = [f"{__class__.__name__}("]
        # one entry per field
        for i, f in enumerate(fields(self)):
            value = getattr(self, f.name)
            lines.append(f"\t{f.name:<25} = {value!r}")
        lines.append(")")
        return "\n".join(lines)
    
    def freeze(self):
        # TODO: make this immutable & unfreeze perhaps
        raise NotImplementedError

    def validate_quant_config(self):
        def check_set(boolname, q1_name, q2_name):
            if getattr(self, boolname):
                for attr in [q1_name, q2_name]:
                    if getattr(self, attr) is None:
                        raise ValueError(f"{attr} cannot be None when {boolname} is True")
            else:
                for attr in [q1_name, q2_name]:
                    if getattr(self, attr) is not None:
                        warnings.warn(f"{boolname} is None, {attr} is not None, reseting", UserWarning, stacklevel=3)
                        setattr(self, attr, None)

        # if double_quantization is true,
        # "quant_fwd_matmul", "quant_bwd_grad_x_matmul" "quant_bwd_grad_w_matmul" must be all True
        # raise error if any of fwd_x, fwd_wt, bwd_w, bwd_x is None
        if self.double_quantization:
            if not (self.quant_fwd_matmul and self.quant_bwd_grad_x_matmul and self.quant_bwd_grad_w_matmul):
                raise ValueError("All of 'quant_fwd_matmul', 'quant_bwd_grad_x_matmul', and 'quant_bwd_grad_w_matmul' must be True when double_quantization is True")

            for qconfig in ["fwd_x", "fwd_wt", "bwd_w", "bwd_x"]:
                if getattr(self, qconfig) is None:
                    raise ValueError(f"{qconfig} cannot be None when double_quantization is True")
                
        check_set("quant_fwd_matmul", "fwd_x", "fwd_wt")
        check_set("quant_bwd_grad_x_matmul", "bwd_grad_y", "bwd_w")
        check_set("quant_bwd_grad_w_matmul", "bwd_grad_yt", "bwd_x")

        # check quant_format of for each QuantConfig equals to recipe quant_format
        for qconfig in ["fwd_x", "fwd_wt", "bwd_grad_y", "bwd_w", "bwd_grad_yt", "bwd_x"]:
            if getattr(self, qconfig) and getattr(self, qconfig).quant_format != self.quant_format:
                raise ValueError(f"QuantConfig of {qconfig} uses {getattr(self, qconfig).quant_format}, does not match Recipe's {self.quant_format}")
    

def create_mx_baseline_recipe():
    mxfp4 = QuantFormat.MXFP4
    rounding = Rounding.Nearest
    scale = ScaleImpl.MX

    mx_recipe = RecipeConfig(
        name = "MXFP4_Baseline", 
        quant_format = mxfp4,
        quant_fwd_matmul = True,
        fwd_x = QuantConfig(mxfp4, BlockAxis.RowWise, rounding, scale),
        fwd_wt = QuantConfig(mxfp4, BlockAxis.ColWise, rounding, scale),
        quant_bwd_grad_x_matmul = True,
        bwd_grad_y = QuantConfig(mxfp4, BlockAxis.RowWise, rounding, scale),
        bwd_w = QuantConfig(mxfp4, BlockAxis.ColWise, rounding, scale),
        quant_bwd_grad_w_matmul= True,
        bwd_grad_yt = QuantConfig(mxfp4, BlockAxis.RowWise, rounding, scale),
        bwd_x = QuantConfig(mxfp4, BlockAxis.ColWise, rounding, scale),
        double_quantization=False
        )

    mx_recipe.validate_quant_config()
    # mx_recipe.freeze()
    return mx_recipe

def create_tetrajet_recipe():
    mxfp4 = QuantFormat.MXFP4
    scale = ScaleImpl.TetraJet

    tetrajet_recipe = RecipeConfig(
        name = "TetraJet", 
        quant_format = mxfp4,
        quant_fwd_matmul = True,
        fwd_x = QuantConfig(mxfp4, BlockAxis.RowWise, Rounding.Nearest, scale),
        fwd_wt = QuantConfig(mxfp4, BlockAxis.ColWise, Rounding.Nearest, scale),
        quant_bwd_grad_x_matmul = True,
        bwd_grad_y = QuantConfig(mxfp4, BlockAxis.RowWise, Rounding.Stochastic, scale),
        bwd_w = QuantConfig(mxfp4, BlockAxis.ColWise, Rounding.Stochastic, scale),
        quant_bwd_grad_w_matmul= True,
        bwd_grad_yt = QuantConfig(mxfp4, BlockAxis.RowWise, Rounding.Stochastic, scale),
        bwd_x = QuantConfig(mxfp4, BlockAxis.ColWise, Rounding.Stochastic, scale),
        double_quantization=True
        )

    tetrajet_recipe.validate_quant_config()
    # tetrajet_recipe.freeze()
    return tetrajet_recipe

def create_nvidia_round_to_infinity_recipe():
    mxfp4 = QuantFormat.MXFP4
    rounding = Rounding.Nearest
    scale = ScaleImpl.Nvidia_To_Infinity

    nvidia_recipe = RecipeConfig(
        name = "Nvidia_Round_to_Infinity", 
        quant_format = mxfp4,
        quant_fwd_matmul = True,
        fwd_x = QuantConfig(mxfp4, BlockAxis.RowWise, rounding, scale),
        fwd_wt = QuantConfig(mxfp4, BlockAxis.ColWise, rounding, scale),
        quant_bwd_grad_x_matmul = True,
        bwd_grad_y = QuantConfig(mxfp4, BlockAxis.RowWise, rounding, scale),
        bwd_w = QuantConfig(mxfp4, BlockAxis.ColWise, rounding, scale),
        quant_bwd_grad_w_matmul= True,
        bwd_grad_yt = QuantConfig(mxfp4, BlockAxis.RowWise, rounding, scale),
        bwd_x = QuantConfig(mxfp4, BlockAxis.ColWise, rounding, scale),
        double_quantization=False
        )

    nvidia_recipe.validate_quant_config()
    # mx_recipe.freeze()
    return nvidia_recipe

def create_fp4_all_the_way_recipe():
    nvfp4 = QuantFormat.NVFP4
    habanalab_scale = ScaleImpl.FP4_All_The_Way

    fp4alltheway_recipe = RecipeConfig(
        name = "FP4_All_The_Way", 
        quant_format = nvfp4,
        quant_fwd_matmul = True,
        fwd_x = QuantConfig(nvfp4, BlockAxis.RowWise, Rounding.Nearest, habanalab_scale),
        fwd_wt = QuantConfig(nvfp4, BlockAxis.ColWise, Rounding.Nearest, habanalab_scale),
        quant_bwd_grad_x_matmul = True,
        bwd_grad_y = QuantConfig(nvfp4, BlockAxis.RowWise, Rounding.Stochastic, habanalab_scale),
        bwd_w = QuantConfig(nvfp4, BlockAxis.ColWise, Rounding.Stochastic, habanalab_scale),
        quant_bwd_grad_w_matmul= True,
        bwd_grad_yt = QuantConfig(nvfp4, BlockAxis.RowWise, Rounding.Stochastic, habanalab_scale),
        bwd_x = QuantConfig(nvfp4, BlockAxis.ColWise, Rounding.Stochastic, habanalab_scale),
        double_quantization=False
        )

    fp4alltheway_recipe.validate_quant_config()
    # fp4alltheway_recipe.freeze()
    
    return fp4alltheway_recipe

mx_baseline_recipe = create_mx_baseline_recipe()
tetrajet_recipe = create_tetrajet_recipe()
fp4_all_the_way_recipe = create_fp4_all_the_way_recipe()
nvidia_round_to_infinity_recipe = create_nvidia_round_to_infinity_recipe()

if __name__ == "__main__":
    mx_baseline = create_mx_baseline_recipe()
    tetrajet = create_tetrajet_recipe()
    fp4_all_the_way = create_fp4_all_the_way_recipe()
    nvidia_recipe = create_nvidia_round_to_infinity_recipe()
    print("end.")