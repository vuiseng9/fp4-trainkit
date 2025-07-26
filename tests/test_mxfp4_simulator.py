import torch
from impl.fp4_quantizer import MXFP4Simulator
from impl.recipe import QuantConfig, QuantFormat, ScaleImpl, BlockAxis, Rounding
import pytest
import numpy as np
 
@pytest.fixture
def mxfp4_simulator():
    # TODO: more variants
    return MXFP4Simulator(QuantConfig(QuantFormat.MXFP4, BlockAxis.RowWise, Rounding.Nearest, ScaleImpl.MX))

@pytest.fixture
def mxfp4_simulator_colwise():
    # TODO: more variants
    return MXFP4Simulator(QuantConfig(QuantFormat.MXFP4, BlockAxis.ColWise, Rounding.Nearest, ScaleImpl.MX))

class TestGetE8M0Scale:
    """Test cases for sim_mxfp4.get_e8m0_scaler function."""

    @pytest.mark.parametrize("datatype", [torch.float64, torch.int64])
    def test_forbidden_datatype(self, datatype, mxfp4_simulator):
        tensor = torch.zeros(3, 32, dtype=datatype)
        with pytest.raises(ValueError):
            _ = mxfp4_simulator.get_e8m0_scale(tensor)

    @pytest.mark.parametrize("datatype", [torch.float32, torch.bfloat16])
    def test_supported_datatype(self, datatype, mxfp4_simulator):
        tensor = torch.zeros(3, 32, dtype=datatype)
        scale = mxfp4_simulator.get_e8m0_scale(tensor)
        assert scale.dtype == datatype

    @pytest.mark.parametrize("shape", [
        (0, ), 
        (1, ), 
        (1, 5), 
        (2, 3, 6), 
        (1, 1, 2, 2),
    ])
    def test_forbidden_shape(self, shape, mxfp4_simulator):
        tensor = torch.rand(*shape)
        with pytest.raises(ValueError):
            _ = mxfp4_simulator.get_e8m0_scale(tensor)

    @pytest.mark.parametrize("shape", [
        (32, ), 
        (2, 32), 
        (5, 32), 
        (2, 1, 32), 
        (1, 1, 4, 32),
    ])
    def test_microscaling_blocked_shape(self, shape, mxfp4_simulator):
        tensor = torch.rand(*shape)
        scale = mxfp4_simulator.get_e8m0_scale(tensor)
        assert scale.shape == tensor.shape[:-1] + (1,)

    def test_out_of_bound_amax(self):
        # FP32 and BF16 has 8 bits for exponent, so amax will not exceed E8M0 
        pass

    # TODO bfloat16
    def test_functional(self, mxfp4_simulator):

        list_of_max_values = [-2**-127, 2**-126+2**-129, 0.00456, 3.142, 123456789, 2**127]
        tensor = torch.tensor(list_of_max_values, dtype=torch.float32).unsqueeze(-1).tile(32)
        
        expected = 2**np.clip(
            np.floor(np.log2(np.absolute(np.array(list_of_max_values, dtype=np.float32))/4.0)),
            -127, 127)

        scale = mxfp4_simulator.get_e8m0_scale(tensor)

        assert torch.allclose(scale.squeeze().T, torch.tensor(expected, dtype=torch.float32))


class TestMXFP4Simulator:
    @pytest.mark.parametrize("datatype", [torch.float64, torch.int64])
    def test_forbidden_datatype(self, datatype, mxfp4_simulator):
        tensor = torch.zeros(3, 32, dtype=datatype)
        with pytest.raises(ValueError):
            _ = mxfp4_simulator(tensor)

    @pytest.mark.parametrize("datatype", [torch.float32, torch.bfloat16])
    def test_supported_datatype(self, datatype, mxfp4_simulator_colwise):
        tensor = torch.rand(128, 6, dtype=datatype)
        emulated_tensor = mxfp4_simulator_colwise(tensor)
        assert emulated_tensor.dtype == datatype
    
    @pytest.mark.parametrize(
            ("input_shape", "blocking_dim"),
            [
                ((3, 127), BlockAxis.RowWise),
                ((1, 1024), BlockAxis.ColWise),
                ((64, 1), BlockAxis.RowWise),
                ((31, 64), BlockAxis.ColWise),
            ]
    )
    def test_forbidden_shape_blocking_dim(self, input_shape, blocking_dim):
        sim_mxfp4 = MXFP4Simulator(QuantConfig(QuantFormat.MXFP4, blocking_dim, Rounding.Nearest, ScaleImpl.MX))
        tensor = torch.rand(*input_shape)
        with pytest.raises(ValueError):
            _ = sim_mxfp4(tensor)

    @pytest.mark.parametrize(
            ("input_shape", "blocking_dim"),
            [
                ((3, 128), BlockAxis.RowWise),
                ((1, 1024), BlockAxis.RowWise),
                ((64, 1), BlockAxis.ColWise),
                ((32, 64), BlockAxis.ColWise),
            ]
    )
    def test_output_shape(self, input_shape, blocking_dim):
        sim_mxfp4 = MXFP4Simulator(QuantConfig(QuantFormat.MXFP4, blocking_dim, Rounding.Nearest, ScaleImpl.MX))
        tensor = torch.rand(*input_shape)
        emulated_tensor = sim_mxfp4(tensor)
        assert emulated_tensor.shape == input_shape

    @pytest.mark.parametrize("datatype", [torch.float32, torch.bfloat16])
    def test_subnormal_values(self, datatype, mxfp4_simulator):
        tensor = (torch.rand(40, 1024, dtype=datatype)*2) - 1
        emulated_tensor = mxfp4_simulator(tensor)
        assert torch.allclose(tensor, emulated_tensor, atol=0.25)

    @pytest.mark.parametrize("datatype", [torch.float32, torch.bfloat16])
    def test_normal_values(self, datatype, mxfp4_simulator):
        tensor = torch.empty(40, 1024, dtype=datatype)
        tensor.uniform_(1, 2**10)

        emulated_tensor = mxfp4_simulator(tensor)
        assert torch.allclose(tensor, emulated_tensor, atol=0.25)