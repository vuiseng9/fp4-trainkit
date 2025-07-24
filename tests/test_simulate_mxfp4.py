import torch
from impl.simulate_mxfp4 import MXFP4Simulator, ScalerImpl, Blocking
import pytest
import numpy as np
 

# TODO
# test for different implementation
# to verify bfloat16

class TestGetE8M0Scaler:
    """Test cases for sim_mxfp4.get_e8m0_scaler function."""

    @pytest.mark.parametrize("datatype", [torch.float64, torch.int64])
    def test_forbidden_datatype(self, datatype):
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)
        tensor = torch.zeros(3, 32, dtype=datatype)
        with pytest.raises(ValueError):
            _ = sim_mxfp4.get_e8m0_scaler(tensor)

    @pytest.mark.parametrize("datatype", [torch.float32, torch.bfloat16])
    def test_supported_datatype(self, datatype):
        tensor = torch.zeros(3, 32, dtype=datatype)
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)
        scaler = sim_mxfp4.get_e8m0_scaler(tensor)
        assert scaler.dtype == datatype

    @pytest.mark.parametrize("shape", [
        (0, ), 
        (1, ), 
        (1, 5), 
        (2, 3, 6), 
        (1, 1, 2, 2),
    ])
    def test_forbidden_shape(self, shape):
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)
        tensor = torch.rand(*shape)
        with pytest.raises(ValueError):
            scaler = sim_mxfp4.get_e8m0_scaler(tensor)
        
    @pytest.mark.parametrize("shape", [
        (32, ), 
        (2, 32), 
        (5, 32), 
        (2, 1, 32), 
        (1, 1, 4, 32),
    ])
    def test_microscaling_blocked_shape(self, shape):
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)
        tensor = torch.rand(*shape)
        scaler = sim_mxfp4.get_e8m0_scaler(tensor)
        assert scaler.shape == tensor.shape[:-1] + (1,)

    def test_out_of_bound_amax(self):
        # FP32 and BF16 has 8 bits for exponent, so amax will not exceed E8M0 
        pass

    # TODO bfloat16
    def test_functional(self):
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)

        list_of_max_values = [-2**-127, 2**-126+2**-129, 0.00456, 3.142, 123456789, 2**127]
        tensor = torch.tensor(list_of_max_values, dtype=torch.float32).unsqueeze(-1).tile(32)
        
        expected = 2**np.clip(
            np.floor(np.log2(np.absolute(np.array(list_of_max_values, dtype=np.float32))/4.0)),
            -127, 127)

        scaler = sim_mxfp4.get_e8m0_scaler(tensor)

        assert torch.allclose(scaler.squeeze().t(), torch.tensor(expected, dtype=torch.float32))


class TestSimulateMXFP4:
    @pytest.mark.parametrize("datatype", [torch.float64, torch.int64])
    def test_forbidden_datatype(self, datatype):
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)
        tensor = torch.zeros(3, 32, dtype=datatype)
        with pytest.raises(ValueError):
            emulated_tensor = sim_mxfp4(tensor, Blocking.ROWWISE)

    @pytest.mark.parametrize("datatype", [torch.float32, torch.bfloat16])
    def test_supported_datatype(self, datatype):
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)
        tensor = torch.rand(128, 6, dtype=datatype)
        emulated_tensor = sim_mxfp4(tensor, Blocking.COLWISE)
        assert emulated_tensor.dtype == datatype
    
    @pytest.mark.parametrize(
            ("input_shape", "blocking_dim"),
            [
                ((3, 127), Blocking.ROWWISE),
                ((1, 1024), Blocking.COLWISE),
                ((64, 1), Blocking.ROWWISE),
                ((31, 64), Blocking.COLWISE),
            ]
    )
    def test_forbidden_shape_blocking_dim(self, input_shape, blocking_dim):
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)
        tensor = torch.rand(*input_shape)
        with pytest.raises(ValueError):
            _ = sim_mxfp4(tensor, blocking_dim)

    @pytest.mark.parametrize(
            ("input_shape", "blocking_dim"),
            [
                ((3, 128), Blocking.ROWWISE),
                ((1, 1024), Blocking.ROWWISE),
                ((64, 1), Blocking.COLWISE),
                ((32, 64), Blocking.COLWISE),
            ]
    )
    def test_output_shape(self, input_shape, blocking_dim):
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)
        tensor = torch.rand(*input_shape)
        emulated_tensor = sim_mxfp4(tensor, blocking_dim)
        assert emulated_tensor.shape == input_shape

    @pytest.mark.parametrize("datatype", [torch.float32, torch.bfloat16])
    def test_subnormal_values(self, datatype):
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)
        tensor = (torch.rand(40, 1024, dtype=datatype)*2) - 1 
        emulated_tensor = sim_mxfp4(tensor, Blocking.ROWWISE)
        assert torch.allclose(tensor, emulated_tensor, atol=0.25)

    @pytest.mark.parametrize("datatype", [torch.float32, torch.bfloat16])
    def test_normal_values(self, datatype):
        sim_mxfp4 = MXFP4Simulator(ScalerImpl.MX)
        tensor = torch.empty(40, 1024, dtype=datatype)
        tensor.uniform_(1, 2**10)

        emulated_tensor = sim_mxfp4(tensor, Blocking.ROWWISE)
        assert torch.allclose(tensor, emulated_tensor, atol=0.25)