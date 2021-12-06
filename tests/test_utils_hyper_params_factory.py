import pytest
from pyriemann_qiskit.utils.hyper_params_factory import (gen_zz_feature_map,
                                                         gen_two_local, gates,
                                                         get_spsa)


class HyperParamsTestCase:
    def setup_method(self, test_method):
        self.n_features = 2
        self.reps = 2
        self.invalid = 'invalid'

class TestZZFeatureMap(HyperParamsTestCase):
    @pytest.mark.parametrize(
        'entanglement', ['full', 'linear', 'circular', 'sca']
    )
    def test_gen_zz_feature_map_entangl_strings(self, entanglement):
        """Test gen_zz_feature_map with different
           string options of entanglement"""
        handle = gen_zz_feature_map(entanglement=entanglement)
        feature_map = handle(self.n_features)
        assert isinstance(feature_map.parameters, set)

    def test_gen_zz_feature_map_entangl_idx(self,
                                            get_pauli_z_linear_entangl_idx):
        """Test gen_zz_feature_map with valid indices value"""
        indices = get_pauli_z_linear_entangl_idx(self.reps, self.n_features)
        handle = gen_zz_feature_map(reps=self.reps, entanglement=indices)
        feature_map = handle(self.n_features)
        assert isinstance(feature_map.parameters, set)

    def test_gen_zz_feature_map_entangl_hdl(self,
                                            get_pauli_z_linear_entangl_handle):
        """Test gen_zz_feature_map with a valid callable"""
        indices = get_pauli_z_linear_entangl_handle(self.n_features)
        feature_map = gen_zz_feature_map(entanglement=indices)(self.n_features)
        assert isinstance(feature_map.parameters, set)

    def test_gen_zz_feature_map_entangl_invalid_value(self):
        """Test gen_zz_feature_map with uncorrect value"""
        handle = gen_zz_feature_map(entanglement=self.invalid)
        feature_map = handle(self.n_features)
        with pytest.raises(ValueError):
            feature_map.parameters


class TestTwoLocal(HyperParamsTestCase):
    def test_gen_two_local_default(self):
        """Test default values of gen_zz_feature_map"""
        two_local_handle = gen_two_local()
        two_local = two_local_handle(self.n_features)
        assert two_local._num_qubits == self.n_features
        assert len(two_local._rotation_blocks) == 2
        assert len(two_local._entanglement_blocks) == 1

    @pytest.mark.parametrize('rotation_blocks', gates)
    @pytest.mark.parametrize('entanglement_blocks', gates)
    def test_gen_two_local_strings(self, rotation_blocks, entanglement_blocks):
        """Test gen_two_local with different string options"""
        handle = gen_two_local(rotation_blocks=rotation_blocks,
                               entanglement_blocks=entanglement_blocks)
        two_local = handle(self.n_features)
        assert isinstance(two_local._rotation_blocks, list)
        assert isinstance(two_local._entanglement_blocks, list)

    def test_gen_two_local_list(self):
        """Test gen_two_local with a list as rotation
           and entanglement blocks"""
        rotation_blocks = ['cx', 'cz']
        entanglement_blocks = ['rx', 'rz']
        handle = gen_two_local(rotation_blocks=rotation_blocks,
                               entanglement_blocks=entanglement_blocks)
        two_local = handle(self.n_features)
        assert isinstance(two_local._rotation_blocks, list)
        assert isinstance(two_local._entanglement_blocks, list)

    def test_gen_two_local_invalid_string(self):
        """Test gen_two_local with invalid strings option"""
        with pytest.raises(ValueError):
            gen_two_local(rotation_blocks=self.invalid,
                          entanglement_blocks=self.invalid)

    def test_gen_two_local_invalid_list(self):
        """Test gen_two_local with invalid strings option"""
        rotation_blocks = [self.invalid] * 2
        entanglement_blocks = [self.invalid] * 2
        with pytest.raises(ValueError):
            gen_two_local(rotation_blocks=rotation_blocks,
                          entanglement_blocks=entanglement_blocks)


class TestSPSA(HyperParamsTestCase):
    def test_get_spsa_default(self):
        """Test to create spsa with default parameters"""
        spsa = get_spsa()
        assert spsa._parameters[4] == 4.0
        assert spsa._maxiter == 40
        assert spsa._skip_calibration

    def test_get_spsa_auto_calibration(self):
        """Test to create spsa with all none control parameters"""
        spsa = get_spsa(c=(None, None, None, None, None))
        for i in range(5):
            # Should use qiskit default values
            assert spsa._parameters[i] is not None
        assert not spsa._skip_calibration

    def test_get_spsa_custom(self):
        """Test to create spsa with custom parameters"""
        spsa = get_spsa(max_trials=100, c=(0.0, 1.0, 2.0, 3.0, 4.0))
        for i in range(5):
            assert spsa._parameters[i] == i
        assert spsa._skip_calibration
        assert spsa._maxiter == 100
