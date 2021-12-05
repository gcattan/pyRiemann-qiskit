import pytest
import numpy as np
from pyriemann.classification import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann_qiskit.classification import QuanticSVM, QuanticVQC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from qiskit.providers.ibmq.api.exceptions import RequestsApiError

rclf = [QuanticVQC, QuanticSVM]


@pytest.mark.parametrize("classif", rclf)
@pytest.mark.parametrize("quantum", [True, False])
class QuantumClassifierTestCase:
    def prepare_data_params(self):
        self.n_samples = 100
        self.class_len = self.n_samples // self.n_classes  # balanced set
        samples, labels = self.prepare_bin_data(self.n_samples, self.n_features)
        return samples, labels

    def prepare_data_quantic_svm(self):
        self.n_samples = 10
        # We need to have different values for first and second classes
        # in our samples or vector machine will not converge
        self.class_len = self.n_samples // self.n_classes  # balanced set
        samples, labels = self.prepare_bin_data(self.n_samples, self.n_features, False)
        return samples, labels

    def prepare_data_quantic_vqc(self):
        # To achieve testing in a reasonnable amount of time,
        # we will lower the size of the feature and the number of trials
        self.n_samples = 4
        samples, labels = self.prepare_bin_data(self.n_samples, self.n_features)
        return samples, labels

    def test_two_classes(self, classif, quantum, prepare_bin_data):
        self.prepare_bin_data = prepare_bin_data
        self.n_channels, self.n_classes = 3, 2
        self.n_features = self.n_channels ** 2

        if quantum:
            self.clf_init_with_quantum_true(classif)
        else:
            self.clf_init_with_quantum_false(classif)
        
        if quantum or (not quantum and classif is QuanticSVM):
            samples, labels = self.prepare_data_params()
            self.clf_params(classif, samples, labels)
            self.clf_split_classes(classif, samples, labels)

        self.clf_fvt(classif, quantum)


class TestClassifier(QuantumClassifierTestCase):
    def clf_params(self, classif, samples, labels):
        clf = make_pipeline(XdawnCovariances(), TangentSpace(),
                            classif())
        skf = StratifiedKFold(n_splits=5)
        cross_val_score(clf, samples, labels, cv=skf, scoring='roc_auc')

    def clf_init_with_quantum_false(self, classif):
        if classif is QuanticVQC:
            with pytest.raises(ValueError):
                classif(quantum=False)
        else:
            # if "classical" computation enable,
            # no provider and backend should be defined
            q = QuanticSVM(quantum=False)
            q._init_quantum()
            assert not q.quantum
            assert not hasattr(q, "backend")
            assert not hasattr(q, "provider")

    def clf_init_with_quantum_true(self, classif):
        """Test init of quantum classifiers"""
        # if "quantum" computation enabled, but no accountToken are provided,
        # then "quantum" simulation will be enabled
        # i.e., no remote quantum provider will be defined
        q = classif(quantum=True)
        q._init_quantum()
        assert q.quantum
        assert hasattr(q, "_backend")
        assert not hasattr(q, "_provider")
        # if "quantum" computation enabled, and accountToken is provided,
        # then real quantum backend is used
        # this should raise a error as uncorrect API Token is passed
        q = classif(quantum=True, q_account_token="Test")
        with pytest.raises(RequestsApiError):
            q._init_quantum()

    def clf_split_classes(self, classif, samples, labels):
        """Test _split_classes method of quantum classifiers"""
        q = classif()

        # As fit method is not called here, classes_ is not set.
        # so we need to provide the classes ourselves.
        q.classes_ = range(0, self.n_classes)

        x_class1, x_class0 = q._split_classes(samples, labels)
        
        assert np.shape(x_class1) == (self.class_len, self.n_features)
        assert np.shape(x_class0) == (self.class_len, self.n_features)

    def clf_fvt(self, classif, quantum): 
        if classif is QuanticSVM:
            samples, labels = self.prepare_data_quantic_svm()
            self.clf_fvt_quantic_svc(classif, samples, labels, quantum)
        elif quantum:
            samples, labels = self.prepare_data_quantic_vqc()
            self.clf_fvt_vqc(classif, samples, labels)

    def clf_fvt_quantic_svc(self, classif, samples, labels, quantum):
        """Perform SVC on a simulated quantum computer.
        This test can also be run on a real computer by providing a qAccountToken
        To do so, you need to use your own token, by registering on:
        https://quantum-computing.ibm.com/
        Note that the "real quantum version" of this test may also take some time.
        """
        # We will use a quantum simulator on the local machine
        q = QuanticSVM(quantum=quantum, verbose=False)

        q.fit(samples, labels)
        prediction = q.predict(samples)
        # In this case, using SVM, predicting accuracy should be 100%
        assert prediction[:self.class_len].all() == q.classes_[0]
        assert prediction[self.class_len:].all() == q.classes_[1]


    def clf_fvt_vqc(self, classif, samples, labels):
        """Perform VQC on a simulated quantum computer"""
        # We will use a quantum simulator on the local machine
        # quantum parameter for VQC is always true
        q = classif(verbose=False)
        
        q.fit(samples, labels)
        prediction = q.predict(samples)
        # Considering the inputs, this probably make no sense to test accuracy.
        # Instead, we could consider this test as a canary test
        assert len(prediction) == len(labels)
