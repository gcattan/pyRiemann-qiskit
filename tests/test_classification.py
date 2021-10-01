import numpy as np
from pyriemann.classification import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann_qiskit.classification import QuanticSVM, QuanticVQC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


def test_GetSetParams(get_covmats, get_labels, run_with_3d_and_2d):
    clf = make_pipeline(XdawnCovariances(), TangentSpace(),
                        QuanticSVM(target=1, quantum=False))
    skf = StratifiedKFold(n_splits=5)
    n_matrices, n_channels, n_classes = 100, 3, 2
    covset_3d = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    def handle(covset, is_3d):
        cross_val_score(clf, covset, labels, cv=skf, scoring='roc_auc')

    run_with_3d_and_2d(covset_3d, handle)


def test_Quantic_init():
    """Test init of quantum classifiers"""
    ta = 1
    # if "classical" computation enable,
    # no provider and backend should be defined
    q = QuanticSVM(target=ta, quantum=False)
    q._init_quantum()
    assert not q.quantum
    assert not hasattr(q, "backend")
    assert not hasattr(q, "provider")
    # if "quantum" computation enabled, but no accountToken are provided,
    # then "quantum" simulation will be enabled
    # i.e., no remote quantum provider will be defined
    q = QuanticSVM(target=ta, quantum=True)
    q._init_quantum()
    assert q.quantum
    assert hasattr(q, "_backend")
    assert not hasattr(q, "_provider")
    # if "quantum" computation enabled, and accountToken is provided,
    # then real quantum backend is used
    # this should raise a error as uncorrect API Token is passed
    try:
        q = QuanticSVM(target=ta, quantum=True, qAccountToken="Test")
        assert False  # Should never reach this line
    except Exception:
        pass


def test_Quantic_splitTargetAndNonTarget(get_covmats, get_labels,
                                         run_with_3d_and_2d):
    """Test _split_target_and_non_target method of quantum classifiers"""
    n_matrices, n_channels, n_classes = 100, 3, 2
    covset_3d = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    q = QuanticSVM(target=1, quantum=False)

    def handle(covset, is_3d):
        xta, xnt = q._split_target_and_non_target(covset, labels)
        # Covariance matrices should be vectorized
        class_len = n_matrices // n_classes  # balanced set
        assert np.shape(xta) == (class_len, n_channels * n_channels)
        assert np.shape(xnt) == (class_len, n_channels * n_channels)

    run_with_3d_and_2d(covset_3d, handle)


def test_Quantic_SelfCalibration(get_covmats, get_labels, run_with_3d_and_2d):
    """Test _self_calibration method of quantum classifiers"""
    n_matrices, n_channels, n_classes = 100, 3, 2
    covset_3d = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)
    test_size = 0.33
    len_test = int(test_size * n_matrices)

    def handle(covset, is_3d):
        q = QuanticSVM(target=1, quantum=False, test_per=test_size)
        q.fit(covset, labels)
        # Just using a little trick as fit and score method are
        # called by self_calibration method

        def fit(X_train, y_train):
            assert len(y_train) == n_matrices - len_test
            # Covariances matrices of fit and score method
            # should always be non-vectorized
            assert X_train.shape == \
                   (n_matrices - len_test, n_channels, n_channels) if is_3d \
                   else (n_matrices - len_test, n_channels * n_channels)

        def score(X_test, y_test):
            assert len(y_test) == len_test
            assert X_test.shape == \
                   (len_test, n_channels, n_channels) if is_3d \
                   else (len_test, n_channels * n_channels)

        q.fit = fit
        q.score = score
        q._self_calibration()

    run_with_3d_and_2d(covset_3d, handle)


def test_Quantic_FVT_Classical(get_labels, run_with_3d_and_2d):
    """ Perform standard SVC test
    (canary test to assess pipeline correctness)
    """
    # When quantum=False, it should use
    # classical SVC implementation from SKlearn
    nt, ta = 0, 1
    q = QuanticSVM(target=ta, quantum=False, verbose=False)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    n_matrices, n_channels, n_classes = 100, 3, 2
    class_len = n_matrices // n_classes  # balanced set
    nt_covset = np.zeros((class_len, n_channels, n_channels))
    ta_covset = np.ones((class_len, n_channels, n_channels))
    covset_3d = np.concatenate((nt_covset, ta_covset), axis=0)
    labels = get_labels(n_matrices, n_classes)

    def handle(covset, is_3d):
        q.fit(covset, labels)
        # This will autodefine testing sets
        prediction = q.predict(covset)
        # In this case, using SVM, predicting accuracy should be 100%
        assert prediction[:class_len].all() == nt
        assert prediction[class_len:].all() == ta

    run_with_3d_and_2d(covset_3d, handle)


def test_QuanticSVM_FVT_SimulatedQuantum(get_labels, run_with_3d_and_2d):
    """Perform SVC on a simulated quantum computer.
    This test can also be run on a real computer by providing a qAccountToken
    To do so, you need to use your own token, by registering on:
    https://quantum-computing.ibm.com/
    Note that the "real quantum version" of this test may also take some time.
    """
    # We will use a quantum simulator on the local machine
    nt, ta = 0, 1
    n_training = 4
    # We are dealing with a small number of trial,
    # therefore we will skip self_calibration as it may happens
    # that self_calibration select only target or non-target trials
    test_input = {"Target": [[ta] * n_training],
                  "NonTarget": [[nt] * n_training]}
    q = QuanticSVM(target=ta, quantum=True,
                   verbose=False, test_input=test_input)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    n_matrices, n_channels, n_classes = 10, 2, 2
    class_len = n_matrices // n_classes  # balanced set
    nt_covset = np.zeros((class_len, n_channels, n_channels))
    ta_covset = np.ones((class_len, n_channels, n_channels))
    covset_3d = np.concatenate((nt_covset, ta_covset), axis=0)
    labels = get_labels(n_matrices, n_classes)

    def handle(covset, is_3d):
        q.fit(covset, labels)
        prediction = q.predict(covset)
        # In this case, using SVM, predicting accuracy should be 100%
        assert prediction[:class_len].all() == nt
        assert prediction[class_len:].all() == ta

    run_with_3d_and_2d(covset_3d, handle)


def test_QuanticVQC_FVT_SimulatedQuantum(get_covmats, get_labels,
                                         run_with_3d_and_2d):
    """Perform VQC on a simulated quantum computer"""
    # We will use a quantum simulator on the local machine
    # quantum parameter for VQC is always true
    nt, ta = 0, 1
    n_training = 4
    # We are dealing with a small number of trial,
    # therefore we will skip self_calibration as it may happens that
    # self_calibration select only target or non-target trials
    test_input = {"Target": [[ta] * n_training],
                  "NonTarget": [[nt] * n_training]}
    q = QuanticVQC(target=1, verbose=False, test_input=test_input)
    # We need to have different values for target and non-target in our covset
    # or vector machine will not converge
    # To achieve testing in a reasonnable amount of time,
    # we will lower the size of the feature and the number of trials
    n_matrices, n_channels, n_classes = 4, 2, 2
    covset_3d = get_covmats(n_matrices, n_channels)
    labels = get_labels(n_matrices, n_classes)

    def handle(covset, is_3d):
        q.fit(covset, labels)
        prediction = q.predict(covset)
        # Considering the inputs, this probably make no sense to test accuracy.
        # Instead, we could consider this test as a canary test
        assert len(prediction) == len(labels)

    run_with_3d_and_2d(covset_3d, handle)
