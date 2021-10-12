"""Module for classification function."""
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from qiskit import BasicAer, IBMQ
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.quantum_instance import logger
from qiskit.aqua.algorithms import QSVM, SklearnSVM, VQC
from qiskit.aqua.utils import get_feature_dimension
from qiskit.providers.ibmq import least_busy
from qiskit.aqua.components.optimizers import SPSA
from datetime import datetime
import logging
logger.level = logging.INFO


class QuanticClassifierBase(BaseEstimator, ClassifierMixin):

    """Quantum classification.

    This class implements a SKLearn wrapper around Qiskit library.
    It provides a mean to run classification tasks on a local and
    simulated quantum computer or a remote and real quantum computer.
    Difference between simulated and real quantum computer will be that:

    * there is no noise on a simulated quantum computer
      (so results are better);
    * real quantum computer are quicker than simulator;
    * real quantum computer tasks are assigned to a queue
      before being executed on a back-end.

    WARNING: At the moment this implementation only supports binary
    classification.

    Parameters
    ----------
    quantum : bool (default: True)
        - If true will run on local or remote backend
        (depending on q_account_token value).
        - If false, will perform classical computing instead
    q_account_token : string (default:None)
        If quantum==True and q_account_token provided,
        the classification task will be running on a IBM quantum backend
    verbose : bool (default:True)
        If true will output all intermediate results and logs
    test_input : dict (default: {}})
        Contains vectorized test set for the two classes
    test_per : double (default: 0.33)
        Indicate the percent of the fitting set that should be kept for
        testing. The remaining percent are used for training.
        This is due to the design of quantum classifiers which require
        training, testing and predicting sets.

    Notes
    -----
    .. versionadded:: 0.0.1

    Attributes
    ----------
    classes_ : list
        list of classes.

    See Also
    --------
    QuanticSVM
    QuanticVQC

    References
    ----------
    .. [1] H. Abraham et al., Qiskit:
           An Open-source Framework for Quantum Computing.
           Zenodo, 2019. doi: 10.5281/zenodo.2562110.

    """

    def __init__(self, quantum=True, q_account_token=None, verbose=True,
                 test_input={}, test_per=0.33):
        self.verbose = verbose
        self._log("Initializing Quantum Classifier")
        self.test_input = test_input
        self.test_per = test_per
        self.q_account_token = q_account_token
        self.quantum = quantum
        # protected field for child classes
        self._training_input = {}

    def _init_quantum(self):
        if self.quantum:
            aqua_globals.random_seed = datetime.now().microsecond
            self._log("seed = ", aqua_globals.random_seed)
            if self.q_account_token:
                self._log("Real quantum computation will be performed")
                IBMQ.delete_account()
                IBMQ.save_account(self.q_account_token)
                IBMQ.load_account()
                self._log("Getting provider...")
                self._provider = IBMQ.get_provider(hub='ibm-q')
            else:
                self._log("Quantum simulation will be performed")
                self._backend = BasicAer.get_backend('qasm_simulator')
        else:
            self._log("Classical SVM will be performed")

    def _log(self, *values):
        if self.verbose:
            print("[QClass] ", *values)

    def _split_classes(self, X, y):
        self._log("""[Warning] Splitting first class from second class.
                 Only binary classification is supported.""")
        n_samples, n_features = X.shape
        self._feature_dim = n_samples * n_features
        self._log("Feature dimension = ", self._feature_dim)
        X_class1 = X[y == self.classes_[1]]
        X_class0 = X[y == self.classes_[0]]
        self._new_feature_dim = len(X_class1[0])
        self._log("Feature dimension after vector processing = ",
                  self._new_feature_dim)
        return (X_class1, X_class0)

    def _additional_setup(self):
        self._log("There is no additional setup.")

    def fit(self, X, y):
        """Prepare the training data and the quantum backend

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : ndarray, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : QuanticClassifierBase instance
            The QuanticClassifierBase instance.
        """
        self._init_quantum()

        self._log("Fitting: ", X.shape)
        self._prev_fit_params = {"X": X, "y": y}
        self.classes_ = np.unique(y)
        class1, class0 = self._split_classes(X, y)

        self._training_input["class1"] = class1
        self._training_input["class0"] = class0
        self._log(get_feature_dimension(self._training_input))
        feature_dim = get_feature_dimension(self._training_input)
        self._feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2,
                                         entanglement='linear')
        self._additional_setup()
        if self.quantum:
            if not hasattr(self, "_backend"):
                def filters(device):
                    return (
                      device.configuration().n_qubits >= self._new_feature_dim
                      and not device.configuration().simulator
                      and device.status().operational)
                devices = self._provider.backends(filters=filters)
                try:
                    self._backend = least_busy(devices)
                except Exception:
                    self._log("Devices are all busy. Getting the first one...")
                    self._backend = devices[0]
                self._log("Quantum backend = ", self._backend)
            seed_sim = aqua_globals.random_seed
            seed_trs = aqua_globals.random_seed
            self._quantum_instance = QuantumInstance(self._backend, shots=1024,
                                                     seed_simulator=seed_sim,
                                                     seed_transpiler=seed_trs)
        return self

    def _run(self, predict_set=None):
        raise Exception("Run method was not implemented")

    def _self_calibration(self):
        X = self._prev_fit_params["X"]
        y = self._prev_fit_params["y"]
        self._log("Test size = ", self.test_per, " of previous fitting.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_per, stratify=y)
        self.fit(X_train, y_train)
        self.score(X_test, y_test)

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        if(len(self.test_input) == 0):
            self._log("There is no test inputs. Self-calibrating...")
            self._self_calibration()
        result = None
        self._log("Prediction: ", X.shape)
        result = self._run(X)
        self._log("Prediction finished. Returning predicted labels")
        return result["predicted_labels"]

    def predict_proba(self, X):
        """This method is implemented for compatibility purpose
           as SVM prediction probabilities are not available.
           This method assigns to each trial a boolean which value
           depends on wheter the label was assigned to classes 0 or 1

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        prob : ndarray, shape (n_samples, n_classes)
            prob[n, 0] == True if the nth sample is assigned to 1st class
            prob[n, 1] == True if the nth sample is assigned to 2nd class
        """
        self._log("""[WARNING] SVM prediction probabilities are not available.
                 Results from predict will be used instead.""")
        predicted_labels = self.predict(X)
        ret = [np.array([c == self.classes_[0], c == self.classes_[1]])
               for c in predicted_labels]
        return np.array(ret)

    def score(self, X, y):
        """Return the testing accuracy.
           You might want to use a different metric by using sklearn
           cross_val_score

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        accuracy : double
            Accuracy of predictions from X with respect y.
        """
        self._log("Scoring: ", X.shape)
        class1, class0 = self._split_classes(X, y)
        self.test_input = {}
        self.test_input["class1"] = class1
        self.test_input["class0"] = class0
        result = self._run()
        accuracy = result["testing_accuracy"]
        self._log("Testing accuracy = ", accuracy)
        return accuracy


class QuanticSVM(QuanticClassifierBase):

    """Quantum-enhanced SVM classification.

    This class implements SVC [1] on a quantum machine [2].
    Note if `quantum` parameter is set to `False`
    then a classical SVC will be perfomed instead.

    Notes
    -----
    .. versionadded:: 0.0.1

    See Also
    --------
    QuanticClassifierBase

    References
    ----------
    .. [1] Available from: \
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    .. [2] V. Havlíček et al.,
           ‘Supervised learning with quantum-enhanced feature spaces’,
           Nature, vol. 567, no. 7747, pp. 209–212, Mar. 2019,
           doi: 10.1038/s41586-019-0980-2.

    """

    def _run(self, predict_set=None):
        self._log("SVM classification running...")
        if self.quantum:
            self._log("Quantum instance is ", self._quantum_instance)
            qsvm = QSVM(self._feature_map, self._training_input,
                        self.test_input, predict_set)
            result = qsvm.run(self._quantum_instance)
        else:
            result = SklearnSVM(self._training_input,
                                self.test_input, predict_set).run()
        self._log(result)
        return result


class QuanticVQC(QuanticClassifierBase):

    """Variational Quantum Classifier

    Note there is no classical version of this algorithm.
    This will always run on a quantum computer (simulated or not)

    Parameters
    ----------
    labels : see QuanticClassifierBase
    q_account_token : see QuanticClassifierBase
    verbose : see QuanticClassifierBase
    parameters : see QuanticClassifierBase

    Notes
    -----
    .. versionadded:: 0.0.1

    See Also
    --------
    QuanticClassifierBase

    References
    ----------
    .. [1] H. Abraham et al., Qiskit:
           An Open-source Framework for Quantum Computing.
           Zenodo, 2019. doi: 10.5281/zenodo.2562110.

    .. [2] V. Havlíček et al.,
           ‘Supervised learning with quantum-enhanced feature spaces’,
           Nature, vol. 567, no. 7747, pp. 209–212, Mar. 2019,
           doi: 10.1038/s41586-019-0980-2.

    """

    def __init__(self, q_account_token=None,
                 verbose=True, **parameters):
        QuanticClassifierBase.__init__(self,
                                       q_account_token=q_account_token,
                                       verbose=verbose, **parameters)

    def _additional_setup(self):
        self._optimizer = SPSA(maxiter=40, c0=4.0, skip_calibration=True)
        self._var_form = TwoLocal(self._new_feature_dim,
                                  ['ry', 'rz'], 'cz', reps=3)

    def _run(self, predict_set=None):
        self._log("VQC classification running...")
        vqc = VQC(self._optimizer, self._feature_map, self._var_form,
                  self._training_input, self.test_input, predict_set)
        result = vqc.run(self._quantum_instance)
        return result