"""
====================================================================
ERP EEG decoding with Quantum Classifier.
====================================================================

Decoding applied to EEG data in sensor space decomposed using Xdawn.
After spatial filtering, covariances matrices are estimated, then projected in
the tangent space and classified with a quantum SVM classifier.
It is compared to the classical SVM.

"""
# Author: Gregoire Cattan
# Modified from plot_classify_EEG_tangentspace.py
# License: BSD (3-clause)

import numpy as np
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann_qiskit.classification import QuanticSVM
from mne import io, read_events, pick_types, Epochs
from mne.datasets import sample
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import TransformerMixin
from matplotlib import pyplot as plt

# cvxpy is not correctly imported due to wheel not building
# in the doc pipeline
__cvxpy__ = True
try:
    import cvxpy
    del cvxpy
except Exception:
    __cvxpy__ = False

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw.fif"
event_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"
tmin, tmax = -0.0, 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method="iir")  # replace baselining with high-pass
events = read_events(event_fname)

raw.info["bads"] = ["MEG 2443"]  # set bad channels
picks = pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
)

# Read epochs
epochs = Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=False,
    picks=picks,
    baseline=None,
    preload=True,
    verbose=False,
)

X = epochs.get_data()
y = epochs.events[:, -1]

# Reduce the number of classes as QuanticBase supports only 2 classes
y[y % 3 == 0] = 0
y[y % 3 != 0] = 1
labels = np.unique(y)

# Reduce trial number to diminish testing time
class0 = X[y == 0]
class1 = X[y == 1]

n_trials = 50
X = np.concatenate((class0[:n_trials], class1[:n_trials]))
y = np.concatenate(([0] * n_trials, [1] * n_trials))

# ...skipping the KFold validation parts (for the purpose of the test only)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1)

###############################################################################
# Decoding in tangent space with a quantum classifier

# Time complexity of quantum algorithm depends on the number of trials and
# the number of elements inside the covariance matrices
# Thus we reduce elements number by using restrictive spatial filtering
sf = XdawnCovariances(nfilter=1)


# ...and dividing the number of remaining elements by two
class Downsampler(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, ::2]


# Projecting correlation matrices into the tangent space
# as quantum algorithms take vectors as inputs
# (If not, then matrices will be inlined inside the quantum classifier)
tg = TangentSpace()

# https://stackoverflow.com/questions/61825227/plotting-multiple-confusion-matrix-side-by-side

f, axes = plt.subplots(1, 2, sharey='row')

disp = None

# Results will be computed for QuanticSVM versus SKLearnSVM for comparison
for quantum in [True, False]:
    # This is a hack for the documentation pipeline
    if(not __cvxpy__):
        continue

    qsvm = QuanticSVM(verbose=True, quantum=quantum)
    # clf = make_pipeline(sf, cov, tg, Downsampler(), qsvm)
    clf = make_pipeline(sf, tg, Downsampler(), qsvm)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Printing the results
    acc = np.mean(y_pred == y_test)
    acc_str = "%0.2f" % acc

    names = ['0', '1']
    title = ("Quantum (" if quantum else "Classical (") + acc_str + ")"
    axe = axes[0 if quantum else 1]
    cm = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(ax=axe, xticks_rotation=45)
    disp.ax_.set_title(title)
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')
    if not quantum:
        disp.ax_.set_ylabel('')

if disp:
    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()
