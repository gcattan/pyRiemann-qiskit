"""
TODO

"""

# Modified from plot_classify_EEG_tangentspace.py of pyRiemann
# License: BSD (3-clause)

from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from moabb import set_log_level
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances, ERPCovariances, Covariances
from pyriemann.preprocessing import Whitening
from moabb.datasets import Hinss2021

from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.paradigms import P300
from sklearn.decomposition import PCA

# inject convex distance and mean to pyriemann (if not done already)
# from pyriemann_qiskit.utils import distance, mean  # noqa
# from pyriemann_qiskit.pipelines import (
#     QuantumMDMVotingClassifier,
#     QuantumMDMWithRiemannianPipeline,
# )
from sklearn.pipeline import make_pipeline
from pyriemann_qiskit.autoencoders import BasicQnnAutoencoder
from pyriemann_qiskit.utils.preprocessing import Vectorizer, Devectorizer
from pyriemann.spatialfilters import Xdawn

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Initialization
# ----------------
#
# 1) Create paradigm
# 2) Load datasets
from moabb.paradigms import RestingStateToP300Adapter

events = dict(easy=2, medium=3)
paradigm = RestingStateToP300Adapter(events=events, tmin=0, tmax=0.5)

datasets = [Hinss2021()]

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
start_subject = 14
stop_subject = 15
title = "Datasets: "
for dataset in datasets:
    title = title + " " + dataset.code
    dataset.subject_list = dataset.subject_list[start_subject:stop_subject]

# Change this to true to test the quantum optimizer
quantum = False

##############################################################################
# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {"Target": 1, "NonTarget": 0}

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

pipelines = {}

n_times = 64
n_filter = 4
n_classes = 2
n_components = n_classes * n_filter
pipelines["LDA_denoised"] = make_pipeline(
    # select only 2 components
    Xdawn(nfilter=n_filter),
    Vectorizer(),
    BasicQnnAutoencoder(n_components, 1),
    Devectorizer(n_components, n_times),
    Covariances(),
    TangentSpace(),
    # PCA(n_components=4),
    LDA()
)

pipelines["LDA"] = make_pipeline(
    Xdawn(nfilter=4), # Replace by electrodes selection
    Vectorizer(),
    Devectorizer(n_components, n_times),
    Covariances(),
    TangentSpace(),
    LDA()
)

##############################################################################
# Run evaluation
# ----------------
#
# Compare the pipeline using a within session evaluation.

# Here should be cross session
evaluation = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    overwrite=True,
)

results = evaluation.process(pipelines)

autoencoder = pipelines["LDA_denoised"].named_steps['basicqnnautoencoder']

print(autoencoder.costs)
plt.plot(autoencoder.costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Autoencoder Cost')
plt.show()

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

# ##############################################################################
# # Plot Results
# # ----------------
# #
# # Here we plot the results to compare two pipelines

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

sns.stripplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1").set(
    title=title
)

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()