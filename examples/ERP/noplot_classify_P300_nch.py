"""
====================================================================
Classification of P300 datasets from MOABB using NCH
====================================================================

Demonstrates classification with QunatumNCH.
Evaluation is done using MOABB.

If parameter "shots" is None then a classical SVM is used similar to the one
in scikit learn.
If "shots" is not None and IBM Qunatum token is provided with "q_account_token"
then a real Quantum computer will be used.
You also need to adjust the "n_components" in the PCA procedure to the number
of qubits supported by the real quantum computer you are going to use.
A list of real quantum  computers is available in your IBM quantum account.

"""
# Author: Anton Andreev
# Modified from plot_classify_EEG_tangentspace.py of pyRiemann
# License: BSD (3-clause)

import warnings

from matplotlib import pyplot as plt
from moabb import set_log_level
from moabb.datasets import bi2013a, bi2012, Cattan2019_VR, Cattan2019_PHMD
from moabb.datasets.compound_dataset import Cattan2019_VR_Il
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import P300, RestingStateToP300Adapter
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances, Covariances, Shrinkage
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann_qiskit.pipelines import QuantumMDMWithRiemannianPipeline
from qiskit_algorithms.optimizers import SPSA, COBYLA, SLSQP
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann_qiskit.classification import QuanticNCH
from pyriemann_qiskit.utils.hyper_params_factory import create_mixer_rotational_X_gates, create_mixer_rotational_XY_gates
from pyriemann.spatialfilters import CSP

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

##############################################################################
# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {"Target": 1, "NonTarget": 0}

# paradigm = P300(resample=128)
events = ["on", "off"]
paradigm = RestingStateToP300Adapter(events=events)

# datasets = [Cattan2019_VR(virtual_reality=True, screen_display=False)]  # MOABB provides several other P300 datasets
datasets = [Cattan2019_PHMD()]
# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
# n_subjects = 5
# for dataset in datasets:
#     dataset.subject_list = dataset.subject_list[0:n_subjects]

overwrite = True  # set to True if we want to overwrite cached results

pipelines = {}

# sf = XdawnCovariances(
#         nfilter=3,
#         classes=[labels_dict["Target"]],
#         estimator="lwf",
#         xdawn_estimator="scm",
#     ),
seed = 884451
# seed = 475751
# seed = None

sf = make_pipeline(
    Covariances(estimator="lwf"),
    CSP(nfilter=3, log=False)
)
pipelines["NCH+RANDOM_HULL"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_hulls_per_class=1,
        n_samples_per_hull=3,
        n_jobs=12,
        subsampling="random",
        quantum=False,
    ),
)

pipelines["NCH+MIN_HULL"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    sf,
    QuanticNCH(
        seed=seed,
        n_hulls_per_class=1,
        n_samples_per_hull=3,
        n_jobs=12,
        subsampling="min",
        quantum=False,
    ),
)

# this is a non quantum pipeline
pipelines["XD+MDM"] = make_pipeline(
    sf,
    MDM(),
)

pipelines["Ts+LDA"] = make_pipeline(
      sf,
      TangentSpace(metric="riemann"),
      LDA(),
  )

pipelines["NCH+RANDOM_HULL_QAOACV"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    sf,
    QuanticNCH(
        seed=seed,
        n_hulls_per_class=1,
        n_samples_per_hull=3,
        n_jobs=12,
        subsampling="random",
        quantum=True,
        # Provide create_mixer to force QAOA-CV optimization
        create_mixer=create_mixer_rotational_X_gates(0),
        shots=100,
        qaoa_optimizer=SPSA(maxiter=100),
        n_reps=2
    ),
)

pipelines["NCH+RANDOM_HULL_NAIVEQAOA"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    sf,
    QuanticNCH(
        seed=seed,
        n_hulls_per_class=1,
        n_samples_per_hull=3,
        n_jobs=12,
        subsampling="random",
        quantum=True,
    ),
)

pipelines["NCH_MIN_HULL_QAOACV"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_hulls_per_class=1,
        n_samples_per_hull=3,
        n_jobs=12,
        subsampling="min",
        quantum=True,
        # Provide create_mixer to force QAOA-CV optimization
        create_mixer=create_mixer_rotational_X_gates(0),
        shots=100,
        qaoa_optimizer=SPSA(maxiter=100),
        n_reps=2
    ),
)

pipelines["NCH_MIN_HULL_NAIVEQAOA"] = make_pipeline(
    sf,
    QuanticNCH(
        seed=seed,
        n_hulls_per_class=1,
        n_samples_per_hull=3,
        n_jobs=12,
        subsampling="min",
        quantum=True,
    ),
)

# pipelines["QMDM_mean"] = QuantumMDMWithRiemannianPipeline(
#     metric={"mean": "qeuclid", "distance": "euclid"},
#     quantum=True,
#     regularization=Shrinkage(shrinkage=0.9),
#     shots=1024,
#     seed=696288,
# )


print("Total pipelines to evaluate: ", len(pipelines))

evaluation = CrossSubjectEvaluation(
    paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite,
    n_splits=3
)

results = evaluation.process(pipelines)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results to compare the two pipelines

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
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1")

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)
plt.xticks(rotation=45)
plt.show()
