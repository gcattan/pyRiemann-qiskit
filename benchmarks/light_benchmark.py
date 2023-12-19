"""
====================================================================
Light Benchmark
====================================================================

This benchmark is a non-regression performance test, intended
to run on Ci with each PRs.

"""
# Author: Gregoire Cattan
# Modified from plot_classify_P300_bi.py of pyRiemann
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from moabb import set_log_level
from moabb.datasets import bi2012
from moabb.paradigms import P300
from pyriemann_qiskit.utils import distance, mean  # noqa
from pyriemann_qiskit.pipelines import (
    QuantumClassifierWithDefaultRiemannianPipeline,
    QuantumMDMWithRiemannianPipeline,
)
import warnings
import os

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Prepare data
# ----------------
#
##############################################################################

paradigm = P300(resample=128)

dataset = bi2012()  # MOABB provides several other P300 datasets

X, y, _ = paradigm.get_data(dataset, subjects=[1])

# Reduce the dataset size for Ci
_, X, _, y = train_test_split(X, y, test_size=0.7, random_state=42, stratify=y)

y = LabelEncoder().fit_transform(y)

# Separate into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.
#
##############################################################################

pipelines = {}

pipelines["RG+QSVM"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=100,
    nfilter=2,
    dim_red=PCA(n_components=5),
)

pipelines["RG+VQC"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=100, spsa_trials=5, two_local_reps=2
)

pipelines["QMDM-mean"] = QuantumMDMWithRiemannianPipeline(
    convex_metric="mean", quantum=True
)

pipelines["QMDM-dist"] = QuantumMDMWithRiemannianPipeline(
    convex_metric="distance", quantum=True
)

pipelines["RG+LDA"] = make_pipeline(
    XdawnCovariances(
        nfilter=2,
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    TangentSpace(),
    PCA(n_components=5),
    LDA(solver="lsqr", shrinkage="auto"),
)

##############################################################################
# Compute score
# --------------
#
##############################################################################

scores = {}

for key, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    scores[key] = 0

print("Scores: ", scores)

##############################################################################
# Compare scores between PR and main branches
# -------------------------------------------
#
##############################################################################

# parse environment variables
env_file = os.getenv("GITHUB_ENV")
vars = open(env_file, "a").readlines()
git_env = {}
for v in vars:
    pair = v.split("=")
    git_env[pair[0]] = pair[1]

success = True

for key, score in scores.items():
    pr_score = os.getenv(f"PR_SCORE_{key}")
    if not f"PR_SCORE_{key}" in git_env:
        # PR branch
        env_file.write(f"PR_SCORE_{key}", str(score))
    else:
        # Main branch
        success = success and (True if float(pr_score) >= score else False)

env_file.write(f"SUCCESS", "1" if success else "0")
