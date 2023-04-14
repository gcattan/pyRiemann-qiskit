"""
====================================================================
Synchronization of classification results with Firebase
====================================================================

It demonstrates how to use firebase utils function to synchronize
the classification of P300 dataset using MOABB.

"""
# Author: Gregoire Cattan
# Modified from classify_P300_bi.py
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann_qiskit.utils import (
        generate_caches,
        filter_subjects_by_incomplete_results,
        add_moabb_dataframe_results_to_caches,
        convert_caches_to_dataframes
    )
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from moabb import set_log_level
from moabb.datasets import bi2012
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300
from pyriemann_qiskit.classification import \
    QuantumClassifierWithDefaultRiemannianPipeline
from sklearn.decomposition import PCA

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

labels_dict = {"Target": 1, "NonTarget": 0}

paradigm = P300(resample=128)

datasets = [bi2012()]  
copy_datasets = [bi2012()]

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
n_subjects = 5
for dataset, copy_dataset in zip(datasets, copy_datasets):
    dataset.subject_list = dataset.subject_list[0:n_subjects]
    copy_dataset.subject_list = copy_dataset.subject_list[0:n_subjects]

pipelines = {}

pipelines["RG+QuantumSVM"] = QuantumClassifierWithDefaultRiemannianPipeline(
    shots=None,
    nfilter=2,  
    dim_red=PCA(n_components=5)
    )

pipelines["RG+LDA"] = make_pipeline(
    XdawnCovariances(
        nfilter=2,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm"
    ),
    TangentSpace(),
    PCA(n_components=10),
    LDA(solver="lsqr", shrinkage="auto")
)

"""
We cache the results on Firebase.
Instead of connection to Firebase, you can provide
a dictionnary with mock_data to `generate_caches`. For example:

mock_data = {
   "Brain Invaders 2012":{
      "RG+QuantumSVM":{
         "Brain Invaders 2012":{
            "1":{
               "RG+QuantumSVM":{
                  "true_labels":0.3433986306190491,
                  "predicted_labels":0.5
               }
            },
         }
      },
      "RG+LDA":{
         "Brain Invaders 2012":{
            "1":{
               "RG+LDA":{
                  "true_labels":0.13163451850414276,
                  "predicted_labels":0.6777283549308777
               }
            },
            "2":{
               "RG+LDA":{
                  "true_labels":0.13357791304588318,
                  "predicted_labels":0.8143579959869385
               }
            }
         }
      }
   }
}
"""

caches = generate_caches(datasets, pipelines)

# This method remove a subject in a dataset if we already have evaluated
# all pipelines for this subject.
# Therefore we will use a copy of the original datasets.
filter_subjects_by_incomplete_results(caches, copy_datasets, pipelines)

print("Total pipelines to evaluate: ", len(pipelines))
print("Subjects to evaluate",
      sum([len(dataset.subject_list) for dataset in copy_datasets]))

evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=copy_datasets,
    suffix="examples",
    overwrite=True
)

try:
    results = evaluation.process(pipelines)
    add_moabb_dataframe_results_to_caches(results,
                                          copy_datasets,
                                          pipelines,
                                          caches)
except ValueError:
    print("No subjects left to evaluate.")

df = convert_caches_to_dataframes(caches, datasets, pipelines)

print("Averaging the session performance:")
print(df.groupby('pipeline').mean('score')[['score', 'time']])

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results to compare the two pipelines

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

sns.stripplot(
    data=df,
    y="score",
    x="pipeline",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)

sns.pointplot(data=df,
              y="score",
              x="pipeline",
              ax=ax,
              palette="Set1")

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()
