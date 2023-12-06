# https://geomstats.github.io/notebooks/05_practical_methods__simple_machine_learning_on_tangent_spaces.html
# Learning on Tangent Data
import numpy as np
import matplotlib.pyplot as plt

import geomstats
import geomstats.geometry.spd_matrices
import geomstats.datasets.utils
import geomstats.learning.preprocessing

import sklearn.pipeline
import sklearn.linear_model
import sklearn.model_selection

data, patient_ids, labels = geomstats.datasets.utils.load_connectomes()
flat_data, _, _ = geomstats.datasets.utils.load_connectomes(as_vectors=True)
# data (np,float64,(86,28,28))
# flat_data (np,float64,(86,378)) #378=27*28/2
labels_str = ["Healthy", "Schizophrenic"]

manifold = geomstats.geometry.spd_matrices.SPDMatrices(28, equip=False)
assert np.all(manifold.belongs(data))
spd_ai = geomstats.geometry.spd_matrices.SPDMatrices(28, equip=False)
spd_ai.equip_with_metric(geomstats.geometry.spd_matrices.SPDAffineMetric)
spd_le = geomstats.geometry.spd_matrices.SPDMatrices(28, equip=False)
spd_le.equip_with_metric(geomstats.geometry.spd_matrices.SPDLogEuclideanMetric)

for space in [spd_ai, spd_le]:
    pipeline = sklearn.pipeline.Pipeline(
        steps=[
            ("feature_ext", geomstats.learning.preprocessing.ToTangentSpace(space=space)),
            ("classifier", sklearn.linear_model.LogisticRegression(C=2)),
        ]
    )
    result = sklearn.model_selection.cross_validate(pipeline, data, labels)
    print(result["test_score"].mean())
# 0.7098039215686274
# 0.6862745098039216

model = sklearn.linear_model.LogisticRegression()
flat_result = sklearn.model_selection.cross_validate(model, flat_data, labels)
print(flat_result["test_score"].mean())
# 0.7333333333333334
