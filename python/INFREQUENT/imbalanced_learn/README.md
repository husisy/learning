# imbalanced-learn

1. link
   * [official documents](http://contrib.scikit-learn.org/imbalanced-learn/stable/introduction.html)
2. `Sampler`: `.fit() .sample() .fit_sample()`
   * `.__call__(data, target)`: for sqarse data, prefer `csr` representation

## module over_sampling

1. `RandomOverSampler`
2. `SMOTE`
3. `ADASYN`
   * focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier: more samples will be generated in the area that the nearest neighbor rule is not respected
4. both `SMOTE` and `ADASYN` may lead to sub-optimal decision function
   * $x_{new}=x_i + \lambda \times \left( x_j -x_i \right)$
   * `SMOTE` variant and `ADASYN` differ by selecting the samples $x_i$ ahead of generating the new samples
   * `regular`: randomly pick-up all possible $x_i$
   * `borderline1`: `noise / in danger / safe`, choose $x_i$ belonging to `in danger` , $x_j$ from same label
   * `borderline2`: `noise / in danger / safe`, choose $x_i$ belonging to `in danger` , $x_j$ from any label
   * `svm`: uses an SVM classifier to find support vectors and generate samples considering them

## module under_sampling

1. prototype generation
2. prototype selection
3. `ClusterCentroids`
   * the centrroids of the cluster
4. `RandomUnderSampler`
5. `NearMiss`
   * `version=1`: selects samples from the majority class for which the average distance of the k` nearest samples of the minority class is the smallest
   * `version=2`: selects the samples from the majority class for which the average distance to the farthest samples of the negative class is the smallest
   * `version=3`: for each minority sample, their :m nearest-neighbors will be kept; then, the majority samples selected are the on for which the average distance to the k nearest neighbors is the largest
6. `EditedNearestNeighbours`
7. `RepeatedEditedNearestNeighbours`
8. `AllKNN`
9. `CondensedNearestNeighbour`: sensitive to noise
10. `OneSidedSelection`
    * `TomekLinks` to remove the samples considered noise
11. `NeighbourhoodCleaningRule`
    * `EditedNearestNeighbours` to remove samples
12. `InstanceHardnessThreshold`: use the prediction of classifier to exclude samples

## module combine

1. `SMOTENN`
2. `SMOTETomek`

## module ensemble

1. `EasyEnsemble`
   * `n_subsets`
   * `replacement`
2. `BalanceCascade`
3. `BalanceBaggingClassifier`
