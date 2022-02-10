# sklearn

1. link
   * [official site](https://scikit-learn.org/)
   * [github](https://github.com/scikit-learn/scikit-learn)
   * [quick start](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)
   * [user guide](https://scikit-learn.org/stable/user_guide.html)
   * [tutorial](http://scikit-learn.org/stable/tutorial/index.html)
   * [github/boost-presentation](https://github.com/madrury/boosting-presentation)
2. install
   * `conda install -c conda-forge scikit-learn`
   * `pip install scikit-learn`
3. TODO: PCA, ICA, random forest, [OOB](http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_oob.html), partial dependence
4. 术语: grid search, cross validation, KNN, nearest neighbor

## manifold

1. t-SNE (t-distributed Stochastic Neighbor Embedding)
   * [distill/how-to-use-t-SNE-effectively](https://distill.pub/2016/misread-tsne/)
   * [sklearn-user-guide/t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne)
   * [github-pages/Laurens-van-der-Maaten/tsne](https://llvdmaaten.github.io/tsne/)
   * [towardsdatascience/t-SNE-clearly-explained](https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a)

## ensemble

### bagging

1. `from sklearn.ensemble import BaggingClassifier`
2. `from sklearn ensemble import BaggingRegressor`
3. randomization in constructing meta-estimator
   * random subsets of original training data
   * random feature selection (decision tree)
4. recommand fully developed decision tree
   > bagging methods work best with strong and complex models (e.g., fully developed decision trees), in contrast with boosting methods which usually work best with weak models (e.g., shallow decision trees)
5. random subset
   * pasting: random samples without replacement (不可放回) `bootstrap=False`
   * bagging: random samples with replacement (可放回) `bootstrap=True`
   * random subsplaces: random subsets without replacement (不可放回), random subsets of the features `bootstrap_features=False`
   * random patches: random samples with replacement (可放回), random subsets of the features `bootstrap_features=True`
6. compare bagging and single model
   * bagging slightly increase the bias terms
   * bagging reduce the variance

### random forest

1. `sklearn.ensemble.RandomForestClassifier`, `sklearn.ensemble.RandomForestRegressor`
2. `bootstrap=True`
3. split feature: best among a random subset of the the features
4. averaging the probabilistic prediction
5. mainly parameters
   * `n_estimators`: stop getting significantly better beyond a critical number of trees
   * `max_features`: Empirical good default values are `n_feature` for regression problems, `sqrt(n_feature)` for classification
   * [empirical]: `max_depth=None` (consume a lot of memory)
6. `n_jobs`

### Extra-Trees (Extremely Randomized Trees)

1. `sklearn.ensemble.ExtraTreesClassifier`, `sklearn ensemble.ExtraTreesRegressor`
2. randomness goes one step further than `RandomForest`
   > thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule
3. `bootstrap=False`
4. on `iris` data, `10` fold cross validation, `ExtraTrees` performs better than `RadomForest`
5. `n_jobs`

### totally random trees embedding

1. **TBA**
2. [Hashing feature transformation using Totally Random Trees](http://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_embedding.html#sphx-glr-auto-examples-ensemble-plot-random-forest-embedding-py)
3. [Manifold learning on handwritten digits: Locally Linear Embedding, Isomap](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py)
4. [Feature transformations with ensembles of trees](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py)

### AdaBoost

1. **TBA**: [adaboost SAMME](https://github.com/jinxin0924/multi-adaboost)
2. `sklearn.ensemble.AdaBoostClassifier`, `sklearn ensemble.AdaBoostRegressor`
3. principal parameter
   * `n_estimators`
   * `learning_rate`: the contribution of the weak learners in the final combination
4. **TBA**: sklearn example

### Gradient Tree Boosting

1. GBDT (Gradient Boosted Decision Tree), GBRT (Gradient Boosted Regression Tree)
2. advantage
   * able to handle data of mixed type (heterogeneous features)
   * rebustness to outliers in output space via robust loss functions
3. disadvantage: scalability, hardly be parallelized (sequential nature)
4. `sklearn.ensemble.GradientBoostingClassifier`
5. principal parameter
   * `n_estimators`
   * `max_depth`, `max_leaf_nodes`: *`max_leaf_nodes=k` gives comparable results to `max_depth=k-1`` but is significantly faster to train at the expense of a slightly higher training error*
   * `learning_rate`
   * `loss`: `least squares[default]` for regression
6. `warm_start`
7. `max_depth`, `max_leaf_nodes`
8. `learning_rate` [sklearn](http://scikit-learn.org/stable/modules/ensemble.html#shrinkage)
   > The parameter learning_rate strongly interacts with the parameter n_estimators, the number of weak learners to fit. Smaller values of learning_rate require larger numbers of weak learners to maintain a constant training error. Empirical evidence suggests that small values of learning_rate favor better test error. [HTF2009] recommend to set the learning rate to a small constant (e.g. learning_rate <= 0.1) and choose n_estimators by early stopping. For a more detailed discussion of the interaction between learning_rate and n_estimators see [R2007].
9. use `max_features` to reduce runtime
10. subsampling, [Stochastic gradient boosting](https://www.sciencedirect.com/science/article/pii/S0167947301000652)
    * shrinkage outperforms no-shrinkage
    * Subsampling with shrinkage can further increase the accuracy of the model
11. OOB estimates are usually very pessimistic thus we recommend to use cross-validation instead and only use OOB if cross-validation is too time consuming
    * **TBA**: add sklearn example

Note Classification with more than 2 classes requires the induction of n_classes regression trees at each iteration, thus, the total number of induced trees equals n_classes * n_estimators. For datasets with a large number of classes we strongly recommend to use RandomForestClassifier as an alternative to GradientBoostingClassifier .

## voting classifier

1. `sklearn.ensemble.VotingClassifier`
2. majority voting (hard voting)
3. weighted average probabilities (soft voting)

## module linear_model

1. `sklearn.linear_model`
2. `linear_model.LinearRegression` (ordinary least squares)
   * `coef_`, `interrcept_`
   * instable when terms are correlated (linear depedent between columns): a slight change in the target variable can cause huge variances in the calculated weights
   * for dataset `X(np,?,(n,p))`, complexity $O \left( np^2 \right)$
3. `linear_model.Ridge()` (`L2` regularizer)
   * `linear_model.RidgeCV`: Generalized Cross-Validation
4. `linear_model.Lasso` (`L1` regularizer)
   * reducing the number of variables
   * **compressed sensing** [Compressive sensing: tomography reconstruction with L1 prior (Lasso)](http://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#sphx-glr-auto-examples-applications-plot-tomography-l1-reconstruction-py)
   * coordinate descent algorithm. [Least angle Regression](http://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression)for another implementation
   * `linear_model.lasso_path`
   * [L1-based feature selection](http://scikit-learn.org/stable/modules/feature_selection.html#l1-feature-selection)
   * `linear_model.LassoCV`: high-dimensional datasets with many collinear regressors
   * `linear_model.LassoLarsCV`: AIC (Akaike Information Criterion) and BIC (Bayes Information Criterion)
   * `linear_model.MultiTaskLasso`
5. `linear_model.ElasticNet` (mixed `L1` `L2` regularizer)
   * `.l1_ratio`
   * when multiple features correlates, `Lasso` picks one at random, while `ElasticNet` is likely to pick both
   * `linear_model.ElasticNetCV`
   * `linear_model.MultiTaskElasticNet`
6. `linear_model.OrthogonalMatchingPursuit`
7. Bayesian Regression
   * Bayesian Ridge Regression
   * ARD (Automatic Relevance Determination)
8. Logistic regression (logit regression, maximum-entropy classfication, log-linear classifier)
   * `L1`: `liblinear` or `saga`
   * Multinomail loss: `lbfgs` `sag` `newton-cg`
   * large dataset: `sag` or `saga`
   * `saga` is often the best, `liblinear` solver is used by default for historical reasons
   * `linear_model.LogisticRegressionCV`
9. `linear_model.SGDClassifier` `linear_model.SGDRegressor`
   * `loss=log`: logistic regression
   * `loss=hinge`: linear SVM
10. `linear_model.Percepron`
11. `linear_model.PassiveAggressiveClassifier`, `linear_model.PassiveAggressiveRegressor`

## module feature_selection

1. `sklearn.feature_selection`
2. `VarianceThreshold`
3. `SelectKBest`, `SelectPercentile`, `SelectFpr`, `SelectFdr`, `SelectFwe`, `GenericUnivariateSelect`
   * regression: `f_regression` `mutual_info_regression`
   * classification: `chi2` `f_classif` `mutual_info_classif`
4. recursive feature elimination: `RFE`
   * `RFECV`
5. `SelectFromModel`
   * L1-based feature selection
   * [Classification of text documents using sparse features](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py)
   * L1-recovery and compressive sensing
   * Tree based feature selection

## module random_projection

1. `from sklean import random_projection`
2. `random_projection.GaussianRandomProjection()`

## module datasets

1. `from sklearn import datasets`
2. `datasets.load_digits()`
3. Iris flower dataset
4. [Pen-Based Recognition of Handwritten Digits Data Set](http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)
5. boston house prices
6. breast cancer
7. diabetes
8. linnerud
9. mlcomp

## module metrics

1. `from sklearn import metrics`
2. `metrics.classification_report()`
3. `metrics.confusion_matrix()`

## module preprocessing

1. `from sklearn import preprocessing`
2. **RBF kernel of SVM**, **L1**, **L2** assume that all features are centered around zero and have variance in the same order
3. `StandardScaler()`
   * `scale()`
   * `mean_`, `var_`
4. `MinMaxScaler()` / `MaxAbsScaler()`
   * robust to small standard deviations of features
   * preserve zero entries in sparse data
5. sparse data
   * scaling only (not recommand centering)
   * recommand in CSR (Compressed Sparse Rows) representation
6. `RobustScaler()`
   * outliers: [Should I normalize/standardize/rescale the data](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)
7. `QuantileTransformer()`
   * `quantiles_`
   * `output_distribution='normal'`
8. `Normalizer()`
   * `normalize()`
9. `Binarizer()`
10. `OneHotEncoder()`
11. `Imputer()`
    * `strategy='mean/mdeian/most_frequent'`
    * not always improve the predictions, please check with cross-validation
    * `median` is more robust for data with high magnitude variables (long tail)
12. `PolynomialFeatures()`
    * polynomial features are used implicitly in kernel methods (e.g., sklearn.svm.SVC, sklearn.decomposition.KernelPCA) when using polynomial Kernel functions
13. `FunctionTransformer()`

## module pipeline

1. `sklearn.pipeline`
2. advantage
   * call `fit()` and ``predict()` once
   * `GridSearch()` over all parameters in the pipeline at once
3. all estimators except the last one, must be transformers `transform()`, the last one can be any type (transformer, classifier, regressor etc.)
4. `from sklearn.pipeline import Pipeline`
   * `.steps[0]`
   * `.named_steps['pca']`
   * `<estimator>__<parameter>`, `SVC_C`
   * `.set_params(SVC_C=10)`
5. `from sklearn.pipeline import make_pipeline`
6. `GridSearchCV(pipe, {'SVC_C':[0.1,10,100]})`
7. individual may be replaced as parameter
8. non-final steps may be ignored by setting as `None`

## Outliers

1. preprocess: `preprocessing.RobustScaler`
2. RANSAC
3. Theil Sen
4. HuberRegressor

## cluster

### K-means (Lloyd’s algorithm)

1. minimize inertia (within-cluster sum-of-squares)
2. cluster centroids
3. specified k
4. scalable (large number of samples)
5. drawback
   * assumption that clusters are convex and isotropic, (e.g. elongated clusters, manifold with irregular shapes)
   * inertia is not a normalized metric, inflated in very high-dimensional spaces (PCA before K-means is necessary)
6. equivalent to the expectation-maximization algorithm with a small, all-equal, diagonal covariance matrix
7. local minimum (compute several times with different initializations, ```init='k-means++'```), [paper](http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)

### Mini Batch K-means

1. variant of the K-Means: drastically reduce the time, generally only slightly worse than the standard algorithm

### Affinity Propagation

1. send messages between pairs of samples
2. choose the number of clusters based on the data provided
3. important parameters
   * `preference`: control how many exemples are used
   * `dampling`: dump the responsibility and availability messages to avoid numerical oscillations when updating messages
4. drawback:
   * time complexity: `O(N^2T)`, where N is the number of sample and T is the number of iterations
   * memory complexity: `O(N^2)`

### Mean Shift

1. discover blobs in a smooth density of samples, centroid based algorithm
2. automatically set the number of clusters
3. drawback
   * not highly scalable

### Spectral

1. specified the number of clusters
2. required sparese affinity matrix and [pyamg](https://github.com/pyamg/pyamg) module
3. work well for a small number of clusters but is not advised when using many clusters

### hierarchical clustering

1. [wiki](https://en.wikipedia.org/wiki/Hierarchical_clustering)
2. linkage criteria
   * ward: minimize the sum of squared differences within all clusters
   * maximum / complete linkage: minimize the maximum distance between observations of pairs of clusters
   * average linkage: minimize the average of the distance between all observations of pairs of clusters
   * single linkage: minimize the distance between the closest observation of pairs of clusters
3. scalable

## ensemble集成

1. reference
   * [sklearn-ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html#forest)
2. 两大类
   * averaging method: bagging, forests of randomized tree
   * boosting method: adaboost, gradient tree boosting
3. bagging
   * pasting: random subsets of the dataset are drawn as random subsets of the samples
   * bagging: samples are drawn with replacement
   * random subspace: random subsets of the features
   * random patch: subsets of both samples and features
4. `sklearn.ensemble` randomized tree
   * `.RandomForestClassifier`
   * `.RandomForestRegressor`
   * `.ExtraTreesClassifier`, 随机选择一些`threshold`再从中选最优
   * `.ExtraTreesRegressor`
   * `n_estimators`：越大越好，直至饱和
   * `max_features`：（sklearn-doc经验规则）对于回归问题`max_features=None`选择所有feature，对于分类问题`max_features="sqrt"`
   * `max_dpeth`：（sklearn-doc经验规则）`max_depth=None`不限制树深度
   * `min_samples_split`（sklearn-doc经验规则）`min_samples_split=2`
   * feature importance evalution
5. totally random trees embedding，用random forest来做词嵌入，感兴趣
6. adaboost, 存在`SAMME.R`显著优于`SAMME`的数据
