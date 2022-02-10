import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification, make_regression, load_iris

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-5: round(hfe(x,y,eps),5)

def hf_softmax(data, axis=-1):
    tmp0 = data - data.max(axis=axis, keepdims=True)
    tmp1 = np.exp(tmp0)
    return tmp1 / tmp1.sum(axis=axis, keepdims=True)
hf_logistic = lambda x: 1 / (1 + np.exp(-x))

class MyCopyTree(object):
    def __init__(self, sk_tree):
        self.children_left = sk_tree.children_left #(np,int,(num_node,))
        self.children_right = sk_tree.children_right #(np,int,(num_node,))
        self.feature = sk_tree.feature #(np,int,(num_node,))
        self.threshold = sk_tree.threshold #(np,float,(num_node,))
        self.value = sk_tree.value #(np,float,(num_node,num_label,num_class))
    def is_leaf(self, ind1):
        return (self.children_left[ind1]==-1) and (self.children_right[ind1]==-1)
    def transform_single(self, X, ind1):
        if self.is_leaf(ind1):
            return self.value[ind1]
        tmp1 = X[self.feature[ind1]] <= self.threshold[ind1]
        if tmp1:
            return self.transform_single(X, self.children_left[ind1])
        else:
            return self.transform_single(X, self.children_right[ind1])
    def transform(self, X):
        return np.array([self.transform_single(x,0) for x in X])


def sklearn_decision_tree_regressor(num_train=233, num_feature=13, num_test=23):
    x_train, y_train = make_regression(n_samples=num_train, n_features=num_feature)
    x_test = np.random.rand(num_test, num_feature)

    regr = DecisionTreeRegressor(max_depth=5)
    regr.fit(x_train, y_train)

    ret_ = regr.predict(x_test)
    copy_tree = MyCopyTree(regr.tree_)
    ret = copy_tree.transform(x_test)[:,0,0]
    print('sklearn_decision_tree_regressor:: sklearn vs np :', hfe_r5(ret_, ret))


def sklearn_decision_tree_classifier(num_test=23):
    tmp0 = load_iris()
    x_train = tmp0.data
    y_train = tmp0.target
    x_test = np.random.rand(num_test, x_train.shape[1])
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(x_train, y_train)
    ret_ = clf.predict_proba(x_test)
    copy_tree = MyCopyTree(clf.tree_)
    tmp0 = copy_tree.transform(x_test)[:,0]
    ret = tmp0 / tmp0.sum(axis=1, keepdims=True)
    print('sklearn_decision_tree_classifier:: sklearn vs np :', hfe_r5(ret_, ret))


# TODO classification
def sklearn_random_forest_regressor(num_train=233, num_feature=13, num_test=23, num_estimator=13):
    x_train, y_train = make_regression(n_samples=num_train, n_features=num_feature)
    x_test = np.random.rand(num_test, num_feature)

    regr = RandomForestRegressor(n_estimators=num_estimator, min_samples_split=2)
    regr.fit(x_train, y_train)

    ret_ = regr.predict(x_test)
    copy_tree = [MyCopyTree(x.tree_) for x in regr.estimators_]
    ret = np.stack([x.transform(x_test)[:,0,0] for x in copy_tree]).mean(axis=0)
    print('sklearn_random_forest_regressor:: sklearn vs np: ', hfe_r5(ret_, ret))


def sklearn_random_forest_classifier(num_test=23, num_estimator=13):
    tmp0 = load_iris()
    x_train = tmp0.data
    y_train = tmp0.target
    x_test = np.random.rand(num_test, x_train.shape[1])

    clf = RandomForestClassifier(n_estimators=num_estimator, min_samples_split=2)
    clf.fit(x_train, y_train)

    ret_ = clf.predict_proba(x_test)
    copy_tree = [MyCopyTree(x.tree_) for x in clf.estimators_]
    tmp0 = [x.transform(x_test)[:,0] for x in copy_tree]
    ret = np.stack([x/x.sum(axis=1,keepdims=True) for x in tmp0], axis=0).mean(axis=0)
    print('sklearn_random_forest_classifier:: sklearn vs np: ', hfe_r5(ret_, ret))

def test_export_graphviz():
    try:
        import graphviz
    except ModuleNotFoundError:
        print('install graphviz first, "conda install -c conda-forge graphviz"')
        return
    clf = DecisionTreeClassifier(max_depth=3)
    tmp0 = load_iris()
    clf.fit(tmp0.data, tmp0.target)
    dot_data = export_graphviz(clf)
    graph = graphviz.Source(dot_data)
    os.startfile(graph.render(tempfile.NamedTemporaryFile().name))


def sklearn_adaboost_classifier(num_test=23, num_estimator=13):
    tmp0 = load_iris()
    x_train = tmp0.data
    y_train = tmp0.target
    x_test = np.random.rand(num_test, x_train.shape[1])

    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=num_estimator, learning_rate=0.5, algorithm='SAMME.R')
    clf.fit(x_train, y_train)
    ret0_ = clf.predict_proba(x_test)
    ret1_ = list(clf.staged_predict_proba(x_test))

    hf_clip_zero = lambda x: np.clip(x, np.finfo(x.dtype).eps, None)
    tmp0 = np.stack([np.log(hf_clip_zero(x.predict_proba(x_test))) for x in clf.estimators_], axis=0)
    tmp1 = np.arange(1,num_estimator+1)[:,np.newaxis,np.newaxis]
    ret1 = hf_softmax(np.cumsum(tmp0, axis=0) / tmp1, axis=-1)
    print('sklearn_adaboost_classifier.predict_proba:: np vs sklearn: ', hfe_r5(ret0_, ret1[-1]))
    print('sklearn_adaboost_classifier.staged_predict_proba:: np vs sklearn: ', hfe_r5(ret1_, ret1))


def sklearn_gradient_boosting_regressor(num_train=233, num_feature=13, num_test=23, num_estimator=13, learning_rate=0.1):
    x_train, y_train = make_regression(n_samples=num_train, n_features=num_feature)
    x_test = np.random.rand(num_test, num_feature)

    regr = GradientBoostingRegressor(loss='ls', learning_rate=learning_rate, n_estimators=num_estimator, max_depth=3)
    regr.fit(x_train, y_train)
    ret0_ = regr.predict(x_test)
    ret1_ = np.stack(list(regr.staged_predict(x_test)), axis=0)

    tmp0 = [x.predict(x_test) for x in regr.estimators_[:,0]]
    ret1 = np.cumsum(np.stack(tmp0, axis=0), axis=0) * learning_rate + regr.init_.constant_[0,0]
    print('sklearn_gradient_boosting_regressor.predict:: np vs sklearn: ', hfe_r5(ret0_, ret1[-1]))
    print('sklearn_gradient_boosting_regressor.staged_predict:: np vs sklearn: ', hfe_r5(ret1_, ret1))


def sklearn_gradient_boosting_classifier(num_train=233, num_feature=13, num_test=23, num_estimator=13, learning_rate=0.233):
    x_train, y_train = make_classification(n_samples=num_train, n_features=num_feature, n_classes=2)
    x_test = np.random.rand(num_test, x_train.shape[1])

    clf = GradientBoostingClassifier(n_estimators=num_estimator, learning_rate=learning_rate)
    clf.fit(x_train, y_train)

    ret0_ = clf.decision_function(x_test)
    ret1_ = np.stack(list(clf.staged_decision_function(x_test)), axis=0)[:,:,0]
    ret2_ = np.stack(list(clf.staged_predict_proba(x_test)), axis=0)

    class_prior = np.log(np.sum(y_train==1) / (np.sum(y_train==0)))
    tmp0 = [MyCopyTree(x.tree_).transform(x_test)[:,0,0] for x in clf.estimators_[:,0]]
    ret1 = np.cumsum(np.stack(tmp0, axis=0), axis=0)*learning_rate + class_prior
    ret2 = np.stack([1-hf_logistic(ret1), hf_logistic(ret1)], axis=2)
    print('sklearn_gradient_boosting_classifier.decision_function:: np vs sklearn: ', hfe_r5(ret0_, ret1[-1]))
    print('sklearn_gradient_boosting_classifier.staged_decision_function:: np vs sklearn: ', hfe_r5(ret1_, ret1))
    print('sklearn_gradient_boosting_classifier.staged_predict_proba:: np vs sklearn: ', hfe_r5(ret2_, ret2))


if __name__=='__main__':
    sklearn_decision_tree_regressor()
    print()
    sklearn_decision_tree_classifier()
    print()
    sklearn_random_forest_regressor()
    print()
    sklearn_random_forest_classifier()
    print()
    test_export_graphviz()
    print()
    sklearn_adaboost_classifier()
    print()
    sklearn_gradient_boosting_regressor()
    print()
    sklearn_gradient_boosting_classifier()
