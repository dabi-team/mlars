from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, neighbors,  linear_model, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


def new_function():
    new_models = [
        ["SVM", [
            lambda nu: linear_model.SGDOneClassSVM(nu=nu),
            lambda fit_intercept: linear_model.SGDOneClassSVM(
                fit_intercept=fit_intercept),
            lambda tol: linear_model.SGDOneClassSVM(tol=tol),
            lambda max_iter: linear_model.SGDOneClassSVM(max_iter=max_iter),
            lambda loss: linear_model.SGDOneClassSVM(loss=loss),
            lambda alpha: linear_model.SGDOneClassSVM(alpha=alpha),
            lambda l1_ratio: linear_model.SGDOneClassSVM(l1_ratio=l1_ratio),
            lambda fit_intercept: linear_model.SGDOneClassSVM(
                fit_intercept=fit_intercept),
            lambda max_iter: linear_model.SGDOneClassSVM(max_iter=max_iter),
            lambda loss: linear_model.SGDOneClassSVM(loss=loss),
            lambda alpha: linear_model.SGDOneClassSVM(alpha=alpha),
        ], [
            # nu
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # fit_intercept
            [True, False],
            # tol
            [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,
                0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # max_iter
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            # alpha
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # l1_ratio
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # fit_intercept
            [True, False],
            # max_iter
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            # loss
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            # alpha
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ], ],
        ["logistic regression", [
            lambda penalty: linear_model.LogisticRegression(penalty=penalty),
            lambda dual: linear_model.LogisticRegression(dual=dual),
            lambda tol: linear_model.LogisticRegression(tol=tol),
            lambda C: linear_model.LogisticRegression(C=C),
            lambda fit_intercept: linear_model.LogisticRegression(
                fit_intercept=fit_intercept),
            lambda intercept_scaling: linear_model.LogisticRegression(
                intercept_scaling=intercept_scaling),
            lambda class_weight: linear_model.LogisticRegression(
                class_weight=class_weight),
            lambda random_state: linear_model.LogisticRegression(
                random_state=random_state),
            lambda max_iter: linear_model.LogisticRegression(
                max_iter=max_iter),
            lambda multi_class: linear_model.LogisticRegression(
                multi_class=multi_class),
            lambda verbose: linear_model.LogisticRegression(verbose=verbose),
            lambda warm_start: linear_model.LogisticRegression(
                warm_start=warm_start),
            lambda n_jobs: linear_model.LogisticRegression(n_jobs=n_jobs),
            lambda l1_ratio: linear_model.LogisticRegression(l1_ratio=l1_ratio), ], [
            # penalty
            ["l1", "l2"],
            # dual
            [True, False],
            # tol
            [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,
                0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # C
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # fit_intercept
            [True, False],
            # intercept_scaling
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # class_weight
            [None, "balanced"],
            # random_state
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # max_iter
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            # multi_class
            ["ovr", "multinomial"],
            # verbose
            [0, 1, 2],
            # warm_start
            [True, False],
            # n_jobs
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # l1_ratio
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ], ],
        ["Random Forest", [
            lambda n_estimators: RandomForestClassifier(
                n_estimators=n_estimators),
            lambda max_depth: RandomForestClassifier(max_depth=max_depth),
            lambda min_samples_split: RandomForestClassifier(
                min_samples_split=min_samples_split),
            lambda min_samples_leaf: RandomForestClassifier(
                min_samples_leaf=min_samples_leaf),
            lambda min_weight_fraction_leaf: RandomForestClassifier(
                min_weight_fraction_leaf=min_weight_fraction_leaf),
            lambda max_features: RandomForestClassifier(
                max_features=max_features),
            lambda max_leaf_nodes: RandomForestClassifier(
                max_leaf_nodes=max_leaf_nodes),
            lambda min_impurity_decrease: RandomForestClassifier(
                min_impurity_decrease=min_impurity_decrease),
            lambda min_impurity_split: RandomForestClassifier(
                min_impurity_split=min_impurity_split),
            lambda bootstrap: RandomForestClassifier(bootstrap=bootstrap),
            lambda oob_score: RandomForestClassifier(oob_score=oob_score),
            lambda n_jobs: RandomForestClassifier(n_jobs=n_jobs),
            lambda random_state: RandomForestClassifier(
                random_state=random_state),
            lambda verbose: RandomForestClassifier(verbose=verbose),
            lambda warm_start: RandomForestClassifier(warm_start=warm_start),
            lambda class_weight: RandomForestClassifier(
                class_weight=class_weight),
            lambda ccp_alpha: RandomForestClassifier(ccp_alpha=ccp_alpha),

        ], [
            # n_estimators
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            # max_depth
            [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_samples_split
            [2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_samples_leaf
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_weight_fraction_leaf
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # max_features
            ["auto", "sqrt", "log2"],
            # max_leaf_nodes
            [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_impurity_decrease
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # min_impurity_split
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # bootstrap
            [True, False],
            # oob_score
            [True, False],
            # n_jobs
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # random_state
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # verbose
            [0, 1, 2],
            # warm_start
            [True, False],
            # class_weight
            [None, "balanced"],
            # ccp_alpha
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ], ],
        ["KNN", [
            lambda n_neighbors: neighbors.KNeighborsClassifier(
                n_neighbors=n_neighbors),
            lambda weights: neighbors.KNeighborsClassifier(weights=weights),
            lambda algorithm: neighbors.KNeighborsClassifier(
                algorithm=algorithm),
            lambda leaf_size: neighbors.KNeighborsClassifier(
                leaf_size=leaf_size),
            lambda p: neighbors.KNeighborsClassifier(p=p),
            lambda metric: neighbors.KNeighborsClassifier(metric=metric),
            lambda metric_params: neighbors.KNeighborsClassifier(
                metric_params=metric_params),
            lambda n_jobs: neighbors.KNeighborsClassifier(n_jobs=n_jobs),
        ], [
            # n_neighbors
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # weights
            ["uniform", "distance"],
            # algorithm
            ["auto", "ball_tree", "kd_tree", "brute"],
            # leaf_size
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # p
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # metric
            ["euclidean", "manhattan", "chebyshev"],
            # metric_params
            [{}, {"p": 1}, {"p": 2}],
            # n_jobs
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ], ],
        ["Decision Tree", [
            lambda criterion: DecisionTreeClassifier(criterion=criterion),
            lambda splitter: DecisionTreeClassifier(splitter=splitter),
            lambda max_depth: DecisionTreeClassifier(max_depth=max_depth),
            lambda min_samples_split: DecisionTreeClassifier(
                min_samples_split=min_samples_split),
            lambda min_samples_leaf: DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf),
            lambda min_weight_fraction_leaf: DecisionTreeClassifier(
                min_weight_fraction_leaf=min_weight_fraction_leaf),
            lambda max_features: DecisionTreeClassifier(
                max_features=max_features),
            lambda random_state: DecisionTreeClassifier(
                random_state=random_state),
            lambda max_leaf_nodes: DecisionTreeClassifier(
                max_leaf_nodes=max_leaf_nodes),
            lambda min_impurity_decrease: DecisionTreeClassifier(
                min_impurity_decrease=min_impurity_decrease),
            lambda class_weight: DecisionTreeClassifier(
                class_weight=class_weight),
            lambda ccp_alpha: DecisionTreeClassifier(
                ccp_alpha=ccp_alpha),
        ], [
            # criterion
            ["gini", "entropy"],
            # splitter
            ["best", "random"],
            # max_depth
            [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_samples_split
            [2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_samples_leaf
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_weight_fraction_leaf
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # max_features
            ["auto", "sqrt", "log2"],
            # max_leaf_nodes
            [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_impurity_decrease
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # class_weight
            [None, "balanced"],
            # ccp_alpha
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ], ],
        ["Multinomial Naive Bayes", [
            # alpha=1, fit_prior=True, class_prior=None
            lambda alpha: MultinomialNB(alpha=alpha),
            lambda fit_prior: MultinomialNB(fit_prior=fit_prior),
            lambda class_prior: MultinomialNB(class_prior=class_prior),
        ], [
            # alpha
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # fit_prior
            [True, False],
            # class_prior
            [None, [0.5, 0.5]],
        ], ],
        ["Bernoulli Naive Bayes", [
            # alpha=1, fit_prior=True, class_prior=None
            lambda alpha: BernoulliNB(alpha=alpha),
            lambda fit_prior: BernoulliNB(fit_prior=fit_prior),
            lambda class_prior: BernoulliNB(class_prior=class_prior),
        ], [
            # alpha
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # fit_prior
            [True, False],
            # class_prior
            [None, [0.5, 0.5]],
        ], ],
        ["AdaBoost", [
            # base_estimator, n_estimators, learning_rate, algorithm, random_state
            lambda base_estimator: AdaBoostClassifier(
                base_estimator=base_estimator),
            lambda n_estimators: AdaBoostClassifier(
                n_estimators=n_estimators),
            lambda learning_rate: AdaBoostClassifier(
                learning_rate=learning_rate),
            lambda algorithm: AdaBoostClassifier(algorithm=algorithm),
            lambda random_state: AdaBoostClassifier(random_state=random_state),
        ], [
            # base_estimator
            [DecisionTreeClassifier(),
             RandomForestClassifier(),
             ExtraTreesClassifier(),
             GradientBoostingClassifier()],
            # n_estimators
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # learning_rate
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # algorithm
            ["SAMME", "SAMME.R"],
            # random_state
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ], ],
        ["Gradient Boosting", [
            # base_estimator, n_estimators, learning_rate, algorithm, random_state
            lambda base_estimator: GradientBoostingClassifier(
                base_estimator=base_estimator),
            lambda n_estimators: GradientBoostingClassifier(
                n_estimators=n_estimators),
            lambda learning_rate: GradientBoostingClassifier(
                learning_rate=learning_rate),
            lambda algorithm: GradientBoostingClassifier(algorithm=algorithm),
            lambda random_state: GradientBoostingClassifier(
                random_state=random_state),
        ], [
            # base_estimator
            [DecisionTreeClassifier(),

             RandomForestClassifier(),
             ExtraTreesClassifier(),
             GradientBoostingClassifier()],
            # n_estimators
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # learning_rate
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # algorithm
            ["SAMME", "SAMME.R"],
            # random_state
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ], ],
        ["Extra Trees", [
            # n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, class_weight, random_state
            lambda n_estimators: ExtraTreesClassifier(
                n_estimators=n_estimators),
            lambda criterion: ExtraTreesClassifier(
                criterion=criterion),
            lambda max_depth: ExtraTreesClassifier(
                max_depth=max_depth),
            lambda min_samples_split: ExtraTreesClassifier(
                min_samples_split=min_samples_split),
            lambda min_samples_leaf: ExtraTreesClassifier(
                min_samples_leaf=min_samples_leaf),
            lambda min_weight_fraction_leaf: ExtraTreesClassifier(
                min_weight_fraction_leaf=min_weight_fraction_leaf),
            lambda max_features: ExtraTreesClassifier(
                max_features=max_features),
            lambda max_leaf_nodes: ExtraTreesClassifier(
                max_leaf_nodes=max_leaf_nodes),
            lambda min_impurity_decrease: ExtraTreesClassifier(
                min_impurity_decrease=min_impurity_decrease),
            lambda class_weight: ExtraTreesClassifier(
                class_weight=class_weight),
            lambda random_state: ExtraTreesClassifier(
                random_state=random_state),
        ], [
            # n_estimators
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # criterion
            ["gini", "entropy"],
            # max_depth
            [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_samples_split

            [2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_samples_leaf
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_weight_fraction_leaf
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # max_features
            [None, "auto", "sqrt", "log2"],
            # max_leaf_nodes
            [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # min_impurity_decrease
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # class_weight
            [None, "balanced"],
            # random_state
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ], ],





    ]
