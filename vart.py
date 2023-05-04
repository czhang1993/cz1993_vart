from sklearn.ensemble._forest import RandomForestRegressor

# wrapping structure in sklearn:
# BaseEstimator
# BaseEnsemble
# BaseForest
# ForestRegressor
# RandomForestRegressor


# define the variational forest regressor class, which is a super class of sklearn.ensemble.RandomForestRegressor
class VFR(RandomForestRegressor):
    def __init__(
        self,
        n_estimators=100,  # number of decision tree estimators = 100
        *,
        criterion="squared_error",  # criterion = [squared error]
        max_depth=None,  # maximum depth
        min_samples_split=2,  # minimum samples split = 2
        min_samples_leaf=1,  # minimum samples leaf = 1
        min_weight_fraction_leaf=0.0,  # minimum weight fraction leaf = 0.0
        max_features=1.0,  # maximum features = 1.0
        max_leaf_nodes=None,  # maximum leaf nodes = Node
        min_impurity_decrease=0.0,  # minimum impurity decrease = 0.0
        bootstrap=True,  # whether to do bootstrap = True
        oob_score=False,  # out of bag score = False
        n_jobs=None,  # number of jobs = None
        random_state=None,  # random state = None
        verbose=0,  # verbose = 0
        warm_start=False,  # warm start = False
        ccp_alpha=0.0,  # ccp_alpha = 0.0
        max_samples=None,  # maximum samples = None
    ):
        super().__init__(
            estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )





