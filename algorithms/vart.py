from sklearn.ensemble._forest import RandomForestRegressor


class VFR(RandomForestRegressor):
    def __init__(self):
        super().__init__(estimator=DecisionTreeRegressor())
    
    def method(self):
        pass



