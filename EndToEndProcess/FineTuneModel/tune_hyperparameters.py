#script number 11
#using gridsearch
#using randomsearch
from EndToEndProcess.PrepareDataForML.pipeline_data_preparation import housing_preparation_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from EndToEndProcess.IngestData.load_data import load_housing_data
from sklearn.model_selection import GridSearchCV
import numpy as np
from EndToEndProcess.IngestData.split_data import stratified_split

def best_estimator_for_reg(data,labels):
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8, 10, 12]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(data, labels)

    return grid_search
if __name__ == "__main__":

    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    strat_train_set, strat_test_set = stratified_split(housing, 0.2, "income_cat")
    print(len(strat_train_set), "strat_train", len(strat_test_set), "strat_test")

    housing_prepared, housing_labels = housing_preparation_pipeline(strat_train_set)


    grid_search = best_estimator_for_reg(housing_prepared, housing_labels)

    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    print(feature_importances)

    #extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    #cat_one_hot_attribs = list(encoder.classes_)
    #attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    #sorted(zip(feature_importances, housing_prepared.columns), reverse=True)