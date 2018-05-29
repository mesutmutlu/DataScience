#script number 12
from EndToEndProcess.PrepareDataForML.pipeline_data_preparation import housing_preparation_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from EndToEndProcess.IngestData.load_data import load_housing_data
from sklearn.model_selection import GridSearchCV
from EndToEndProcess.FineTuneModel.tune_hyperparameters import best_estimator_for_reg
from EndToEndProcess.IngestData.split_data import stratified_split
from sklearn.metrics import mean_squared_error

import numpy as np

if __name__ == "__main__":
    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    strat_train_set, strat_test_set = stratified_split(housing, 0.2, "income_cat")
    print(len(strat_train_set), "strat_train", len(strat_test_set), "strat_test")
    print(strat_test_set.head())
    housing_prepared, housing_labels = housing_preparation_pipeline(strat_train_set)



    X_test, y_test = housing_preparation_pipeline(strat_test_set)

    print(X_test, y_test)

    grid_search = best_estimator_for_reg(housing_prepared, housing_labels)

    final_model = grid_search.best_estimator_
    final_predictions = final_model.predict(X_test)

    print(type(y_test.values), type(final_predictions))
    print(np.c_(y_test.values,final_predictions))
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse) # => evaluates to 48,209.6
    print(final_mse)