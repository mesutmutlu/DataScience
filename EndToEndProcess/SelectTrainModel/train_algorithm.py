#script number 10
#with whole training set
#with whole training set using CV
#do not forget about ensemble methods (existings or custom created using voting, bagging, pasting, out-of-bag, boosting, stacking, etc.
from EndToEndProcess.PrepareDataForML.pipeline_data_preparation import housing_preparation_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from EndToEndProcess.IngestData.split_data import stratified_split
from EndToEndProcess.IngestData.load_data import load_housing_data
import numpy as np
import sys


def train_simple():
    pass


def regression_evaluate_traing_using_cv(lst_algorithms, dataset, labels):

    for alg_name, alg_obj, eval_met in lst_algorithms:
        print(alg_name, alg_obj, eval_met)

        scores = cross_val_score(alg_obj, housing_prepared, housing_labels,
                                      scoring=eval_met, cv=10)
        rmse_scores = np.sqrt(-scores)

        print(alg_name)
        display_scores(rmse_scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


if __name__ == "__main__":
    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    housing_prepared, housing_labels = housing_preparation_pipeline(housing)

    alg_list= []

    alg_list.append(["tree_reg", DecisionTreeRegressor(), "neg_mean_squared_error"])
    alg_list.append(["lin_reg", LinearRegression(), "neg_mean_squared_error"])
    alg_list.append(["for_reg", RandomForestRegressor(),"neg_mean_squared_error"])

    regression_evaluate_traing_using_cv(alg_list, housing_prepared, housing_labels)

    #print(alg_list)

    sys.exit()


    tree_reg = DecisionTreeRegressor(LinearRegression(),)


    display_scores(rmse_scores)



