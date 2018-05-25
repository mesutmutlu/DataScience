#script number 05
import numpy as np
import pandas as pd

from EndToEndProcess.IngestData.load_data import load_housing_data
from EndToEndProcess.IngestData.split_data import stratified_split
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix


if __name__ == "__main__":

    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    strat_train_set, strat_test_set = stratified_split(housing, 0.2, "income_cat")
    print(len(strat_train_set), "strat_train", len(strat_test_set), "strat_test")

    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)

    housing = strat_train_set.copy()

    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))