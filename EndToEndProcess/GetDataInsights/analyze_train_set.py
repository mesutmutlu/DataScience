#script number 04
import numpy as np
import pandas as pd

from EndToEndProcess.IngestData.load_data import load_housing_data
from EndToEndProcess.IngestData.split_data import stratified_split
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

def corr_matrix(data, column=None):

    corr_matrix = data.corr()

    if column is None:
        return corr_matrix
    else:
        return corr_matrix[column]

if __name__ == "__main__":

    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    strat_train_set, strat_test_set = stratified_split(housing, 0.2, "income_cat")
    print(len(strat_train_set), "strat_train", len(strat_test_set), "strat_test")

    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)

    housing = strat_train_set.copy()

    housing.plot(kind="scatter", x="longitude", y="latitude")
    plt.show(block=False)

    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show(block=False)

    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population",
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                 )
    plt.legend()
    plt.show(block=False)

    print(corr_matrix(housing,"median_house_value").sort_values(ascending=False))

    #correlation graph for seleted attributes
    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show(block=False)

    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1)
    plt.show()