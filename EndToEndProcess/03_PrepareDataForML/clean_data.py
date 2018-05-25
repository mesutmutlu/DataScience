#script number 06
import numpy as np
import pandas as pd

from EndToEndProcess.IngestData.load_data import load_housing_data
from EndToEndProcess.IngestData.split_data import stratified_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import Imputer


def spareta_input_output(data,label):
    #X is attributes, Y is labels depending to X
    X = data.drop(label, axis=1)
    Y = data[label].copy()
    return X, Y


def housing_fill_na_numeric(dataset, columns=None, exc_columns=None, strategy = "median"):

    if not isinstance(columns, list) and columns is not None:
        return "columns variable should be list"

    if not isinstance(exc_columns, list) and exc_columns is not None:
        return "excluded columns variable should be list"

    imputer = Imputer(strategy=strategy)

    if columns is not None:
        dataset = dataset[columns]

    if exc_columns is not None:
        housing_num = dataset.drop(exc_columns, axis=1)
    else:
        housing_num = dataset

    imputer.fit(housing_num)
    print(imputer.statistics_)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    return housing_tr


if __name__ == "__main__":

    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    strat_train_set, strat_test_set = stratified_split(housing, 0.2, "income_cat")
    print(len(strat_train_set), "strat_train", len(strat_test_set), "strat_test")

    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)

    housing = strat_train_set.copy()

    housing, housing_labels = spareta_input_output(housing, ["median_house_value"])
    print(housing.columns)
    print(housing_labels)

    housing_tr = housing_fill_na_numeric(dataset=housing, exc_columns=["ocean_proximity"])
    print(housing_tr)
