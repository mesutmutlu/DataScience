#script number 03
import numpy as np
from EndToEndProcess.IngestData.load_data import load_housing_data
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


#to enable new data arrivals or deletes to list we should select an identifier which is unique to rows and will not be change
def split_housing_by_geo(data, test_ratio):
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    return train_set, test_set

#to avoid sampling error of dataset while splitting we should split them by paying attention to distribution of dataset on several features.
def stratified_split(data, test_ratio, column):
    #here we do sampling over income_cat because the most attribute for us is median and the data distribution is very large.
    #we hould have same representation of important feautres in train/test datasets as whole dataset
    strat_train_set = None
    strat_test_set = None
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    for train_index, test_index in split.split(data, data[column]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    return strat_train_set, strat_test_set


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

if __name__ == "__main__":

    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    random_train_set, random_test_set = split_train_test(housing, 0.2)
    print(len(random_train_set), "random_train", len(random_test_set), "random_test")

    housing_with_id = housing.reset_index()  # adds an `index` column
    hashed_train_set, hashed_test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    print(len(hashed_train_set), "hashed_train", len(hashed_test_set), "hashed_test")

    geo_train_set, geo_test_set = split_housing_by_geo(housing, 0.2)
    print(len(geo_train_set), "geo_train", len(geo_test_set), "geo_test")

    strat_train_set, strat_test_set = stratified_split(housing, 0.2, "income_cat")
    print(len(strat_train_set), "strat_train", len(strat_test_set), "strat_test")

    compare_props = pd.DataFrame({
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(random_test_set),
    }).sort_index()
    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

    print(compare_props)

    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)