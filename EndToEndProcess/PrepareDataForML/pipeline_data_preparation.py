#script number 10
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from EndToEndProcess.PrepareDataForML.extract_new_features import CombinedAttributesAdder
from EndToEndProcess.IngestData.load_data import load_housing_data
from EndToEndProcess.IngestData.split_data import stratified_split
import numpy as np
from sklearn.pipeline import FeatureUnion
from EndToEndProcess.tools.data_frame_selector import DataFrameSelector
from EndToEndProcess.tools.future_encoders import OneHotEncoder

def housing_preparation_pipeline():
    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    strat_train_set, strat_test_set = stratified_split(housing, 0.2, "income_cat")
    print(len(strat_train_set), "strat_train", len(strat_test_set), "strat_test")

    housing = strat_train_set
    housing_labels = strat_train_set["median_house_value"].copy()
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('one_hot_encoder', OneHotEncoder(sparse=False)),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared, housing_labels


if __name__ == "__main__":

    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    strat_train_set, strat_test_set = stratified_split(housing, 0.2, "income_cat")
    print(len(strat_train_set), "strat_train", len(strat_test_set), "strat_test")

    housing = strat_train_set
    housing_labels = strat_train_set["median_house_value"].copy()
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('one_hot_encoder', OneHotEncoder(sparse=False)),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing.columns)
    print(housing_prepared.shape)