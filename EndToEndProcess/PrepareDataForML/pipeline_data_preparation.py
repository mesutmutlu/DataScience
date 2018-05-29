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

def housing_preparation_pipeline(strat_train_set):

    #dataset = strat_train_set
    dataset_labels = strat_train_set["median_house_value"].copy()
    dataset = strat_train_set.drop("median_house_value", axis=1)
    dataset_num = dataset.drop("ocean_proximity", axis=1)

    num_attribs = list(dataset_num)
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

    dataset_prepared = full_pipeline.fit_transform(dataset)
    return dataset_prepared, dataset_labels


if __name__ == "__main__":

    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    strat_train_set, strat_test_set = stratified_split(housing, 0.2, "income_cat")
    print(len(strat_train_set), "strat_train", len(strat_test_set), "strat_test")

    housing_preparation_pipeline(strat_train_set)

