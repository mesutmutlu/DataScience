#script number 07
import numpy as np
from EndToEndProcess.tools.future_encoders import OrdinalEncoder
from EndToEndProcess.IngestData.load_data import load_housing_data
from EndToEndProcess.IngestData.split_data import stratified_split
from EndToEndProcess.tools.future_encoders import OneHotEncoder



if __name__ == "__main__":
    housing = load_housing_data()

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    strat_train_set, strat_test_set = stratified_split(housing, 0.2, "income_cat")
    print(len(strat_train_set), "strat_train", len(strat_test_set), "strat_test")

    housing_cat = housing[['ocean_proximity']]
    print(housing_cat)

    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print(housing_cat_encoded)
    print(ordinal_encoder.categories_)
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot)
    print(housing_cat_1hot.toarray())

    cat_encoder = OneHotEncoder(sparse=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot)

    print(cat_encoder.categories_)