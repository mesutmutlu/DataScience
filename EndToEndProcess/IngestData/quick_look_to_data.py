#script number 02
from EndToEndProcess.IngestData.load_data import load_housing_data

if __name__ == "__main__":

    housing = load_housing_data()
    print(housing.head())

    # check for number of observations of each fature, types, nullables, etc.
    print(housing.info())

    #check non numerical features
    print(housing["ocean_proximity"].value_counts())

    #check numerical features for scaling, variance, std, mean, outliers, etc.