#script number 02
from EndToEndProcess.IngestData.load_data import load_housing_data
import matplotlib.pyplot as plt

if __name__ == "__main__":

    housing = load_housing_data()
    print(housing.head())

    # check for number of observations of each fature, types, nullables, etc.
    print(housing.info())

    #check non numerical features
    print(housing["ocean_proximity"].value_counts())

    #check numerical features for scaling, variance, std, mean, outliers, etc.
    print(housing.describe())

    #check value distributiosn using histograms, check if values are scaled,
    # there are maximum limits defined for example 500.000  of house value for each observation having > 500.000
    #check graph shapes, features with heavy long tail should be processed to have better curved shapes
    # by scaling or removing outliers, etc.. But not now this will be done in data preparation
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()

    #now time to split data to train and test sets