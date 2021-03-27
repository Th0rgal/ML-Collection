from IPython import get_ipython
from platform import python_version

import os, re
import numpy as np  # Matrix algebra library
import pandas as pd  # Data table (DataFrame) libraryry
from PIL import Image  # Image processing library


from utilities import *

dataset_dir = os.path.join("./src/", "images")
images = load_images(dataset_dir, "*.png")

# we apply the features
df = pd.DataFrame(
    {
        "redness": images.apply(redness),
        "elongation": images.apply(elongation),
        "fruit": images.index.map(lambda name: 1 if name[0] == "a" else -1),
    }
)

# we normalize our values
dfstd = (df - df.mean()) / df.std()
# we fix the fruit column (it doesn't make sense to normalize it)
dfstd["fruit"] = df["fruit"]

# our input set
X = dfstd[["redness", "elongation"]]
# our output set
Y = dfstd["fruit"]

# we randomly split train and test sets (we specify a seed to keep the same results)
# here we get the indexes of the groups: TRAIN: [14 19  2 11  3 18 12  5  9  1] TEST: [ 0 17  4 13  6  8  7 16 10 15]
train_index, test_index = split_data(X, Y, verbose=False, seed=0)
# we find the input data that correspond
Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
# we find the output data that correspond
Ytrain, Ytest = Y.iloc[train_index], Y.iloc[test_index]

# this returns the proportion of wrong guesses
def error_rate(solution, prediction):
    return np.sum(solution != prediction) / len(solution)


from sklearn.neighbors import KNeighborsClassifier

# we select the classifier model we wanna use
neigh = KNeighborsClassifier(n_neighbors=1)
# we train it with our train set
neigh.fit(Xtrain, Ytrain)

# let's try our model
Ytrain_predicted = neigh.predict(Xtrain)
Ytest_predicted = neigh.predict(Xtest)
print(error_rate(Ytrain_predicted, Ytrain)) # should display 0
print(error_rate(Ytest_predicted, Ytest)) # should display 0.2
