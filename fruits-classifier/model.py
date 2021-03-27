from IPython import get_ipython
from platform import python_version

import os, re
from glob import glob as ls
import numpy as np                    # Matrix algebra library
import pandas as pd                   # Data table (DataFrame) library
import seaborn as sns; sns.set()      # Graphs and visualization library
from PIL import Image                 # Image processing library


from utilities import *

dataset_dir = os.path.join("./src/", 'images')
images = load_images(dataset_dir, "*.png")

for image in images:
    print(image.mode)
df = pd.DataFrame({'redness':    images.apply(redness),
                   'elongation': images.apply(elongation),
                   'fruit':      images.index.map(lambda name: 1 if name[0] == 'a' else -1),
                  })

print(df.describe())

# %% [markdown]
# Standardizing the dataframe -- that is, for each column, subtracting the mean and dividing by the standard deviation -- is as easy as:

# %%
dfstd =  (df - df.mean()) / df.std()
dfstd

# %% [markdown]
# Admittedly, it takes a bit of head scratching to figure out why this actually works thanks to columnwise operations :-)
# %% [markdown]
# Now all columns are indeed standardized:

# %%
dfstd.describe()

# %% [markdown]
# This is not quite what we want however: the fruit value is not really
# numerical; we want to keep the original values -1 and 1.

# %%
dfstd['fruit'] = df['fruit']
dfstd

# %% [markdown]
# ### Observations
# %% [markdown]
# Let's look at the heatmap of the standardized data frame, which is
# similar to the original one:

# %%
dfstd.style.background_gradient(cmap='RdYlGn_r') # Panda's native heatmap

# %% [markdown]
# We see that **redness is correlated with fruit type**: unsurprisingly,
# apples tend to be red (with exceptions) and bananas green (with
# exceptions).  Meanwhile elongation is **anti-correlated with fruit
# type**: round apples and long bananas. This is confirmed by the
# correlation matrix:

# %%
fig = Figure(figsize=(7,7))
sns.heatmap(dfstd.corr(), fmt='0.2f', annot=True, square=True, cmap='RdYlGn_r', vmin=-1, vmax=1, ax=fig.add_subplot())
fig

# %% [markdown]
# We can spot on the heatmap some outliers (e.g. two red bananas). This
# is confirmed by looking at the scatter plot:

# %%
make_scatter_plot(dfstd, images.apply(transparent_background_filter), axis='square')

# %% [markdown]
# We can also visualize the dataset with pair plots:

# %%
sns.pairplot(dfstd, hue="fruit", diag_kind="hist", palette='Set2');

# %% [markdown]
# Notice that a single feature (redness alone or elongation alone) is
# almost sufficient to perfectly separate apples and bananas: the
# problem is easy!
# %% [markdown]
# ## Step 2: [ME]asuring performance
# 
# Now that we have well understood and prepared our data, we want to
# determine how to evaluate performance for the task at hand: separating
# apples from bananas (a classification problem).
# %% [markdown]
# ### Splitting the data into a training set and a test set
# %% [markdown]
# Let's separate the information in our data table according to its
# nature:
# - `X` will hold the features from which to make predictions
# - `Y` will hold the ground truth: what we try to predict: is it
#   actually an apple or a banana?

# %%
X = dfstd[['redness', 'elongation']]
Y = dfstd['fruit']

# %% [markdown]
# <div class="alert alert-info">
# 
# Why the notations $X$ and $Y$? Because we are on a quest for
# *predictive models* $f$; these try, for each index $i$, to predict the
# ground truth $Y_i$ from the features in $X_i$: ideally, $Y_i=f(X_i)$, for all $i$.
# 
# </div>
# %% [markdown]
# Now, we want to split our images into two subsets, a training set
# (that we will use to adjust the parameters of our predictive models)
# and a test set (to compute prediction performance without being overly
# optimistic).

# %%
# Make one training-test split in a stratified manner, i.e. same number of apples and bananas in each set.
train_index, test_index = split_data(X, Y, verbose = True, seed=0)
Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
Ytrain, Ytest = Y.iloc[train_index], Y.iloc[test_index]

# %% [markdown]
# These images will serve for training the predictive models:

# %%
image_grid(images.iloc[train_index], titles=train_index)

# %% [markdown]
# And these for testing the predictive models:

# %%
image_grid(images.iloc[test_index], titles=test_index)

# %% [markdown]
# <div class="alert alert-info">
# 
# Note that both the training set and the testing set contain the same
# proportion of apples and bananas as in the original data set. This is
# guaranteed, and on purpose!
# 
# </div>
# %% [markdown]
# We plot the training and test data as scatter plots. The test data are shown with question marks, because their class identities (apple or banana) are hidden:

# %%
make_scatter_plot(dfstd, images, train_index, test_index, filter=transparent_background_filter, axis='square')

# %% [markdown]
# ### Exercise: error rate computation
# 
# The *error rate* is a performance metric for predictions. It is defined as the fraction $\frac e n$, where $e$ is the number of incorrect predictions and $n$ the total number of predictions.
# 
# Write a function that computes the error rate, taking as input:
# - A vector `solution` containing the target values (1 for apples and
#   -1 for bananas).
# - A vector `prediction` containing the predicted value

# %%
def error_rate(solution, prediction):
    return np.sum(solution != prediction)/len(solution)

# %% [markdown]
# Then write unit tests (with assert) that check that:
# - the error rate between `solution=Ytrain` and `prediction=Ytrain` is zero (why?)
# - the error rate between `solution=Ytrain` and `prediction=[1,...,1]` is 0.5 (why?)
# - the error rate between `solution=Ytrain` and `prediction=[0,...,0]` is one (why?)
# 
# **Hint**: you may use `np.zeros(Ytrain.shape)` to generate an array `[0,...,0]` of same size as `Ytrain`, and similarly for `[1,...,1]`.

# %%
assert(error_rate(Ytrain, Ytrain) == 0.0)
assert(error_rate(Ytrain, np.ones(len(Ytrain))) == 0.5)
assert(error_rate(Ytrain, np.zeros(len(Ytrain))) == 1)

# %% [markdown]
# The Machine Learning library [`scikit-learn`](https://en.wikipedia.org/wiki/Scikit-learn) also called `sklearn` has a function `accuracy_score`. As an additional verification, we test that `error_rate + accuracy_score = 1` using the same examples as above.

# %%
from sklearn.metrics import accuracy_score
assert abs(error_rate(Ytrain, Ytrain)                 + accuracy_score(Ytrain, Ytrain)                 - 1) <= .1
assert abs(error_rate(Ytrain, np.zeros(Ytrain.shape)) + accuracy_score(Ytrain, np.zeros(Ytrain.shape)) - 1) <= .1
assert abs(error_rate(Ytrain, np.ones(Ytrain.shape))  + accuracy_score(Ytrain, np.ones (Ytrain.shape)) - 1) <= .1

# %% [markdown]
# ## Step 3: [BA]seline results
# %% [markdown]
# ### Exercise: 1-nearest-neighbor classifier
# 
# In a k-nearest neighbor algorithm (KNN), an unlabeled input is
# classified regarding the proximity of its labeled neighbors. The input
# will be predicted as belonging to the class C if the majority of the k
# nearest neighbors belongs to class C. Here, we take k=1 so we only
# consider the closest labeled image to classify an unlabeled image.
# 
# The 1-nearest neighbor classifier is a nice and simple method. It is
# implemented in `scikit-learn` along with many others. You may also
# want to implement it yourself later in the semester at the occasion of
# your project.
# %% [markdown]
# Import the `KNeighborsClassifier` classifier from
# `sklearn.neighbors`. Use it to construct a new model, setting the
# number of neighbors to one. Train this model with `Xtrain` by calling
# the method `fit`. Then use the trained model to create two vectors of
# prediction `Ytrain_predicted` and `Ytest_predicted` by calling the
# method `predict`. Compute `e_tr`, the training error rate, and `e_te`
# the test error rate.  
# **Hint**: look up the documentation as often as needed, starting with
# `KNeighborsClassifier?`
# 
# <!-- WARNING: `scikit-learn` uses lists for prediction labels instead
# of column vectors. You will have to replace `Ytrain` by
# `Ytrain.ravel()` and `Ytest` by `Ytest.ravel()` to avoid an error
# message and wrong error rates.-->

# %%
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(Xtrain, Ytrain)
Ytrain_predicted = neigh.predict(Xtrain)
Ytest_predicted = neigh.predict(Xtest)
print(error_rate(Ytrain_predicted, Ytrain))
print(error_rate(Ytest_predicted, Ytest))

# %% [markdown]
# This problem is too easy! We get zero error on training data and one error on test data!
# %% [markdown]
# ### Here we overlay the predictions on test examples on the scatter plot ...

# %%
# The training examples are shown as white circles and the test examples are black squares.
# The predictions made are shown as letters in the black squares.
make_scatter_plot(X, images.apply(transparent_background_filter),
                  train_index, test_index, 
                  predicted_labels=Ytest_predicted, axis='square')

# %% [markdown]
# ### ... then, we show the "ground truth" and compute the error rate

# %%
# The training examples are shown as white circles and the test examples are blue squares.
make_scatter_plot(X, images.apply(transparent_background_filter),
                  train_index, test_index, 
                  predicted_labels='GroundTruth', axis='square')

# %% [markdown]
# ## Step 4: [BAR]s of error and test set size
# Last but not least, let us evaluate the significance of our results by
# computing error bars. Obviously, since we have only 10 test examples,
# we cannot see at least 100 errors (which is the target we gave to
# ourselves in class). But this is only a toy example.
# %% [markdown]
# ### Exercise: Test set standard error
# 
# Compute the 1-sigma error bar of the test error rate `e_te` using the
# standard error formula defined in class, and assign it to `sigma`.
# How many test examples would we need to divide this error bar by a
# factor of two?

# %%
# YOUR CODE HERE
raise NotImplementedError()
print("TEST SET ERROR RATE: {0:.2f}".format(e_te))
print("TEST SET STANDARD ERROR: {0:.2f}".format(sigma))


# %%
assert abs( sigma - 0.13 ) < 0.1

# %% [markdown]
# ### Cross-validation (CV) error bar
# Another way of computing an error bar is to repeat multiple times the
# train/test split and compute the mean and standard deviation of the
# test error. In some sense this is more informative because it involves
# both the variability of the training set and that of the test set. But
# is is known to be a biased estimator of the error variability.

# %%
n_te = 10
SSS = StratifiedShuffleSplit(n_splits=n_te, test_size=0.5, random_state=5)
E = np.zeros([n_te, 1])
k = 0
for train_index, test_index in SSS.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
    Ytrain, Ytest = Y.iloc[train_index], Y.iloc[test_index]
    neigh.fit(Xtrain, Ytrain.ravel()) 
    Ytrain_predicted = neigh.predict(Xtrain)
    Ytest_predicted = neigh.predict(Xtest)
    e_te = error_rate(Ytest, Ytest_predicted)
    print("TEST ERROR RATE:", e_te)
    E[k] = e_te
    k = k+1
    
e_te_ave = np.mean(E)
# It is bad practice to show too many decimal digits:
print("\n\nCV ERROR RATE: {0:.2f}".format(e_te_ave))
print("CV STANDARD DEVIATION: {0:.2f}".format(np.std(E)))

sigma = np.sqrt(e_te_ave * (1-e_te_ave) / n_te)
print("TEST SET STANDARD ERROR (for comparison): {0:.2f}".format(sigma))

# %% [markdown]
# ## Conclusion
# 
# <div class="alert alert-info">
# 
# This is the end of our first data analysis, where we applied the
# VI-ME-BA-BAR schema to classify pictures of apples and bananas.  We
# applied a lightweight preprocessing to the images to extract two
# features: the redness and elongation of the depicted fruits. Then, we
# [VI]sualized the obtained data, introduced a [ME]tric for the
# classification problem, namely the error rate for predictions. We
# proceeded with a [BA]seline method using a simple nearest neighbor
# classifier. Finally, we estimated the performance of this first
# classifier by computing error [BAR]s over many samples of training /
# testing sets.
# 
# The obtained predictions are fairly robust, which is unsurprising
# given that, up to a few gentle outliers, the pictures in the data are
# well constrained.
# 
# Aiming at more complex data sets from real life, we will in the
# following weeks progressively enrich this schema with more tools.
# 
# </div>
# 
# <div class="alert alert-success">
# 
# You have reached the end of this assignment. Congratulations!
# 
# All you have to do now is to double check the quality of your code
# with the [code review notebook](revue_de_code.ipynb), submit your work,
# and fetch the feedback.
# 
# If you can't wait to explore further, you can engage in the next
# section where you will build your own classifier.
# 
# </div>
# %% [markdown]
# ## Extra credit: build a oneR classifier
# 
# Using the template below, creates a "one rule" (oneR) classifier
# which:
# * selects the "good" feature G (Redness or Elongation), which is most
#   correlated (in absolute value) to the fruit target values y = +- 1;
# * uses G to classify a new example as an apple or a banana by setting
#   a threshold on its values.
#         
# You may follow the template below or try other ideas of your own.

# %%
class oneR():
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.is_trained = False  
        self.ig = 0     # Index of the good feature G
        self.w = 1      # Feature polarity
        self.theta = 0  # Threshold on the good feature

    def fit(self, X, Y):
        '''
        This function should train the model parameters.
        
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            Y: Training label matrix of dim num_train_samples * 1.
        Both inputs are panda dataframes.
        '''
        # Compute correlations
        # YOUR CODE HERE
        raise NotImplementedError()
        np.fill_diagonal(C, 0) # avoid that the max be the diagonal value
        # Select the most correlated feature in absolute value using the last line,
        # and store it in self.ig
        # YOUR CODE HERE
        raise NotImplementedError()
        # Get feature polarity and store it in self.w
        # YOUR CODE HERE
        raise NotImplementedError()
        # Fetch the feature values and multiply by polarity
        G = X.iloc[:, self.ig] * self.w
        # Compute the threshold as a mid-point between cluster centers
        # YOUR CODE HERE
        raise NotImplementedError()
        self.is_trained=True
        print("FIT: Training Successful: Feature selected = %d; Polarity = %d; Threshold = %5.2f." % (self.ig, self.w, self.theta))

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        
        Args:
            X: Test data matrix of dim num_test_samples * num_feat.
        Return:
            Y: Predicted label matrix of dim num_test_samples * 1.
        '''
        # Fetch the feature of interest and multiply by polarity
        G = X.iloc[:,self.ig] * self.w
        # Make decisions according to threshold
        Y = G.copy()
        Y[G < self.theta] = -1  
        Y[G >= self.theta] = 11  
        print("PREDICT: Prediction done")
              
        return Y


# %%
# Use this code to test your classifier
clf = oneR()
clf.fit(Xtrain, Ytrain) 
Ytrain_predicted = clf.predict(Xtrain)
Ytest_predicted = clf.predict(Xtest)
e_tr = error_rate(Ytrain, Ytrain_predicted)
e_te = error_rate(Ytest, Ytest_predicted)
print("MY FIRST CLASSIFIER")
print("Training error:", e_tr)
print("Test error:", e_te)


# %%
# This is what you get as decision boundary.
# The training examples are shown as white circles and the test examples are blue squares.
make_scatter_plot(X, images.apply(transparent_background_filter),
                  [], test_index, 
                  predicted_labels='GroundTruth',
                  feat = clf.ig, theta=clf.theta, axis='square')


# %%
# Compare with what you would get if you used both features, voting with the same weight.
make_scatter_plot(X, images.apply(transparent_background_filter),
                  [], test_index, 
                  predicted_labels='GroundTruth',
                  show_diag=True, axis='square')


