﻿# Classification data

It's a version of Python project for data classification using machine learning models

- [x] numpy
- [x] sklearn

The objective of the assignment is to propose a model for data prediction/classification. It is possible to use models
for classification/regression available within the scikit-learn module.

## Data

There is X_public data of size N samples, where each sample contains F flags, i.e. NxF. A corresponding tag vector
y_public of dimension N elements is published to this data. The data can be loaded using numpy.load(). The validation of
your model will be performed on the X_eval data of dimension NexF, which is generated by the same handle as X_public.

## The procedure for working with split data:

- split public into training and test data,
- use fit and transform imputer on training data,
- use fit and transform one-hot-encoder or label-encoderon string columns of training data,
- use fit and transform scalerana training data,
- use the imputer transform on the test data,
- use the one-hot-encoder or label-encoder transform (whichever you choose) on the string columns of the test data,
- apply a scalera transform to the test data,
- perform a grid search on the TEST data,
- verify the accuracy of the best parameters on the test data,
- use imputer transform on eval data,
- use one-hot-encoder (or label-encoder) transform on eval data,
- apply a scalera transform on eval data,
- predict y for eval data.
