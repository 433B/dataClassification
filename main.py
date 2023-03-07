import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

X_public = np.load("X_public.npy", allow_pickle=True)
y_public = np.load("y_public.npy", allow_pickle=True)
X_eval = np.load("X_eval.npy", allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(X_public,
                                                    y_public,
                                                    test_size=0.25,
                                                    random_state=None)

enc = preprocessing.OrdinalEncoder(unknown_value=np.nan,
                                   handle_unknown='use_encoded_value')

X_train[:, 180:] = enc.fit_transform(X_train[:, 180:])
X_test[:, 180:] = enc.transform(X_test[:, 180:])

input = SimpleImputer(strategy='most_frequent',
                      missing_values=np.nan)

X_train = input.fit_transform(X_train)
X_test = input.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SVC(kernel='rbf', probability=True)

param_range = [0.1, 1, 10]

parameters_grid = [{'C': param_range,
                    'kernel': ['linear']},
                   {'C': param_range,
                    'gamma': param_range,
                    'kernel': ['rbf']},
                   {'C': param_range,
                    'gamma': param_range,
                    'degree': [1, 2, 3],
                    'kernel': ['poly']}]

grid_search = GridSearchCV(estimator=clf,
                           param_grid=parameters_grid,
                           scoring='roc_auc',
                           cv=5,
                           n_jobs=-1,
                           verbose=1)

grid_search = grid_search.fit(X_train, y_train)
print('Best score: ', grid_search.best_score_)
print('Best parameters: ', grid_search.best_params_)

clf = SVC(kernel='poly',
          decision_function_shape='ovr',
          degree=2,
          C=1,
          gamma='auto')

clf.fit(X_train, y_train)
a = clf.predict(X_test)

print('result: ', roc_auc_score(a, y_test))
X_eval[:, 180:] = enc.transform(X_eval[:, 180:])
X_eval = input.transform(X_eval)
X_eval = scaler.transform(X_eval)

result = grid_search.predict(X_eval)
np.save('y_predikcia.npy', result)
