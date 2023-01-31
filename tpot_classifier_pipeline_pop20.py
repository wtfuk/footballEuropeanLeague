import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from sklearn.impute import SimpleImputer

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: 0.48328786548615266
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_leaf=1, min_samples_split=16)),
    RobustScaler(),
    RobustScaler(),
    LogisticRegression(C=20.0, dual=False, penalty="l2")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
