import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount
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

# Average CV score on the training set was: 0.4849060445307831
exported_pipeline = make_pipeline(
    ZeroCount(),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_features=0.2, min_samples_leaf=3, min_samples_split=18, n_estimators=100, subsample=0.65)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
