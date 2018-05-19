import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-8.831734170906685
exported_pipeline = GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="lad", max_depth=5,
                                              max_features=0.8500000000000001, min_samples_leaf=13,
                                              min_samples_split=14, n_estimators=100, subsample=0.6500000000000001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
