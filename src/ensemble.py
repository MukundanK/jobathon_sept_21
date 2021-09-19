import pandas as pd
import numpy as np
import os
import config

# read data
dir_path = config.OUTPUT
Model_1 = pd.read_csv(os.path.join(dir_path, 'submission13.csv'))
Model_2 = pd.read_csv(os.path.join(dir_path, 'submission14.csv'))

# average predictions

target_col = config.TARGET
avg_pred = (Model_1[target_col] + Model_2[target_col])/2.
final_pred = {'ID': Model_1['ID'], target_col :avg_pred}
submission = pd.DataFrame(data=final_pred, columns=['ID',target_col])

output = os.path.join(config.OUTPUT, 'ensemble_xgb_cat.csv')
submission.to_csv(output, index=None)
