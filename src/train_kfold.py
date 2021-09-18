
# import
import pandas as pd
import config
import os
from sklearn import model_selection

def create_folds(df, target_col, processed_dir):

    # create kfold column and fill with 0
    df['kfold'] = 0

    # kfold
    kf = model_selection.KFold(n_splits=3)
    for f, (t_, v_) in enumerate(kf.split(X = df, y = df[target_col].values)):
        df.loc[v_, 'kfold'] = f

    output_file = os.path.join(processed_dir, 'train_kfold.csv')
    df.to_csv(output_file, index = None)

    return None

if __name__ == '__main__':

    data = config.TRAIN
    target_col = config.TARGET
    processed_dir = config.PROCESSED

    train = pd.read_csv(data)
    create_folds(train, target_col, processed_dir)
