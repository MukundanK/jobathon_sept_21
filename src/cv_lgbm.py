import config
import pandas as pd
import models
import argparse
import preprocessor
from sklearn.metrics import mean_squared_log_error


def run(df, target_col, model_name):

    print('drop columns : ID, target, #Orders')

    # drop ID, target_col
    X = df.drop(['ID', '#Order', target_col], axis=1).copy()
    y = df[target_col].copy()

    # date ranges for train and test

    train_start = '2018-01-01'
    train_1_end = '2018-11-30'

    test_1_start = '2018-12-01'
    test_1_end = '2019-01-01'

    train_2_end = test_1_end

    test_2_start = '2019-02-01'
    test_2_end = '2019-03-31'

    train_3_end = test_2_end

    test_3_start = '2019-04-01'
    test_3_end = '2019-05-31'

    train_end = [train_1_end, train_2_end, train_3_end]
    test_start = [test_1_start, test_2_start, test_3_start]
    test_end = [test_1_end, test_2_end, test_3_end]

    cv_scores = []

    for a,b,c in zip(train_end, test_start, test_end):
            
        X_train = X.loc[(X['Date']>=train_start) & (X['Date']<=a)]
        X_test = X.loc[(X['Date']>=b) & (X['Date']<=c)]
        y_train = y.iloc[X_train.index]
        y_test = y.iloc[X_test.index]

        # preprocessing

        # extract date, month, year
        X_train = preprocessor.get_date_month_year(X_train)
        X_test = preprocessor.get_date_month_year(X_test)

        # add new features
        X_train = preprocessor.add_new_features(X_train)
        X_test = preprocessor.add_new_features(X_test)

        
        # select categorical variables
        X_train = preprocessor.transform_to_categorical(X_train)
        X_test = preprocessor.transform_to_categorical(X_test)

        #fit model

        print('initiate model')
        model = models.model_dict[model_name]

        print('train')
        model.fit(X_train, y_train, categorical_feature = 'auto')

        print('predict')
        y_pred = model.predict(X_test)
        score = mean_squared_log_error(y_test, y_pred)

        cv_scores.append(score*1000)
    
    print (cv_scores)

    return None


if __name__ == '__main__':
    
    # obtain model and filename from user
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    # parser.add_argument('--filename', type=str)

    args = parser.parse_args()

    # read train 
    train = pd.read_csv(config.TRAIN, parse_dates=[config.DATE])

    target_col = config.TARGET
    model_name = args.model

    run(train , target_col, model_name) 










