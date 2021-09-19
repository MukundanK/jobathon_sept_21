from sklearn.pipeline import Pipeline
import config
import pandas as pd
import models
import argparse
import preprocessor
from sklearn.metrics import mean_squared_log_error
from catboost import Pool


def run(df, target_col, model_name):

    # drop ID
    print ('drop ID column')
    df.drop('ID', axis=1, inplace=True)

    # train-test split
    print ('train-test split')
   
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

        train = df.loc[(df['Date']>=train_start) & (df['Date']<=a)].reset_index(drop=True)
        test = df.loc[(df['Date']>=b) & (df['Date']<=c)].reset_index(drop=True)

        # preprocessing

        # extract date, month, year
        train = preprocessor.get_date_month_year(train)
        test = preprocessor.get_date_month_year(test)

        # make discount binary
        train = preprocessor.make_discount_binary(train)
        test = preprocessor.make_discount_binary(test)

        # get weekend
        train = preprocessor.get_weekend(train)
        test = preprocessor.get_weekend(test)

        # get feature_combination
        train = preprocessor.get_feature_combinations(train)
        test = preprocessor.get_feature_combinations(test)

        # add lag features
        train = preprocessor.get_lag_features(train)
        test = preprocessor.get_lag_features(test)

        # add season
        train = preprocessor.add_seasons(train)
        test = preprocessor.add_seasons(test)

        # dropna
        train.dropna(inplace=True)
        test.dropna(inplace=True)

        # split into features and target
        X_train = train.drop(['#Order', 'Date', target_col], axis=1).copy()
        y_train = train[target_col].copy()

        X_test = test.drop(['#Order', 'Date', target_col], axis=1).copy()
        y_test = test[target_col].copy()

        # select categorical variables
        categorical_cols = X_train.select_dtypes(include = ['object', 'category']).columns.to_list()

        train_data = Pool(data=X_train, label=y_train, cat_features= categorical_cols)
        test_data = Pool(data=X_test, label=y_test, cat_features= categorical_cols)
                 
        #fit model
        print('initiate model')
        model = models.model_dict[model_name]

        print('train')
        model.fit(train_data, eval_set = test_data, plot=False, early_stopping_rounds=50, use_best_model=True)

        print('predict')
        
        y_pred = model.predict(X_test)
        score = mean_squared_log_error(y_test, y_pred)

        cv_scores.append(score*1000)
    
    print(cv_scores)

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










