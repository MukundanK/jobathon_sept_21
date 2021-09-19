import pandas as pd
import config
import argparse
import models
import preprocessor
from sklearn.pipeline import Pipeline
from catboost import Pool
import os

def run(train, test, target_col, model_name ):

    # drop ID column
    print('drop ID column')     
    train.drop('ID', axis=1, inplace=True)
    test.drop('ID', axis=1, inplace=True)

    # extract date, month, year
    train = preprocessor.get_date_month_year(train)
    test = preprocessor.get_date_month_year(test)

    # add sales value to test
    test = test.merge(train[['date','month','Store_id','Sales']], how='inner', on=['date','month','Store_id'])

    # avg sales 2019 / avg sales 2018 = 1.009
    test['Sales'] = test['Sales'] * 1.009

    # concat test and train
    train_test = pd.concat([train, test])
    
    # lag features
    train_test = preprocessor.get_lag_features(train_test)

    # split into train and test
    test = train_test[((train_test['year']==2019) & (train_test['month'].isin([6,7])))].copy()
    train = train_test[~((train_test['year']==2019) & (train_test['month'].isin([6,7])))].copy()

    # make discount binary
    train = preprocessor.make_discount_binary(train)
    test = preprocessor.make_discount_binary(test)

    # get weekend
    train = preprocessor.get_weekend(train)
    test = preprocessor.get_weekend(test)

    # get feature_combination
    train = preprocessor.get_feature_combinations(train)
    test = preprocessor.get_feature_combinations(test)

     # split into features and target
    X_train = train.drop(['#Order', 'Date', target_col], axis=1).copy()
    y_train = train[target_col].copy()

    X_test = test.drop(['Date', target_col], axis=1).copy()

    # select categorical variables
    categorical_cols = train.select_dtypes(include = ['object', 'category']).columns.to_list()
    train_data = Pool(data=X_train, label=y_train, cat_features= categorical_cols)
    test_data = Pool(data = X_test, cat_features=categorical_cols)
    
    #fit model
    print('initiate model')
    model = models.model_dict[model_name]

    print('train')
    model.fit(train_data, plot=False)

    print('predict')
        
    y_pred = model.predict(test_data)

    print('complete')
    return y_pred


if __name__ == '__main__':
    
    # obtain model and filename from user
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--filename', type=str)

    args = parser.parse_args()

    # read data
    train = pd.read_csv(config.TRAIN, parse_dates=[config.DATE])
    test = pd.read_csv(config.TEST, parse_dates =[config.DATE])

    target_col = config.TARGET
    model_name = args.model
    test_ID = test['ID'].values

    y_pred = run(train, test, target_col, model_name) 

    # output
    data = {'ID':test_ID, 'Sales':y_pred}
    test_pred = pd.DataFrame(data, columns = ['ID','Sales'])

    output = os.path.join(config.OUTPUT, args.filename)
    test_pred.to_csv(output, index=None)
