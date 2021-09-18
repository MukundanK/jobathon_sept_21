import pandas as pd
import config
import argparse
import models
import preprocessor
import os

def run(train, test, target_col, model_name ):

    print('drop columns : ID, target, #Orders from train')
    
    # drop columns
    X_train = train.drop(['ID', '#Order', target_col], axis=1).copy()
    y_train = train[target_col].copy()

    X_test = test.drop('ID', axis=1).copy()

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
    print('make pipeline')
    model = models.model_dict[model_name]

    print('train')
    model.fit(X_train, y_train, categorical_feature = 'auto')

    print('predict')
    y_pred = model.predict(X_test)

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
