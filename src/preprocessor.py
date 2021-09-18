import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import config

def preprocessing(df, target_col):

    # drop ID, fold, target_col
    df = df.drop(['ID','kfold', target_col], axis=1).copy()

    # get date, month, year
    df['date'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year

    df = df.drop('Date', axis=1).copy()

    # make Discount column binary
    df['Discount_binary'] = df['Discount'].apply( lambda x: 1 if x=='Yes' else 0)
    df = df.drop('Discount', axis=1).copy()

    # select categorical variables
    categorical_cols = df.select_dtypes(exclude = ['int64', 'float64']).columns

    categorical_transformer = OneHotEncoder()
    preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_cols)], remainder ='passthrough')
    
    return preprocessor


if __name__ == '__main__':


    train = pd.read_csv(config.TRAIN, parse_dates=['Date'])
    target_col = config.TARGET

    preprocessing(train, target_col)