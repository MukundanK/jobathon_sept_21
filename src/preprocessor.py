import pandas as pd
import config

def get_lag_features(df):

    # 1-7 days lag
    for i in range(1,8,1):
        df.loc[:,f'Sales_lag_{i}'] = df.groupby(['Store_id'])['Sales'].shift(i)

    # 2 weeks lag
    # df.loc[:,'Sales_2weeks_lag'] = df.groupby(['Store_id'])['Sales'].shift(14)
    return df

def add_seasons(df):

    # winter = 11,12,1,2
    # spring = 3,4,5
    # summer = 6,7,8
    # fall = 9, 10

    def season_func(x):

        if x in [11, 12, 1]:
            return 'Winter'
        elif x in [2, 3, 4]:
            return 'Spring'
        elif x in [5, 6, 7]:
            return 'Summer'
        else :
            return 'Fall'

    df.loc[:,'Season'] = df['month'].apply(lambda x : season_func(x))
    return df


def transform_to_categorical(df):
    for c in df.columns:
        col_type = df[c].dtype
        if col_type == 'object':
            df[c] = df[c].astype('category')
    return df

    
def get_mean_sales(train_full, df, col):
    train_full_col_mean_sales = train_full.groupby(col, as_index= False ).agg(mean_sales=('Sales','mean'))
    df = df.merge(train_full_col_mean_sales, on=col, how='left')
    return df

def get_date_month_year(df):

    # get date, month, year

    df.loc[:,'date'] = df['Date'].dt.day
    df.loc[:,'month'] = df['Date'].dt.month
    df.loc[:,'year'] = df['Date'].dt.year
    df.loc[:,'day_of_week'] = df['Date'].dt.weekday
    df.loc[:,'day_of_year'] = df['Date'].dt.dayofyear
    df.loc[:,'week_of_year'] = df['Date'].dt.weekofyear
    df.loc[:,'quarter'] = df['Date'].dt.quarter

    return df

def make_discount_binary(df):
    
    # make Discount column binary
    df.loc[:,'Discount_binary'] = df['Discount'].apply( lambda x: 1 if x=='Yes' else 0)
    df = df.drop('Discount', axis=1).copy()

    return df

def get_weekend(df):

    # is_weekend
    df.loc[:,'is_weekend'] =[ 1 if x >=5 else 0 for x in df['day_of_week']]
    df.drop('is_weekend', axis=1).copy()

    return df

def get_feature_combinations (df):

    # combination of store_type, location_type and region_code

    # df.loc[:,'Store_Type_Location_Type'] = df['Store_Type'].astype('str')+ df['Location_Type'].astype('str')
    # df.loc[:,'Location_Type_Region_Code'] = df['Location_Type'].astype('str')+ df['Region_Code'].astype('str')
    # df.loc[:,'Store_Type_Region_Code'] = df['Store_Type'].astype('str')+ df['Region_Code'].astype('str')
    df.loc[:,'Store_Type_Location_Type_Region_Code'] = df['Store_Type'].astype('str')+ df['Region_Code'].astype('str')+df['Location_Type'].astype('str')
    return df

 # discount on weekend

def discount_weekend(df):

    def func(X,y):
        if (X == 1) & (y == 1):
            return 1
        else:
            return 0

    df.loc[:,'Discount_on_weekend'] = df.apply(lambda row: func(row['Discount_binary'], row['is_weekend']),axis=1)
    return df


# weekend_holiday
def weekend_holiday(df):
    def func(x,y):
        if (x==1) & (y==1):
            return 'weekend_holiday'
        elif (x==1) & (y==0):
            return 'weekday_holiday'
        elif (x==0) & (y==1):
            return 'weekend_working'
        else:
            return 'weekday_working'
        
    df['weekend_holiday'] = df.apply(lambda row: func(row['Holiday'], row['is_weekend']), axis=1)
    return df
