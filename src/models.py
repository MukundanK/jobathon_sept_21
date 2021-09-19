from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

model_dict = {'linear_reg': LinearRegression(), 'rf_reg': RandomForestRegressor(),
'xgb_reg': XGBRegressor(objective='reg:squarederror', n_estimators = 450, eta = 0.05, max_depth = 6, subsample = 0.75, colsample_bytree = 0.75), 'lgbm_reg': LGBMRegressor(),
'cat_reg': CatBoostRegressor(iterations= 400, learning_rate=0.05, depth=6, subsample=0.75, colsample_bylevel=0.75)}