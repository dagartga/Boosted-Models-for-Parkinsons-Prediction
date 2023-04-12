# model_dispatcher.py

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor

models = {"rf_reg": RandomForestRegressor(random_state = 42),
          "xgboost": XGBRegressor(random_state = 42),
          "rf_class": RandomForestClassifier(random_state = 42),
         }