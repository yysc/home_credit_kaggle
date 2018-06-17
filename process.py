import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

df_train_data = pd.get_dummies(df_train_raw_data)
df_train = df_train_data.iloc[1:246009]# 307512 lines in total.
df_valid = df_train_data.iloc[246010:]

y_train = df_train['TARGET'].values
X_train = df_train.drop(['SK_ID_CURR','TARGET'], axis=1).values

y_valid = df_valid['TARGET'].values
X_valid = df_valid.drop(['SK_ID_CURR','TARGET'], axis=1).values

X_test = pd.get_dummies(df_test_raw_data).drop(['SK_ID_CURR'], axis=1).values

# # create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# # specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_valid, y_pred) ** 0.5)

y_final_result = gbm.predict(X_test, num_iteration=gbm.best_iteration)
predict_index = df_test_raw_data['SK_ID_CURR']
data = {'SK_ID_CURR':predict_index.values, 'TARGET':y_final_result}
pred_result= pd.DataFrame(data=data).set_index('SK_ID_CURR')
pred_result