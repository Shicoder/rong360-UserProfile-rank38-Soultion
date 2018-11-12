# coding=utf-8
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
import numpy as np
import gc
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler


valid_data = pd.read_csv("../result/submission_5x-LGB0598965_seed_1024.txt")
print(valid_data.count())
valid_data = valid_data.rename(columns={'prob':'prob1'})
valid2 = pd.read_csv("../result/submission_5x-LGB0600478_seed_1024.txt")
valid2 = valid2.rename(columns={'prob':'prob2'})
valid3 = pd.read_csv("../result/submission_5x-LGB0605474_seed_1024.txt")
valid3 = valid3.rename(columns={'prob':'prob3'})
valid4 = pd.read_csv("../result/submission_5x-LGB0609101_seed_1024.txt")
valid4 = valid4.rename(columns={'prob':'prob4'})
valid5 = pd.read_csv("../result/submission_5x-XGB0596604_seed_1024.txt")
valid5 = valid5.rename(columns={'prob':'prob5'})
valid6 = pd.read_csv("../result/submission_5x-XGB0605969_seed_1024.txt")
valid6 = valid6.rename(columns={'prob':'prob6'})
valid7 = pd.read_csv("../result/submission_5x-XGB0607207_seed_1024.txt")
valid7 =valid7.rename(columns={'prob':'prob7'})

valid_data = pd.merge(valid_data,valid2,how='left',on='id')
valid_data = pd.merge(valid_data,valid3,how='left',on='id')
valid_data = pd.merge(valid_data,valid4,how='left',on='id')
valid_data = pd.merge(valid_data,valid5,how='left',on='id')
valid_data = pd.merge(valid_data,valid6,how='left',on='id')
valid_data = pd.merge(valid_data,valid7,how='left',on='id')

print(valid_data.head())
valid_data['prob'] = (valid_data['prob1']+valid_data['prob2']+
                      valid_data['prob3']+valid_data['prob4']+
                      valid_data['prob5']+valid_data['prob6']+
                      valid_data['prob7'])/7
print(valid_data.count())
valid_data = valid_data[['id','prob']]

# valid_data.to_csv("../result/sub_final.txt",index=False)










params = [
    {'boosting_type': 'gbdt',
     'objective': 'binary',
     'metric': {'auc'},
     'learning_rate': 0.1,
     'max_depth': 10,
     'num_leaves': 40,
     'feature_fraction': 0.3,
     'bagging_freq': 3,
     # 'min_split_gain': 0.2,
     'min_child_weight': 12,
     'reg_alpha': 4,
     'reg_lambda':4 ,
     # 'is_unbalance':True
     # 'verbose': 1},
     },
    {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eta': 0.1,
        # 'max_depth': 10,
        'subsample': 0.5,
        # 'min_child_weight': 5,
        'colsample_bytree': 0.6,
        # 'scale_pos_weight': 0.1,
        'eval_metric': 'auc',
        'alpha':2,
        'lambda': 5,
        'nthread':4
    }

]
############################################################
import lightgbm as lgb
import xgboost as xgb
from  sklearn.linear_model import LogisticRegression
feature = [x for x in train_data.columns if x not in ('label','id')]
print(feature)
valid_data = train_data[train_data['label']==-1]
train_data = train_data[train_data['label']!=-1]
# valid_X = valid_data[feature]
# valid_Y = valid_data['label']
# X = train_data[feature]
y = train_data['label']

# sc = StandardScaler()
# train_data = sc.fit_transform(train_data[[x for x in train_data.columns if x not in ('symbol','label')]])

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1024)
# oof_preds = np.zeros(train_data.shape[0])
# sub_preds = np.zeros(valid_data.shape[0])
# feature_importance_df = pd.DataFrame()
# for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_data, y)):
#     X_train, y_train = train_data[feature].iloc[trn_idx], y.iloc[trn_idx]
#     X_test, y_test = train_data[feature].iloc[val_idx], y.iloc[val_idx]
#
#     lgb_train = lgb.Dataset(X_train, y_train)
#     del X_train, y_train,
#     lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
#
#     clf = lgb.train(params[0],
#                     lgb_train,
#                     num_boost_round=20000,
#                     valid_sets=[lgb_eval],
#                     early_stopping_rounds=200,
#                     verbose_eval=100)
#
#     oof_preds[val_idx] = clf.predict(X_test, num_iteration=clf.best_iteration)
#
#     sub = pd.Series(clf.predict(valid_data[feature], num_iteration=clf.best_iteration)).rank(pct=True).values
#     sub_preds += sub / (folds.n_splits)
#
#     fold_importance_df = pd.DataFrame()
#     fold_importance_df["feature"] = clf.feature_name()
#     fold_importance_df["importance"] = clf.feature_importance()
#     fold_importance_df["fold"] = n_fold + 1
#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
#
#     print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(y_test, oof_preds[val_idx])))
#     del X_test, y_test
#     gc.collect()
# print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
# oof_preds1 = oof_preds
# sub_preds1 = sub_preds
# ################################
oof_preds = np.zeros(train_data.shape[0])
sub_preds = np.zeros(valid_data.shape[0])
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_data, y)):
    X_train, y_train = train_data[feature].iloc[trn_idx], y.iloc[trn_idx]
    X_test, y_test = train_data[feature].iloc[val_idx], y.iloc[val_idx]
    xgb_train = xgb.DMatrix(X_train,y_train)
    # lgb_train = lgb.Dataset(X_train, y_train)
    del X_train, y_train,
    xgb_eval = xgb.DMatrix(X_test,y_test)
    watchlist = [(xgb_train,'train'),(xgb_eval,'val')]

    # lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # clf = lgb.train(params[0],
    #                 lgb_train,
    #                 num_boost_round=20000,
    #                 valid_sets=[lgb_eval],
    #                 early_stopping_rounds=200,
    #                 verbose_eval=100)
    clf = xgb.train(params[1],xgb_train,num_boost_round=10000,evals=watchlist,early_stopping_rounds=100)
    # print(clf.predict(X_test).head())
    test = xgb.DMatrix(X_test)
    oof_preds[val_idx] = clf.predict(test,ntree_limit=clf.best_iteration)

    valid = xgb.DMatrix(valid_data[feature])
    sub = pd.Series(clf.predict(valid,ntree_limit=clf.best_iteration)).rank(pct=True).values
    sub_preds += sub / (folds.n_splits)
########################################################################
    # print(clf.feature_importance)
    # fold_importance_df = pd.DataFrame()
    # fold_importance_df["feature"] = clf.feature_name()
    # fold_importance_df["importance"] = clf.feature_importance()
    # fold_importance_df["fold"] = n_fold + 1
    # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(y_test, oof_preds[val_idx])))
    del X_test, y_test
    gc.collect()
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
# oof_preds1 = (oof_preds1+oof_preds)/2
# sub_preds = (sub_preds+sub_preds1)/2
# print('Full AUC score final %.6f' % roc_auc_score(y, oof_preds1))




res = valid_data[['id']]
res['prob']=sub_preds

res_t = train_data[['id']]
res_t['prob'] = oof_preds

# In[ ]:

score = str(round(roc_auc_score(y, oof_preds), 6)).replace('.', '')
sub_file = '../result/submission_5x-XGB' + score + "_seed_" + str(1024) + '.txt'
sub_t_file = '../result/submission_t_5x-XGB' + score + "_seed_" + str(1024) + '.txt'
res.to_csv(sub_file, index=False)
res_t.to_csv(sub_t_file,index=False)

# In[ ]:

folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(train_data, y)]
# display_importances(feature_importance_df_=feature_importance_df)

# In[ ]:

display_roc_curve(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)

