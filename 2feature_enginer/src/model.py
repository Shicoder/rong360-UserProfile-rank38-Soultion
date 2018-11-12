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

def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)]
    best_features.to_csv('../photo/lgbm_importances-01.csv',index=False)
    plt.figure(figsize=(10, 15))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('../photo/lgbm_importances-01.svg')

def display_roc_curve(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(fpr, tpr, color='b', label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)), lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig('../photo/roc_curve-01.svg')

train_data = pd.read_csv("../open_data/sample_train.txt",delimiter='\t',encoding='utf-8')
print("label 1:",train_data[train_data['label']==1]['id'].count())
print("label 0:",train_data[train_data['label']==0]['id'].count())
# train_data = train_data.append(train_data[train_data['label']==1])
valid_data = pd.read_csv("../open_data/test_id.txt")
#
valid_data['label'] = -1
train_data = train_data.append(valid_data)
print(train_data.head())
print("all_id:",train_data['label'].count())
# #################################################
dat_risk = pd.read_csv("../feature/dat_risk.txt",delimiter='\t')
print("dat_risk:",dat_risk.head())
dat_risk['de_cnt'] = dat_risk['d_cnt']+dat_risk['e_cnt']
dat_risk['cde_cnt'] = dat_risk['c_cnt']+dat_risk['d_cnt']+dat_risk['e_cnt']
dat_risk['bc_cnt'] = dat_risk['b_cnt']+dat_risk['c_cnt']
# dat_risk['bc_cross_cnt'] = dat_risk['b_cnt']*dat_risk['c_cnt']
dat_risk['all_cnt'] = dat_risk['a_cnt']+dat_risk['b_cnt']+dat_risk['c_cnt']+\
                   dat_risk['d_cnt']+dat_risk['e_cnt']
dat_risk['de_cross_cnt'] = dat_risk['d_cnt']*dat_risk['e_cnt']
train_data = pd.merge(train_data,dat_risk,how='left',on='id')
# print("train_data:",train_data[train_data['label']==1].head())
train_data = train_data.fillna(0)
print("all row1:",train_data['id'].count())
del dat_risk
# #################################################
dat_symbol1 = pd.read_csv("../feature/dat_symbol_feature.txt",delimiter='\t')
# dat_symbol1 = dat_symbol1[[x for x in dat_symbol1.columns if x not in
#                            ('symbol_1','symbol_2','symbol_3','symbol_4','symbol_5','symbol_6',
#                             'symbol_7','symbol_8','symbol_9','symbol_10','symbol_11','symbol_12','symbol_13',
#                             'symbol_14','symbol_15','symbol_16','symbol_17','symbol_18','symbol_19','symbol_20',
#                             'symbol_21','symbol_22','symbol_23','symbol_24')]]
# dat_symbol2 = pd.read_csv("../feature/dat_symbol_feature_2.txt",delimiter='\t')
dat_symbol2 = pd.read_csv("../feature/dat_symbol.txt",delimiter='\t')
train_data = pd.merge(train_data,dat_symbol1,how='left',on='id')
train_data = train_data.fillna(0)
# train_data = train_data.dropna()
# print("row!!:",train_data['id'].count())
train_data = pd.merge(train_data,dat_symbol2,how='left',on='id')
train_data = train_data.fillna('UNKNOWN')

train_data['symbol'] = LabelEncoder().fit_transform(train_data['symbol'])

dat_symbol_c = [x for x in train_data.columns if x.startswith('symbol_')]
train_data[dat_symbol_c] = train_data[dat_symbol_c].astype('int')
# train_data['a_cnt_symbol'] = train_data['symbol']*train_data['a_cnt']
# train_data['b_cnt_symbol'] = train_data['symbol']*train_data['b_cnt']
# train_data['c_cnt_symbol'] = train_data['symbol']*train_data['c_cnt']
# train_data['d_cnt_symbol'] = train_data['symbol']*train_data['d_cnt']
# train_data['e_cnt_symbol'] = train_data['symbol']*train_data['e_cnt']

print("train_data2:",train_data.head())
print("all row1:",train_data['id'].count())
# del dat_symbol1
del dat_symbol2

#######################################################
# dat_edge = pd.read_csv("../feature/train_valid_test_dat_edge_feature.txt",delimiter='\t',header=None,
#                           names=['from_id','to_id','year','month','cnum','weight'])
# dat_edge['year_month'] = dat_edge['year'].astype('str')+dat_edge['month'].astype('str')
# t1 = dat_edge[['from_id','year_month','cnum','weight']].groupby(['from_id','year_month']).sum().reset_index()
# t1 = t1.rename(columns={'cnum':'y_id_cnum','weight':'y_id_weight'})
#
# t1 = t1.set_index(['from_id','year_month'])
# t1 = t1.unstack()
# t1.columns = ['201711_cnum','201712_cnum','20181_cnum',
#                       '201711_weight','201712_weight','20181_weight']
# t1['all_cnum'] = t1['201711_cnum']+t1['201712_cnum']+t1['20181_cnum']
# t1['all_weight'] = t1['201711_weight'] +t1['201712_weight']+t1['20181_weight']
# t1['all_mean_cnum_weight'] = t1['all_weight'].astype('float')/t1['all_cnum']
# t1['all_cross_cnum_weight'] = t1['all_weight']*t1['all_cnum']
# t1 = t1.fillna(0)
# t1.to_csv("../feature/train_valid_test_dat_edge_info_feature.txt",index=True)
edge_info = pd.read_csv("../feature/train_valid_test_dat_edge_info_feature.txt")
# edge_info['201711_mean_cnum_weight'] = edge_info['201711_weight'].astype('float')/edge_info['201711_cnum']
# edge_info['201712_mean_cnum_weight'] = edge_info['201712_weight'].astype('float')/edge_info['201712_cnum']
# edge_info['20181_mean_cnum_weight'] = edge_info['20181_weight'].astype('float')/edge_info['20181_cnum']
# edge_info = edge_info[[x for x in edge_info.columns if x not in ('201711_cnum','201711_weight',
#                                                                  '201712_cnum','201712_weight',
#                                                                  '20181_cnum','20181_weight')]]
edge_info = edge_info.rename(columns={'from_id':'id'})
train_data = pd.merge(train_data,edge_info,how='left',on='id')
train_data = train_data[[x for x in train_data.columns if x not in ('all_cross_cnum_weight')]]

# train_data['20181_mean_cnt_weight'] = train_data['20181_weight'].astype('float')/train_data['20181_cnum']
# train_data['201712_mean_cnt_weight'] = train_data['201712_weight'].astype('float')/train_data['201712_cnum']
# train_data['201711_mean_cnt_weight'] = train_data['201711_weight'].astype('float')/train_data['201711_cnum']
# train_data['20181_201712_mean'] = train_data['20181_mean_cnt_weight']/train_data['201712_mean_cnt_weight']
# train_data['20181_201711_mean'] = train_data['20181_mean_cnt_weight']/train_data['201711_mean_cnt_weight']
# train_data['201712_201711_mean'] = train_data['201712_mean_cnt_weight']/train_data['201711_mean_cnt_weight']
train_data['symbol_mean_cnum_weight'] = train_data['symbol']*train_data['all_mean_cnum_weight']
del edge_info
print("all row2:",train_data['id'].count())
print(train_data.head())
############################tttt###################################################
# dat_edge = pd.read_csv("../feature/train_valid_test_dat_edge_feature.txt",delimiter='\t',header=None,
#                           names=['from_id','to_id','year','month','cnum','weight'])
#
# dat_symbol_feature = pd.read_csv("../open_data/dat_symbol_feature.txt",delimiter='\t')
# dat_edge['to_id'] = dat_edge['to_id'].astype('int')
# dat_symbol_feature = dat_symbol_feature.rename(columns={'id':'to_id'})
# dat_edge = pd.merge(dat_edge,dat_symbol_feature,how='left',on='to_id')
# weight = False
# if weight :
#     for x in dat_edge.columns :
#         if x.startswith('symbol_'):
#             dat_edge[x] = dat_edge[x]*dat_edge['weight']/dat_edge['cnum']
# print(train_dat_edge['to_id'].count())
# dat_edge = dat_edge.fillna(0)
# print(train_dat_edge.head())
# colums = [x for x in dat_edge.columns if x not in ('to_id','year','month','cnum','weight')]
# tmp_data = dat_edge[colums]
# del dat_edge
# tmp_data = tmp_data.groupby('from_id').sum().astype('int').reset_index()
# print(tmp_data.head())
# tmp_data = tmp_data.rename(columns={'from_id':'id'})
# tmp_data.to_csv("../feature/train_valid_test_dat_edge_symbol_mean.txt",index=False)
# ////////
# dat_edge = pd.read_csv("../feature/train_valid_test_dat_edge_feature.txt",delimiter='\t',header=None,
#                           names=['from_id','to_id','year','month','cnum','weight'])
# dat_edge = dat_edge[['from_id','to_id','year','month']]
# dat_edge['ym'] = dat_edge['year'].astype('str')+dat_edge['month'].astype('str')
# dat_edge = dat_edge[['from_id','to_id','ym']]
# dat_edge['to_id'] = dat_edge['to_id'].astype('int')
# dat_edge = dat_edge.groupby(['from_id','ym']).count().astype('int').reset_index()
# dat_edge = dat_edge.rename(columns={'from_id':'id','to_id':'edge_toid_count'})
# dat_edge1 = dat_edge[dat_edge['ym']=='201711']
# print(dat_edge1.head())
# train_data = pd.merge(train_data,dat_edge1,how='left',on='id')
# dat_edge2 = dat_edge[dat_edge['ym']=='201712']
# print(dat_edge2.head())
# train_data = pd.merge(train_data,dat_edge2,how='left',on='id')
# dat_edge3 = dat_edge[dat_edge['ym']=='20181']
# print(dat_edge3.head())
# train_data = pd.merge(train_data,dat_edge3,how='left',on='id')
# train_data = train_data.fillna(0)
# train_data = train_data[[x for x in train_data.columns if not x.startswith('ym')]]
# //////
# dat_edge = dat_edge[['from_id','to_id']]
# dat_edge['to_id'] = dat_edge['to_id'].astype('int')
# dat_edge = dat_edge.groupby(['from_id']).count().astype('int').reset_index()
# dat_edge = dat_edge.rename(columns={'from_id':'id','to_id':'edge_toid_count'})
# train_data = pd.merge(train_data,dat_edge,how='left',on='id')
# train_data = train_data.fillna(0)
# ////////
edge_symbol = pd.read_csv("../feature/train_valid_test_dat_edge_symbol_mean.txt")
# edge_symbol = edge_symbol[[x for x in edge_symbol.columns if x not in
#                            ('symbol_1','symbol_2','symbol_3','symbol_4','symbol_5','symbol_6'
                            # 'symbol_7','symbol_8','symbol_9','symbol_10','symbol_11','symbol_12','symbol_13',
#                             'symbol_14','symbol_15','symbol_16','symbol_17','symbol_18','symbol_19','symbol_20',
#                             'symbol_21','symbol_22','symbol_23','symbol_24')]]
train_data = pd.merge(train_data,edge_symbol,how='left',on='id')
del edge_symbol
print("all row3:",train_data['id'].count())
print(train_data.head())
# cf_data = pd.read_csv("../feature/cf_clustering_100.txt")
# cf_data = cf_data[[x for x in cf_data.columns if not x.startswith('feature')]]
# train_data = pd.merge(train_data,cf_data,how='left',on='id')
cf_data = pd.read_csv("../feature/cf_clustering_30.txt")
cf_data = cf_data[[x for x in cf_data.columns if not x.startswith('feature')]]
train_data = pd.merge(train_data,cf_data,how='left',on='id')
cf_data = pd.read_csv("../feature/cf_clustering_app30.txt")
cf_data = cf_data[[x for x in cf_data.columns if not x.startswith('feature')]]
train_data = pd.merge(train_data,cf_data,how='left',on='id')
data = pd.read_csv("../feature/train_valid_test_dat_app.csv")
data = data.dropna()
data['app_id'] = data['app_id'].astype('int')
tdata = data[['id','app_id']].groupby('id').count().astype('int').reset_index()
tdata = tdata.rename(columns={'app_id':'apps_count'})
train_data = pd.merge(train_data,tdata,how='left',on='id')

tdata = data[['id','app_id']].groupby('id').sum().astype('int').reset_index()
tdata = tdata.rename(columns={'app_id':'apps_sum'})
train_data = pd.merge(train_data,tdata,how='left',on='id')
train_data['id_num'] = train_data['id'].apply(np.log10)
print("all row4:",train_data['id'].count())
############################################################


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
        'scale_pos_weight': 0.2,
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
    clf = xgb.train(params[1],xgb_train,num_boost_round=10000,evals=watchlist,early_stopping_rounds=50)
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
sub_file = '../result2/submission_5x-XGB' + score + "_seed_" + str(1024) + '.txt'
sub_t_file = '../result2/submission_t_5x-XGB' + score + "_seed_" + str(1024) + '.txt'
res.to_csv(sub_file, index=False)
res_t.to_csv(sub_t_file,index=False)

# In[ ]:

folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(train_data, y)]
# display_importances(feature_importance_df_=feature_importance_df)

# In[ ]:

display_roc_curve(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)







#
# X_train,X_test,Y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1024)
# print(X_train.columns)
# # X_train['label'] = Y_train
# # tmp = X_train[X_train['label']==1]
# # X_train = X_train.append(tmp)
# # X_train = X_train.append(tmp)
# # X_train = X_train.append(tmp)
# # Y_train = X_train['label']
# # X_train = X_train[[x for x in X_train.columns if x !='label']]
#
#
#
# lgb_train = lgb.Dataset(X_train,Y_train)
# del X_train, Y_train,
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# params = [
#     {'boosting_type': 'gbdt',
#      'objective': 'binary',
#      'metric': {'auc'},
#      'learning_rate': 0.01,
#      'max_depth': 8,
#      'num_leaves': 35,
#      # 'feature_fraction': 0.125681745820782,
#      # 'bagging_freq': 5,
#      # 'min_split_gain': 0.0970905919552776,
#      # 'min_child_weight': 9.42012323936088,
#      'reg_alpha': 4,
#      'reg_lambda': 4,
#      # 'verbose': 1},
#      },
# ]
# clf = lgb.train(params[0],
#                 lgb_train,
#                 num_boost_round=1000,
#                 valid_sets=[lgb_eval],
#                 early_stopping_rounds=200,
#                 verbose_eval=100)
# feature_importances=sorted(zip(X_test.columns,clf.feature_importance()),key=lambda x:x[1])
# print(feature_importances)
#
# res = valid_data[['id']]
# res['prob']=clf.predict(valid_X, num_iteration=clf.best_iteration)
# res.to_csv("../open_data/res4.txt",index=False)
# # oof_preds[val_idx] = clf.predict(X_test, num_iteration=clf.best_iteration)
#
#
#
