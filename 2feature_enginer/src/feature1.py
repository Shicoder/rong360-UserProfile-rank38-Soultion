# coding=utf-8
import pandas as pd
import sys
print sys.getdefaultencoding()
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
#
# dat_symbol = pd.read_csv("../open_data/dat_symbol.txt",delimiter='\t')
# print("all row:",len(dat_symbol['id'].unique()))
# #('all row:', 2266640)
# dat_symbol["new_class"] = dat_symbol['symbol'].apply(lambda x: x.split(",")[1] if len(x.split(","))>1 else None)
# print(dat_symbol.head())
# print("mutil class :",dat_symbol[dat_symbol['new_class'].notna()]['id'].count())
# # ('mutil class :', 483346)
# dat_symbol["new_class2"] = dat_symbol['symbol'].apply(lambda x: x.split(",")[2] if len(x.split(","))>2 else None)
# print(dat_symbol[dat_symbol['new_class2']!=None].head())
# dat_symbol["max_class"] = dat_symbol['symbol'].apply(lambda x: len(x.split(",")) if len(x.split(",")) else None)
# print("max:",dat_symbol['max_class'].max())
# # ('max:', 9)

train_data = pd.read_csv("../open_data/sample_train.txt",delimiter='\t',encoding='utf-8')
valid_data = pd.read_csv("../open_data/valid_id.txt")
valid_data['label'] = -1
train_data = train_data.append(valid_data)
print(train_data.head())
print(train_data['label'].count())
# print(train_data[train_data['label']==1]['label'].count())
# print(train_data[train_data['label']==0]['label'].count())
# dat_symbol = pd.read_csv("../open_data/dat_symbol_feature.txt",delimiter='\t')
train_symbol = pd.read_csv("../open_data/train_symbol_edge.txt",delimiter='\t')
valid_symbol = pd.read_csv("../open_data/valid_symbol_edge.txt",delimiter='\t')
dat_symbol = train_symbol.append(valid_symbol)
dat_symbol = dat_symbol[[x for x in dat_symbol.columns if x !='to_id']]
dat_symbol_c = [x for x in dat_symbol.columns if x.startswith('symbol')]
dat_symbol = dat_symbol.fillna(0)
dat_symbol[dat_symbol_c] =dat_symbol[dat_symbol_c].astype('int')
dat_symbol = dat_symbol.drop_duplicates()
dat_symbol = dat_symbol.rename(columns={'from_id':'id'})
print(dat_symbol.head())
#
dat_risk = pd.read_csv("../open_data/dat_risk.txt",delimiter='\t')
print("dat_risk:",dat_risk['id'].count())
# print(dat_risk[dat_risk['id']==10981148])
train_data = pd.merge(train_data,dat_risk,how='left',on='id')
print(train_data[train_data['a_cnt'].isnull()]['id'].count())
train_data = train_data.fillna(0)
print(train_data[train_data['a_cnt'].isnull()]['id'].count())
train_data['count_cnt'] = train_data['a_cnt']+train_data['b_cnt']+\
                          train_data['c_cnt']+train_data['d_cnt']+train_data['e_cnt']


train_data = pd.merge(train_data,dat_symbol,how='left',on='id')
dat_symbol = pd.read_csv("../open_data/dat_symbol_feature.txt",delimiter='\t')
# dat_symbol = dat_symbol.rename(columns={})
train_data = pd.merge(train_data,dat_symbol,how='left',on='id')
train_data = train_data.fillna(0)
print(train_data.head())
train_data = train_data.fillna(0)
print("sss",train_data.count())
train_data['count_class1'] = 0
for i in range(1,24):
    train_data['count_class1'] = train_data['count_class1']+train_data['symbol_'+str(i)+'_y']
import lightgbm as lgb
feature = [x for x in train_data.columns if x not in ('label','id')]
valid_data = train_data[train_data['label']==-1]
train_data = train_data[train_data['label']!=-1]
valid_X = valid_data[feature]
valid_Y = valid_data['label']
X = train_data[feature]
y = train_data['label']
X_train,X_test,Y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1024)
lgb_train = lgb.Dataset(X_train,Y_train)
del X_train, Y_train,
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = [
    {'boosting_type': 'gbdt',
     'objective': 'binary',
     'metric': {'auc'},
     'learning_rate': 0.01,
     'max_depth': 8,
     'num_leaves': 200,
     # 'feature_fraction': 0.125681745820782,
     # 'bagging_freq': 5,
     # 'min_split_gain': 0.0970905919552776,
     # 'min_child_weight': 9.42012323936088,
     'reg_alpha': 4,
     'reg_lambda': 4,
     # 'verbose': 1},
     },
]
clf = lgb.train(params[0],
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_eval],
                early_stopping_rounds=200,
                verbose_eval=100)
feature_importances=sorted(zip(X_test.columns,clf.feature_importance()),key=lambda x:x[1])
print(feature_importances)

res = valid_data[['id']]
res['prob']=clf.predict(valid_X, num_iteration=clf.best_iteration)
res.to_csv("../open_data/res3.txt",index=False)
# oof_preds[val_idx] = clf.predict(X_test, num_iteration=clf.best_iteration)
