# coding=utf-8
import pandas as pd
#测试
# data = pd.read_csv("../feature/train_valid_test_dat_app.csv")
# print(data['id'].count())
# data = data.dropna()
# data['app_id'] = data['app_id'].astype('int')
# data = data[['id','app_id']].groupby('id').sum().astype('int').reset_index()
# print(data.head())
train_data = pd.read_csv("../open_data/sample_train.txt",delimiter='\t',encoding='utf-8')

print(train_data['id'].describe())