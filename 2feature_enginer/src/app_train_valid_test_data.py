# coding=utf-8
import os
import pandas as pd
#从全部的app数据中只过滤出训练集，测试集，验证集中的id，减少计算量
data = pd.read_csv("../feature/all_data_dat_app.csv",delimiter='\t',names=['id','app_id','score'])
id = pd.read_csv("../open_data/sample_train.txt",delimiter='\t')
valid_data = pd.read_csv("../open_data/valid_id.txt")
test_data = pd.read_csv("../open_data/test_id.txt")
valid_data['label']= 1
id = id.append(valid_data)
test_data['label'] = 1
id = id.append(test_data)
del valid_data, test_data
print(id['id'].count())
data = pd.merge(id,data,how='left',on='id')
data = data[[x for x in data.columns if x !='label']]
data.to_csv("../feature/train_valid_test_dat_app.csv",index=False)
