# coding=utf-8
import pandas as pd
import os.path

#combine all edge
# f_path = "../open_data/dat_edge"
# files = os.listdir(f_path)
# len_f = len(files)
# for i,f in enumerate(files):
#     if f == 'dat_edge_1':
#         file = pd.read_csv(f_path+'/'+f,delimiter='\t')
#
#     else:
#         file = pd.read_csv(f_path + '/' + f, delimiter='\t',header=None,names=['from_id','to_id','info'])
#         # print(file.head())
#     if i==0:
#         file.to_csv(f_path+'/all_dat_edge.txt',index=False,sep='\t')
#     else:
#         file.to_csv(f_path+'/all_dat_edge.txt',mode='a',header=None,index=False,sep='\t')


# edge = pd.read_csv("../open_data/dat_edge/all_dat_edge.txt",delimiter='\t')
# print(edge.count())

# train_data = pd.read_csv("../open_data/sample_train.txt",delimiter='\t')
# # train_data = train_data[['id']]
# valid_data = pd.read_csv("../open_data/valid_id.txt")
# valid_data['label'] = -1
# test_data = pd.read_csv("../open_data/test_id.txt")
# test_data['label'] = -2
#
# train_data = train_data.append(valid_data)
# train_data = train_data.append(test_data)
# edge = pd.read_csv("../open_data/dat_edge/all_dat_edge.txt",delimiter='\t')
# train_data = train_data.rename(columns={'id':'from_id'})
# train_data = pd.merge(train_data,edge,how='left',on='from_id')
# # train_data = train_data.rename(columns={'from_id':'id'})
# del edge
# valid_data = train_data[train_data['label']==-1]
# valid_data[['from_id','to_id','info']].to_csv("../open_data/dat_edge/valid_dat_edge.txt",index=False,sep='\t')
# del valid_data
# test_data = train_data[train_data['label']==-2]
# test_data[['from_id','to_id','info']].to_csv("../open_data/dat_edge/test_dat_edge.txt",index=False,sep='\t')
# del test_data
# train_data = train_data[train_data['label']>=0]
# train_data[['from_id','to_id','info']].to_csv("../open_data/dat_edge/train_dat_edge.txt",index=False,sep='\t')
# print(train_data.count())
##################################################################
# dat_symbol_feature = pd.read_csv("../open_data/dat_symbol_feature.txt",delimiter='\t')
# train_dat_edge = pd.read_csv("../open_data/dat_edge/train_dat_edge.txt",delimiter='\t')
# train_dat_edge = train_dat_edge.dropna()
# train_dat_edge['to_id'] = train_dat_edge['to_id'].astype('int')
# dat_symbol_feature = dat_symbol_feature.rename(columns={'id':'to_id'})
# train_dat_edge = pd.merge(train_dat_edge,dat_symbol_feature,how='left',on='to_id')
# # print(train_dat_edge['to_id'].count())
# train_dat_edge = train_dat_edge.fillna(0)
# # print(train_dat_edge.head())
# colums = [x for x in train_dat_edge.columns if x not in ('to_id','info')]
# tmp_data = train_dat_edge[colums]
# tmp_data = tmp_data.groupby('from_id').sum().astype('int').reset_index()
# train_id = pd.read_csv("../open_data/sample_train.txt",delimiter='\t')
# train_id = train_id[['id']]
# train_id = train_id.rename(columns={'id':'from_id'})
#
# train_id = pd.merge(train_id,tmp_data,how='left',on='from_id')
# columns = [x for x in train_id.columns if x != 'info']
# train_id[columns].to_csv("../open_data/train_symbol_edge.txt",sep='\t',index=False)
# print(train_dat_edge.head())
#
dat_symbol_feature = pd.read_csv("../open_data/dat_symbol_feature.txt",delimiter='\t')
valid_dat_edge = pd.read_csv("../open_data/dat_edge/valid_dat_edge.txt",delimiter='\t')
valid_dat_edge = valid_dat_edge.dropna()
valid_dat_edge['to_id'] = valid_dat_edge['to_id'].astype('int')
dat_symbol_feature = dat_symbol_feature.rename(columns={'id':'to_id'})
valid_dat_edge = pd.merge(valid_dat_edge,dat_symbol_feature,how='left',on='to_id')
# print(train_dat_edge['to_id'].count())
valid_dat_edge = valid_dat_edge.fillna(0)
# print(train_dat_edge.head())
colums = [x for x in valid_dat_edge.columns if x not in ('to_id','info')]
tmp_data = valid_dat_edge[colums]
tmp_data = tmp_data.groupby('from_id').sum().astype('int').reset_index()
valid_id = pd.read_csv("../open_data/valid_id.txt",delimiter='\t')
# train_id = train_id[['id']]
valid_id = valid_id.rename(columns={'id':'from_id'})
valid_id = pd.merge(valid_id,tmp_data,how='left',on='from_id')
columns = [x for x in valid_id.columns if x != 'info']
valid_id[columns].to_csv("../open_data/valid_symbol_edge.txt",sep='\t',index=False)

#
# ##################################
# # train_dat_edge = pd.read_csv("../open_data/dat_edge/train_dat_edge.txt",delimiter='\t')
# ##########
# # add info