# coding=utf-8
import pandas as pd

# edge_data = pd.read_csv("../open_data/dat_edge/all_dat_edge.txt",delimiter='\t')
# # print(edge_data['from_id'].count())
# edge_data  = edge_data[['from_id','to_id']]
# edge_data = edge_data.drop_duplicates()
# edge_data = edge_data.dropna()
# # print(edge_data['from_id'].count())
# edge_data.to_csv("../open_data/dat_edge/all_dat_edge_dropduplicate.txt",index=False)

dat_edge = pd.read_csv("../feature/train_valid_test_dat_edge_feature.txt",delimiter='\t',header=None,
                          names=['from_id','to_id','year','month','cnum','weight'])
# dat_edge['year_month'] = dat_edge['year'].astype('str')+dat_edge['month'].astype('str')
dat_edge = dat_edge[['from_id','to_id','cnum','weight']]
dat_edge = dat_edge.drop_duplicates()
t1 = dat_edge[['from_id','to_id','cnum','weight']].groupby(['from_id','to_id']).sum().reset_index()
t1 = t1.rename(columns={'cnum':'ft_cnum','weight':'ft_weight'})
t1['ft_mean_cnum_weight'] = t1['ft_weight'].astype('float')/t1['ft_cnum']

dat_edge = pd.merge(dat_edge,t1,how='left',on=['from_id','to_id'])
dat_edge = dat_edge[['from_id','to_id','ft_mean_cnum_weight']]
print(dat_edge['ft_mean_cnum_weight'].count())
dat_edge.to_csv("../feature/train_valid_test_edge_cf_feature.txt",index=False)
