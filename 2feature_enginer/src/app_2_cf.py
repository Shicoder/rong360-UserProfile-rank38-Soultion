# coding=utf-8
import os
import pandas as pd
# f_path = "../open_data/dat_app"
# files = os.listdir(f_path)
# len_f = len(files)
# outfile = open("../feature/all_data_dat_app.csv",'w')
# for i,f in enumerate(files):
#     print("file:",f)
#     file = open(f_path+'/'+f,'r')
#     for line in file:
#         line = line.strip()
#         id = line.split('\t')[0]
#         app_ids = line.split('\t')[1].split(',')
#         for app in app_ids:
#             new_line = id+'\t'+app+'\t'+str(1)
#             outfile.write(new_line+'\n')
#     file.close()
# outfile.close()

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
