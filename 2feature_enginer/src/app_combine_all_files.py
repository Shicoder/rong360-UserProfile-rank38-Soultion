# coding=utf-8
import os
import pandas as pd

# 将app数据合并到一个文件中
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