import pandas as pd

test_id = pd.read_csv("../open_data/test_id.txt")
valid_id = pd.read_csv("../open_data/valid_id.txt")
# print(test_id.head())
# print(test_id['id'].unique())
jiaoji =  set(test_id['id'].unique()).intersection(set(valid_id['id'].unique()))
print(jiaoji)