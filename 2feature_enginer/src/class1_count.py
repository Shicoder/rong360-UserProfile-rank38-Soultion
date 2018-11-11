# coding=utf-8

infile = open("../open_data/dat_app/dat_app_1",'r')
class1 = set()

for line in infile:
    line = line.strip()
    symbol = line.split('\t')[1].split(',')
    for sy in symbol:
        if sy == 'symbol':
            continue
        if sy not in class1:
            class1.add(sy)
            print(sy)
        else:
            continue
infile.close()
print(len(class1))