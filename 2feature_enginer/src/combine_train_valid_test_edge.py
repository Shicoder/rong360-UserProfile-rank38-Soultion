# coding=utf-8
import numpy as np
files = ["../open_data/dat_edge/train_dat_edge.txt","../open_data/dat_edge/valid_dat_edge.txt","../open_data/dat_edge/test_dat_edge.txt"]
outfile = open("../feature/train_valid_test_dat_edge_feature.txt",'w')
for file in files:
    infile = open(file, 'r')
    for line in infile:
        line = line.strip()
        from_id = line.split('\t')[0]
        data = set()
        if from_id == 'from_id':
            # line = 'from_id\tto_id\tyear\tmonth\tcnum\tweight\n'
            # outfile.write(line)
            continue
        if len(line.split('\t'))<=1:
            continue
        infos = line.split('\t')[2]
        for info in  infos.split(','):
            year = info.split('-')[0]
            month = info.split(':')[0].split('-')[1]
            cnum = info.split(':')[1].split('_')[0]
            weight = info.split('_')[1]
            from_id = str(int(line.split('\t')[0]))
            to_id = str(int(float(line.split('\t')[1])))
            line = from_id+'\t'+to_id+'\t'+year+'\t'+month+'\t'+cnum+'\t'+weight+'\n'
            outfile.write(line)
    # outfile.close()
    infile.close()
outfile.close()