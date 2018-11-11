# coding=utf-8
# !/usr/bin/env python
import pandas as pd
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.types import StructType
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import KMeans, BisectingKMeans

from pyspark import SparkContext, HiveContext, SQLContext, SparkConf

spark_conf = SparkConf(). \
        setAppName("test")
    # sc = SparkContext(u'local[4]',appName=job_name)
sc = SparkContext("local[4]", conf=spark_conf)
ssc = SQLContext(sc)
######################################
# data = pd.read_csv("../feature/train_valid_test_edge_cf_feature.txt")
# data['ft_mean_cnum_weight'] = data['ft_mean_cnum_weight'].apply(np.log10)
######################################
data = pd.read_csv("../feature/train_valid_test_dat_app.csv")
data = data.dropna()
data['app_id'] = data['app_id'].astype('int')
print(data.head())
#####################################
# print(data.head())
data = ssc.createDataFrame(data)
# parts = lines.map()
# data = ssc.read.schema()
# train_data,test_data = data.randomSplit([0.1,0.9])
train_data = data
####################################
# als = ALS(maxIter=5,regParam=0.01,userCol='from_id',itemCol='to_id',ratingCol='ft_mean_cnum_weight',
#           coldStartStrategy='drop')
###################################
als = ALS(maxIter=5,regParam=0.01,userCol='id',itemCol='app_id',ratingCol='score',
          coldStartStrategy='drop')
###################################
model = als.fit(train_data)
# userFeaure = model.userFactors
# userFeaure.show()

mean=udf(lambda x: float(np.mean(x)),FloatType())
array2Vec=udf(lambda x :Vectors.dense(x), VectorUDT())
ss = model.userFactors
get_prob1 = udf(lambda x: float(x[0]))
get_prob2 = udf(lambda x: float(x[1]))
get_prob3 = udf(lambda x: float(x[2]))
get_prob4 = udf(lambda x: float(x[3]))
get_prob5 = udf(lambda x: float(x[4]))
get_prob6 = udf(lambda x: float(x[5]))
get_prob7 = udf(lambda x: float(x[6]))
get_prob8 = udf(lambda x: float(x[7]))
get_prob9 = udf(lambda x: float(x[8]))
get_prob10 = udf(lambda x: float(x[9]))
ss = ss.select('*', get_prob1(ss['features']).alias('feature1').cast(FloatType()))
ss = ss.select('*', get_prob2(ss['features']).alias('feature2').cast(FloatType()))
ss = ss.select('*', get_prob3(ss['features']).alias('feature3').cast(FloatType()))
ss = ss.select('*', get_prob4(ss['features']).alias('feature4').cast(FloatType()))
ss = ss.select('*', get_prob5(ss['features']).alias('feature5').cast(FloatType()))
ss = ss.select('*', get_prob6(ss['features']).alias('feature6').cast(FloatType()))
ss = ss.select('*', get_prob7(ss['features']).alias('feature7').cast(FloatType()))
ss = ss.select('*', get_prob8(ss['features']).alias('feature8').cast(FloatType()))
ss = ss.select('*', get_prob9(ss['features']).alias('feature9').cast(FloatType()))
ss = ss.select('*', get_prob10(ss['features']).alias('feature10').cast(FloatType()))
ss = ss.drop('features')
ss.printSchema()
ss = ss.fillna(0)
# ss =ss.fillna(0)
ss = ss.toPandas()
print(ss.head())
# ss.to_csv("../feature/cf_clustering_20.txt",index=False)
print("saddddddd")
userFactors=model.userFactors.withColumn("features",array2Vec("features"))#
userFactors.show()

############################################
# data = pd.read_csv("../feature/cf_clustering_20.txt")
data = ss
y = data['id']
x = data[[x for x in data.columns if x != 'id']]
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=30,max_iter=200)
kmeans = kmeans.fit(x)

lables = kmeans.predict(x)
data['culster30'] = lables
print("##################")
print(data.head())
print(kmeans.cluster_centers_)

# data = data[[x for x in data.columns if x in ('id','lable')]]
data.to_csv("../feature/cf_clustering_app30.txt",index=False)


# kmeans = KMeans(n_clusters=100,max_iter=300)
# kmeans = kmeans.fit(x)
#
# lables = kmeans.predict(x)
# data['culster100'] = lables
# print("##################")
# print(data.head())
# print(kmeans.cluster_centers_)
#
# # data = data[[x for x in data.columns if x in ('id','lable')]]
# data.to_csv("../feature/cf_clustering_app100.txt",index=False)


# userFactors = userFactors.withColumn("f_mean",mean("features"))
#
# Kmeans = KMeans().setK(20).setSeed(1)
# model = Kmeans.fit(userFactors)
# centers = model.clusterCenters()
# wssse = model.computeCost(userFactors)
# print("Within Set Sum of Squared Errors = " + str(wssse))
# userFactors = model.transform(userFactors)
#
# bkm = BisectingKMeans().setK(20).setSeed(1024)
# model = bkm.fit(userFactors)
# cost = model.computeCost(userFactors)
# print("Within Set Sum of Squared Errors = " + str(cost))#2181.4
#
# centers=[[i,center] for i,center in enumerate(centers)]
# #we have to parallelize local variable to transfer it to RDD
# centers = sc.parallelize(centers)
# centers=centers.map(lambda x:Row(prediction=x[0],center=Vectors.dense(x[1])))
# centers=ssc.createDataFrame(centers)
# c_result = centers.toPandas()
# c_result.to_csv("../feature/cf_clustering_20.txt",index=False)
#
# # pandas_res = result.toPandas()
# # pandas_res.to_csv("./data/predict_result.csv")
#
# # predictions  =model.transform(test_data)
# # evaluator = RegressionEvaluator(metricName='rmse',labelCol='ft_mean_cnum_weight',predictionCol='prediction')
# # mse = evaluator.evaluate(predictions)
# # print("root_mean_square error ="+str(mse))
