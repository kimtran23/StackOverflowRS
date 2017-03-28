from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RegressionMetrics
import math

conf = SparkConf().setAppName('tagcount').setMaster('local[*]')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

q_filename = '../data/questions_test.csv'
qt_filename = '../data/question_tags_test.csv'

# Retrieve questions data
rdd1 = sc.textFile(q_filename).map(lambda line: line.split(','))
header1 = rdd1.first()
rdd1 = rdd1.filter(lambda row: row != header1).toDF(header1)

# Retrieve questions_tags data
rdd2 = sc.textFile(qt_filename).map(lambda line: line.split(','))
header2 = rdd2.first()
rdd2 = rdd2.filter(lambda row: row != header2).toDF(header2)

# Join the two CSVs
keep1 = [rdd1.OwnerUserId, rdd2.Tag]
cond = [rdd1.Id == rdd2.Id, rdd1.OwnerUserId != 'NA']
join_result = rdd1.join(rdd2, cond).select(*keep1)

# Order by user ID
id_tag_table = join_result.orderBy(rdd1.OwnerUserId)
id_tag_table.show()

# Count the number of tags per user
id_tag_count = id_tag_table.groupBy(rdd1.OwnerUserId, rdd2.Tag).count()
id_tag_count.show()

# Converts OwnerUserId into an int
id_tag_count = id_tag_count.withColumn("OwnerUserId", id_tag_count[
    "OwnerUserId"].cast("int").alias("OwnerUserId"))

# Creates a new column that associates each tag to an index,
# which becomes the training set
indexer = StringIndexer(inputCol="Tag", outputCol="TagIndex").fit(id_tag_count)
df_index = indexer.transform(id_tag_count)
df_index.show()

keep2 = ['OwnerUserId', 'TagIndex', 'count']
final = df_index.select(*keep2)
final.show()

# Save data into a single file
# id_tag_count.coalesce(1).write.format(
#     'com.databricks.spark.csv').save('../data/tagcount')

# Create training and testing set
# training, test = final.randomSplit([0.8, 0.2])

# Creates the recommendation model
model = ALS.train(final, 5)
testdata = final.rdd.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
print predictions.collect()
ratesAndPreds = final.rdd.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
RMSE = math.sqrt(MSE)
print RMSE
# metrics = RegressionMetrics(ratesAndPreds)
# rmse = metrics.rootMeanSquaredError
# print rmse
# als = ALS(userCol="OwnerUserId", itemCol="TagIndex", ratingCol="count")
# model = als.fit(training)

# Create our own testing set
# test1 = [1, 'c#', 2, 29.0]
# test2 = [1, 'datetime', 2, 29.0]
# test3 = [1, 'datediff', 1, 58.0]
# test4 = [1, 'relative-time-span', 1, 30.0]
# test_set = [test1, test2, test3, test4]
# test = sc.parallelize(test_set).toDF(
#     ['OwnerUserId', 'Tag', 'count', 'TagIndex'])

# Evaluate RMSE on the test data
# keep3 = ['OwnerUserId', 'TagIndex']
# test = final.select(*keep3)
# test.show()
# predictions = model.predictAll(test.rdd)
# print predictions.collect()
# predictions = model.transform(test)


# evaluator = RegressionEvaluator(
#     metricName="rmse", labelCol="count", predictionCol="prediction")
# rmse = evaluator.evaluate(predictions.toDF())
# print rmse
