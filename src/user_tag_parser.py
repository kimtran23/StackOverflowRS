from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import countDistinct, count, rank, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
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
keep1 = [rdd1.OwnerUserId, rdd2.Tag, rdd1.Id]
cond1 = [rdd1.Id == rdd2.Id, rdd1.OwnerUserId != 'NA']
join_result = rdd1.join(rdd2, cond1).select(*keep1)

# Order by user ID
uid_tag_qid_table = join_result.orderBy(rdd1.OwnerUserId)
uid_tag_qid_table.show()

# Count the number of tags per user
id_tag_count = uid_tag_qid_table.select(rdd1.OwnerUserId, rdd2.Tag).groupBy(
    rdd1.OwnerUserId, rdd2.Tag).agg(count('*').alias('TagCount'))
id_tag_count.show()

# Count the total number of questions per user
id_question_count = uid_tag_qid_table.select(
    rdd1.OwnerUserId, rdd1.Id).groupBy(rdd1.OwnerUserId).agg(
    countDistinct(rdd1.Id).alias('QuestionCount'))
id_question_count.show()

# Join the 2 counts dataframes
keep2 = [id_tag_count.OwnerUserId, id_tag_count.Tag, id_tag_count.TagCount,
         id_question_count.QuestionCount]
cond2 = [id_tag_count.OwnerUserId == id_question_count.OwnerUserId]
join_counts = id_tag_count.join(id_question_count, cond2).select(*keep2)
join_counts_ordered = join_counts.orderBy(rdd1.OwnerUserId)
join_counts_ordered.show()

# Normalize the TagCount using QuestionCount
normalized_result = join_counts_ordered.withColumn(
    'FinalTagCount',
    join_counts_ordered.TagCount / join_counts_ordered.QuestionCount)
normalized_result.show()

# Converts OwnerUserId into an int
normalized_result = normalized_result.withColumn(
    'OwnerUserId', normalized_result['OwnerUserId'].cast('int').alias('UserId'))
print normalized_result

# Creates a new column that associates each tag to an index
indexer = StringIndexer(
    inputCol='Tag', outputCol='TagIndex').fit(normalized_result)
normalized_tag_index = indexer.transform(normalized_result)
normalized_tag_index.show()

keep3 = ['OwnerUserId', 'TagIndex', 'FinalTagCount']
final = normalized_tag_index.select(*keep3)
final.show()

# Save data into a single file
# final.coalesce(1).write.format(
#     'com.databricks.spark.csv').save('../data/tagcount')

# Create training and testing set
training, test = final.randomSplit([0.8, 0.2])

# Creates the recommendation model
als = ALS(userCol='OwnerUserId', itemCol='TagIndex', ratingCol='FinalTagCount')
model = als.fit(training)
predictions = model.transform(test)
# Remove any rows containing null/NaN values
predictions = predictions.na.drop()

print 'Predictions (' + str(predictions.count()) + '):'
print predictions.toDF('OwnerUserId', 'TagIndex', 'FinalTagCount', 'prediction').show()

# Evaluate the RMSE
evaluator = RegressionEvaluator(metricName='rmse', labelCol='FinalTagCount',
                                predictionCol='prediction')
rmse = evaluator.evaluate(predictions)
print 'RMSE: ' + str(rmse)

# Get top 2 predictions
predictions_df = predictions.toDF('OwnerUserId', 'TagIndex', 'FinalTagCount', 'prediction')
predictions_df.show()
window = Window.partitionBy(predictions_df['OwnerUserId']).orderBy(predictions_df['prediction'].desc())
top_predictions = predictions_df.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 2)
top_predictions.show()

# Test if top predictions work
new_user = [
    [100, 29.0, 1.0],
    [100, 3.0, 0.5],
    [100, 58.0, 0.5],
    [100, 30.0, 0.5]
]
new_user_rdd = sc.parallelize(new_user)
final_with_new = final.rdd.union(new_user_rdd)
new_model = als.fit(final_with_new)
new_user_df = new_user_rdd.toDF(['OwnerUserId', 'TagIndex','FinalTagCount'])
new_predictions = new_model.transform(new_user_df)
new_predictions_df = new_predictions.toDF('OwnerUserId', 'TagIndex', 'FinalTagCount', 'prediction')

new_window = Window.partitionBy(new_predictions_df['OwnerUserId']).orderBy(new_predictions_df['prediction'].desc())
new_user_rec = new_user_df.select('*', rank().over(new_window).alias('rank')).filter(col('rank') <= 2)
new_user_rec.show()

# Get the tag names to put into the predictions
normalized_tag_index.show()
top_predictions = predictions_df.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 2)
tag_name_index = normalized_tag_index.select(*['TagIndex', 'Tag'])
rec_tags = top_predictions.join(tag_name_index, top_predictions.TagIndex == tag_name_index.TagIndex)
rec_tags.show()

# Creates the recommendation model
# model = ALS.train(final, 5)
# testdata = final.rdd.map(lambda p: (p[0], p[1]))
# predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
# print predictions.collect()
# ratesAndPreds = final.rdd.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
# MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
# RMSE = math.sqrt(MSE)
# print RMSE
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
