from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import countDistinct, count, rank, col, avg
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import math

conf = SparkConf().setAppName('tagcount').setMaster('local[*]')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

q_filename = '../data/questions_5000_test.csv'
qt_filename = '../data/question_tags_5000_test.csv'

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

# Calculate the average of each UserId
tag_avg = id_tag_count.select(rdd1.OwnerUserId, id_tag_count.TagCount).groupBy(
    rdd1.OwnerUserId).agg(avg(id_tag_count.TagCount).alias('TagAvg'))

# Join the tag count dataframe with the average
keep2 = [id_tag_count.OwnerUserId, id_tag_count.Tag, id_tag_count.TagCount,
         tag_avg.TagAvg]
cond2 = [id_tag_count.OwnerUserId == tag_avg.OwnerUserId]
join_counts = id_tag_count.join(tag_avg, cond2).select(*keep2)
join_counts_ordered = join_counts.orderBy(rdd1.OwnerUserId)
join_counts_ordered.show()

# Normalize the TagCount using TagAvg
normalized_result = join_counts_ordered.withColumn(
    'FinalTagCount', join_counts_ordered.TagCount - join_counts_ordered.TagAvg)
normalized_result.show()

# Converts OwnerUserId into an int
normalized_result = normalized_result.withColumn(
    'OwnerUserId',
    normalized_result['OwnerUserId'].cast('int').alias('UserId'))

# Creates a new column that associates each tag to an index
indexer = StringIndexer(
    inputCol='Tag', outputCol='TagIndex').fit(normalized_result)
normalized_tag_index = indexer.transform(normalized_result)
normalized_tag_index.show()

# Only keep values we need for the recommendation model
keep3 = ['OwnerUserId', 'TagIndex', 'FinalTagCount']
final = normalized_tag_index.select(*keep3)
final.show()

# Create training and testing set
training, test = final.randomSplit([0.6, 0.4])

print 'Training size: ' + str(training.count())
print 'Test size: ' + str(test.count())
print 'Total size: ' + str(final.count())

# Creates the recommendation model
als = ALS(userCol='OwnerUserId',
          itemCol='TagIndex', ratingCol='FinalTagCount')
model = als.fit(training)
predictions = model.transform(test)

# Remove any rows containing null/NaN values, caused by the test set
# having tags that were not in the training set
predictions = predictions.na.drop()

print 'Predictions (' + str(predictions.count()) + '):'
predictions_df = predictions.toDF(
    'OwnerUserId', 'TagIndex', 'FinalTagCount', 'Prediction')
predictions_df.show()

# Evaluate the recommendation model using RMSE
evaluator = RegressionEvaluator(metricName='rmse', labelCol='FinalTagCount',
                                predictionCol='prediction')
rmse = evaluator.evaluate(predictions)
print 'RMSE: ' + str(rmse)

# Get the tags' name and index number
tag_name_index = normalized_tag_index.select(*['TagIndex', 'Tag']).distinct()

# Get the corresponding recommended tags for each prediction
rec_tags = predictions_df.join(
    tag_name_index, predictions_df.TagIndex == tag_name_index.TagIndex,
    'leftouter').drop('TagIndex')
rec_tags.show()

# Get the questions that have the recommended tags
# and count the number of recommended tags each question is associated to
rec_tags_id = rec_tags.join(rdd2, rec_tags.Tag == rdd2.Tag, 'left')
rec_tags_id = rec_tags_id.groupBy(
    'OwnerUserId', 'Id').agg(count('Id').alias('RecTagCount'))
rec_tags_id.show()

# Get the top questions that have the most tags recommended
window = Window.partitionBy(rec_tags_id['OwnerUserId']).orderBy(
    rec_tags_id['RecTagCount'].desc())
recommendations = rec_tags_id.select(
    '*', rank().over(window).alias('Rank')).filter(
    col('Rank') <= 2).drop('Rank')
print 'Recommended question IDs: :'
recommendations.show()


# Creating test user to see how the above code works on predictable data
# Create test user that has similar tags as user 1
new_user = [
    [100, 29.0, 1.0, 0.9],
    [100, 3.0, 0.5, 0.4],
    [100, 58.0, 0.5, 0.6],
    [100, 30.0, 0.5, 0.3]
]
new_user_rdd = sc.parallelize(new_user)
new_model = als.fit(final)
new_user_df = new_user_rdd.toDF(
    ['OwnerUserId', 'TagIndex', 'FinalTagCount', 'Prediction'])
new_user_df.show()

# Get the corresponding tags for each prediction
rec_tags = new_user_df.join(
    tag_name_index, new_user_df.TagIndex == tag_name_index.TagIndex,
    'leftouter').drop('TagIndex')
rec_tags.show()

# Get the questions that have the recommended tags
# and count the number of recommended tags each question is associated to
rec_tags_id = rec_tags.join(rdd2, rec_tags.Tag == rdd2.Tag, 'left')
rec_tags_id = rec_tags_id.groupBy(
    'OwnerUserId', 'Id').agg(count('Id').alias('RecTagCount'))
rec_tags_id.show()

# Get the top questions (has the most tags recommended)
window = Window.partitionBy(rec_tags_id['OwnerUserId']).orderBy(
    rec_tags_id['RecTagCount'].desc())
print 'Recommended question IDs: '
rec_tags_id.select('*', rank().over(window).alias('Rank')
                   ).filter(col('Rank') <= 2).drop('Rank').show()
