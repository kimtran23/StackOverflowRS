from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import countDistinct, count, rank, col, avg
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import math


def recommend_questions(final, tag_name_index, rdd1, rdd2):
    # Getting the tags for every user
    users_tags = final.join(
        tag_name_index, final.TagIndex == tag_name_index.TagIndex,
        'leftouter').drop('TagIndex').orderBy(
        final.OwnerUserId).withColumnRenamed('OwnerUserId', 'User')
    users_tags.show()

    # Get the question IDs for all the users
    qid_users_tags = rdd2.join(
        users_tags, users_tags.Tag == rdd2.Tag,
        'left').drop(rdd2.Tag).dropna()
    qid_users_tags.show()

    # Count the number of tags each question is associated to
    qid_users_tags = qid_users_tags.groupBy(
        'User', 'Id').agg(count('Id').alias('RecTagCount'))
    qid_users_tags.show()

    # Get the creator for each question
    questions_owner = rdd1.select(*[rdd1.Id, rdd1.OwnerUserId])
    questions_owner = questions_owner.withColumn(
        'OwnerUserId', questions_owner['OwnerUserId'].cast('int'))
    questions_owner.show()
    qid_users_tags_owner = qid_users_tags.join(
        questions_owner, questions_owner.Id == qid_users_tags.Id,
        'left').drop(questions_owner.Id)
    qid_users_tags_owner.show()

    # The question is not recommended if it is created by the user
    qid_users_tags_owner = qid_users_tags_owner.rdd.filter(
        lambda x: x.User != x.OwnerUserId).toDF()
    qid_users_tags_owner.show()

    # Get the top questions that have the most tags recommended
    window = Window.partitionBy(qid_users_tags_owner['User']).orderBy(
        qid_users_tags_owner['RecTagCount'].desc())
    recommendations = qid_users_tags_owner.select(
        '*', rank().over(window).alias('Rank')).filter(
        col('Rank') <= 10).drop('Rank')

    print 'Recommended question IDs:'
    recommendations.show()

    return recommendations


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

# Finding expected questions to recommend for all users
expected_recommendations = recommend_questions(
    final, tag_name_index, rdd1, rdd2)

# Finding questions to recommend based on predictions for all users
formatted_predictions = predictions.drop('FinalTagCount').withColumnRenamed(
    'prediction', 'FinalTagCount')
final_with_predictions = training.union(formatted_predictions)
recommendations = recommend_questions(
    final_with_predictions, tag_name_index, rdd1, rdd2)

# Finding questions to recommend for specific user
user_id = str(predictions.select('OwnerUserId').first()[0])
print 'Predicting for user ' + user_id
# Expected recommendations for the specified user
expected_user_recommendations = expected_recommendations.filter(
    expected_recommendations.User == user_id)
print 'Expected recommendations for user ' + user_id + ':'
expected_user_recommendations.limit(10).show()

# Recommendations for specified user From predictions
user_recommendations = recommendations.filter(
    recommendations.User == user_id)
print 'Recommendations for user ' + user_id + ' from predictions:'
user_recommendations.limit(10).show()
