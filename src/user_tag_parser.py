import pyspark

conf = pyspark.SparkConf().setAppName('tagcount').setMaster('local[*]')
sc = pyspark.SparkContext(conf=conf)

q_filename = '../data/questions_test.csv'
qt_filename = '../data/question_tags_test.csv'

# Retrieving questions data
rdd1 = sc.textFile(q_filename).map(lambda line: line.split(','))
header1 = rdd1.first()
rdd1 = rdd1.filter(lambda row: row != header1).toDF(header1)
# Retrieving questions_tags data
rdd2 = sc.textFile(qt_filename).map(lambda line: line.split(','))
header2 = rdd2.first()
rdd2 = rdd2.filter(lambda row: row != header2).toDF(header2)

# Join the two CSVs
keep = [rdd1.OwnerUserId, rdd2.Tag]
join_result = rdd1.join(rdd2, rdd1.Id == rdd2.Id).select(*keep)

# Order by user ID and count the number of tags per user
id_tag_table = join_result.orderBy(rdd1.OwnerUserId)
id_tag_table.show()
id_tag_count = id_tag_table.groupBy(rdd1.OwnerUserId, rdd2.Tag).count()
id_tag_count.show()

# Save data into a single file
id_tag_count.coalesce(1).write.format('com.databricks.spark.csv').save('../data/tagcount')
