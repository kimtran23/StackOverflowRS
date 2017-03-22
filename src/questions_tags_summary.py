import csv

csv_filename = '../data/question_tags.csv'
test_filename = '../data/question_tags_test.csv'

with open(csv_filename, 'rb') as csvFile, open(test_filename, 'wb') as testFile:
    datareader = csv.reader(csvFile)
    datawriter = csv.writer(testFile, delimiter=',')
    count = 0

    for row in datareader:
        if count < 101:
            count += 1
            question_id = row[0]
            tag = row[1]

            r = [question_id, tag]
            datawriter.writerow(r)
        else:
            break
