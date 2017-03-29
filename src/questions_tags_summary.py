import csv

size = 500000
csv_filename = '../data/question_tags.csv'
test_filename = '../data/question_tags_' + str(size) + '_test.csv'

with open(csv_filename, 'rb') as csvFile, open(test_filename, 'wb') as testFile:
    datareader = csv.reader(csvFile)
    datawriter = csv.writer(testFile, delimiter=',')
    count = 0

    for row in datareader:
        if count <= size:
            count += 1
            question_id = row[0]
            tag = row[1]

            r = [question_id, tag]
            datawriter.writerow(r)
        else:
            break
