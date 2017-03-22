import csv

csv_filename = '../data/question_tags.csv'
summary_filename = '../data/summary_tags.csv'

with open(csv_filename, 'rb') as csvFile, open(summary_filename, 'wb') as summaryFile:
    datareader = csv.reader(csvFile)
    datawriter = csv.writer(summaryFile, delimiter=',')
    question_id = '0'
    list_tags = []

    for row in datareader:
        if row[0] == 'Id' and row[1] == 'Tag':
            question_id = row[0]
            list_tags.append(row[1])
        else:
            if question_id != row[0]:
                print 'Question ID ' + question_id
                r = [question_id, str(list_tags)]
                datawriter.writerow(r)
                question_id = row[0]
                list_tags = []

            list_tags.append(row[1])

    r = [question_id, str(list_tags)]
    datawriter.writerow(r)
