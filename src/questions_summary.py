import csv

size = 500000
csv_filename = '../data/questions.csv'
test_filename = '../data/questions_' + str(size) + '_test.csv'

with open(csv_filename, 'rb') as csvFile, open(test_filename, 'wb') as testFile:
    datareader = csv.reader(csvFile)
    datawriter = csv.writer(testFile, delimiter=',')
    count = 0

    for row in datareader:
        if count <= size:
            count += 1
            question_id = row[0]
            creation_date = row[1]
            closed_date = row[2]
            deletion_date = row[3]
            score = row[4]
            user_id = row[5]
            answer_count = row[6]

            r = [question_id, creation_date, closed_date,
                 deletion_date, score, user_id, answer_count]
            datawriter.writerow(r)
        else:
            break
