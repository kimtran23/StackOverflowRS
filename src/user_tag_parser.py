import csv
from ast import literal_eval

csv_filename = '../data/sorted_users.csv'
user_tag_filename = '../data/user_tag.csv'

with open(csv_filename, 'rb') as csvFile, open(user_tag_filename, 'wb') as outputFile:
    datareader = csv.reader(csvFile)
    datawriter = csv.writer(outputFile)

    for row in datareader:
        if row[5] != 'NA':
            currentUser = row[5]
            list_tags = literal_eval(row[7])

            for tag in list_tags:
                print 'Writing tag for user ' + currentUser
                r = [currentUser, tag]
                datawriter.writerow(r)
