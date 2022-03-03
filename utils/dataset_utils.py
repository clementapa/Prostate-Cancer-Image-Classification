import csv
import os

def parse_csv(path_dataset, split='train'):
    X = []
    y = []
    path_images = os.path.join(path_dataset, split, split)
    path_csv_file = os.path.join(path_dataset, split+'.csv')
    with open(path_csv_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(spamreader):
            if i > 0:
                X.append(os.path.join(path_images, row[0]+'.tiff'))
                y.append(int(row[-2]))
    return X, y
