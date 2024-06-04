import json
import copy
import csv
from collections import OrderedDict
import argparse
import os

total_score = {}
max_score = {}
min_score = {}
review = []


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--file', default='fig1')
args = parser.parse_args()

file = args.file

with open(file, 'r') as f:
    for line in f:
        review.append(json.loads(line))


domain_dict = {}
cnt=0
for item in review:
    for domain in item["domain_labeled"]:
        if domain not in domain_dict.keys():
            domain_dict[domain]= [0,0]
        for key, score in item["score"].items():
            try:
                domain_dict[domain][0]+=float(item["score"][key])
                domain_dict[domain][1]+=1
            except Exception as e:
                print(e)
                cnt+=1

key_order = ["Humanities", "Language", "Social Science", "History", "Culture", "Technology", "Coding", "Math", "Natural Science", "Health"]
ordered_dict = OrderedDict((key, domain_dict[key]) for key in key_order if key in domain_dict)

file_name = file.split('/')[1].split('.jsonl')[0]

file_directory = 'outputs/stats/'
output_directory = os.path.dirname(file_directory)

# Check if the directory exists, if not, create it
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

with open(file_directory+file_name+"_domain.csv", "w") as f1: 
    write = csv.writer(f1)
    write.writerow(["domain", "sum", "count", "avg"])
    for key, dict in ordered_dict.items():
        list = dict
        print(key+'\t'+str(list[0])+'\t'+str(list[1]))
        write.writerow([key.split(' ')[0], str(list[0]), str(list[1]), str(list[0]/ list[1]) ])