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
author = []

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--file', default='fig1')
args = parser.parse_args()

file = args.file

with open(file, 'r') as f:
    for line in f:
        review.append(json.loads(line))



skill_dict = {}
cnt=0
for index, item in enumerate(review):
    # if len(item["score"])!=3:
        # print("length issue!!!", item["score"], item )
    for key, score in item["score"].items():
        if key.split(' ')[0] not in skill_dict.keys():
            if 'logical' in key:
                # print(key)
                try:
                    if key.split(' ')[1] not in skill_dict:
                        # print(key)
                        skill_dict[key.split(' ')[1]]=[0,0]
                except:
                    print(file, key, index)
            else:
                # print(key)
                skill_dict[key.split(' ')[0]]=[0,0]
        if item["score"][key] == "N/A":
            cnt+=1
        else: 
            if 'logical' in key:
                try:
                    skill_dict[key.split(' ')[1]][0]+=float(score)
                    skill_dict[key.split(' ')[1]][1]+=1
                except:
                    print(file)
            else: 
                try: 
                    skill_dict[key.split(' ')[0]][0]+=float(score)
                    skill_dict[key.split(' ')[0]][1]+=1
                except:
                    print(file)

key_order = ["robustness", "correctness", "efficiency", "factuality", "commonsense", "comprehension", "insightfulness", "completeness",  "metacognition","readability", "conciseness", "harmlessness"]
ordered_dict = OrderedDict((key, skill_dict[key]) for key in key_order if key in skill_dict)

file_name = file.split('/')[-1].split('.jsonl')[0]

file_directory = 'outputs/stats/'
output_directory = os.path.dirname(file_directory)

# Check if the directory exists, if not, create it
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

with open(file_directory+file_name+"_skill.csv", "w") as f1: 
    write = csv.writer(f1)
    write.writerow(["skill", "score", "count", "avg"])
    for key, dict in ordered_dict.items():
        list = dict
        write.writerow([key, str(list[0]), str(list[1]), str(list[0]/list[1])])