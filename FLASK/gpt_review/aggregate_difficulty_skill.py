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

parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
parser.add_argument('-m', '--file', default='fig1')
args = parser.parse_args()

file = args.file

with open(file, 'r') as f:
    for line in f:
        review.append(json.loads(line))

difficulty = []
with open('../evaluation_set/flask_evaluation.jsonl', 'r') as f2:
    for line in f2:
        difficulty.append(json.loads(line))


init = {
    "simple lifestyle knowledge":[0,0],
    "advanced lifestyle knowledge":[0,0],
    "formal education knowledge":[0,0],
    "major level knowledge":[0,0],
    "expert level knowledge":[0,0]
}


difficulty_dict = {}
cnt=0
for index, item in enumerate(review):

    level = difficulty[index]["difficulty_labeled"]
    if len(item["score"])!=3:
        print("length issue!!!", item["score"], item )
    for key, score in item["score"].items():

            
        if key.split(' ')[0] not in difficulty_dict.keys():
            if 'logical' in key:
                # print(key)
                try:
                    if key.split(' ')[1] not in difficulty_dict:
                        print(key)
                        difficulty_dict[key.split(' ')[1]]=copy.deepcopy(init)
                except:
                    print(file, key, index)
            else:
                print(key)
                difficulty_dict[key.split(' ')[0]]=copy.deepcopy(init) 
        if item["score"][key] == "N/A":
            cnt+=1
        else: 
            if 'logical' in key:
                try:
                    difficulty_dict[key.split(' ')[1]][str(level)][0]+=float(score)
                    difficulty_dict[key.split(' ')[1]][str(level)][1]+=1
                except:
                    print(file)
            else: 
                try: 
                    difficulty_dict[key.split(' ')[0]][str(level)][0]+=float(score)
                    difficulty_dict[key.split(' ')[0]][str(level)][1]+=1
                except:
                    print(file)

key_order = ["robustness", "correctness", "efficiency", "factuality", "commonsense", "comprehension", "insightfulness", "completeness",  "metacognition","readability", "conciseness", "harmlessness"]
ordered_dict = OrderedDict((key, difficulty_dict[key]) for key in key_order if key in difficulty_dict)

file_name = file.split('/')[1].split('.jsonl')[0]

file_directory = 'outputs/stats/'
output_directory = os.path.dirname(file_directory)

# Check if the directory exists, if not, create it
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

with open(file_directory+file_name+"_difficulty_skill.csv", "w") as f1: 
    write = csv.writer(f1)
    write.writerow(["skill", "difficulty", "score", "count", "avg", "cumu"])
    for key, dict in ordered_dict.items():
        sum_score = 0 
        sum_count = 0 
        for level, list in dict.items():
            sum_score += list[0]
            sum_count += list[1]
            if list[1] != 0:
                write.writerow([key, level, str(list[0]), str(list[1]), str(list[0]/list[1]), str(sum_score/sum_count)])
            elif sum_count == 0: 
                write.writerow([key, level, str(list[0]), str(list[1]), "N/A", "N/A"])
            else: 
                write.writerow([key, level, str(list[0]), str(list[1]), "N/A", str(sum_score/sum_count)])