import sys
sys.path.append("../")
import argparse
import json
import os

import shortuuid
import logging
from openai_concurrent import OpenAIChatCompletionConcurrent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import re
import ast

def parse_score(review, num):
    try:
        
        match = re.findall(r'{[^}]+}', review)
        if len(match)>0:
            
            dictionary_part = match[-1].replace("\n", "").replace('_', " ").lower()
            lines = ast.literal_eval(dictionary_part)
            for key, value in lines.items():
                if value == 'na':
                    lines[key] = 'N/A'
                elif value == 'n/a':
                    lines[key] = 'N/A'
                elif value == 'not applicable':
                    lines[key]= 'N/A'
            return lines
        else:
            return {}

    except Exception as e:
        logger.error(f'{e}\nContent: {review}\n'
                     'You must manually fix the score pair.')
        return {}


def gen_prompt(reviewer_jsons, prompt_jsons, skills_jsons, response, item):
    reviewer_idx = 1
    prompt_id = reviewer_jsons[reviewer_idx]['prompt_id']
    prompt_json = prompt_jsons[prompt_id-1]
    assert prompt_json['prompt_id'] == prompt_id

    sys_prompt = prompt_json['system_prompt']
    prompt_template = prompt_json['prompt_template']
    defaults = prompt_json['defaults']

    # skills =metrics
    skills = ""
    metric_list = item["skill"]
    for label in metric_list:
        for skill in skills_jsons:
            if label in skill["Skill"]:
                name = skill["Skill"]
                criteria = skill["Criteria"]
                skills+=f"\n{name}: {criteria}"
                scoring = skill["Scoring"]
                skills+=f"\nScore 1: {scoring['1']}"
                skills+=f"\nScore 2: {scoring['2']}"
                skills+=f"\nScore 3: {scoring['3']}"
                skills+=f"\nScore 4: {scoring['4']}"
                skills+=f"\nScore 5: {scoring['5']}\n\n"
                break
    prompt = prompt_template.format(question=item["instruction"], response=response, skills=skills, num=3, sample_answer=item["answer"], **defaults)
    return sys_prompt, prompt


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    file_extension = file_path.split('.')[-1]
    if file_extension=="jsonl":
        with open(file_path, 'r') as f:
            json_list = []
            for line in f:
                json_list.append(json.loads(line))
            return json_list
    else:
        with open(file_path, 'r') as f:
            return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--key-file', default='../openai_info/api_info.json')
    parser.add_argument('-q', '--question-file', default='../evaluation_set/flask_evaluation.jsonl')
    parser.add_argument('-s', '--skillset-file', default='../metadata_annotation/skillset/src/skillset_description.json')
    parser.add_argument('-a', '--answer-file', default='../model_output/outputs/chatgpt.jsonl')
    parser.add_argument('-p', '--prompt-file', default='src/prompt.jsonl')
    parser.add_argument('-r', '--reviewer-file', default='src/reviewer.jsonl')
    parser.add_argument('-o', '--output-review-file', default='outputs/chatgpt_review.jsonl')
    parser.add_argument('-e', '--output-error-file', default='outputs/chatgpt_review_error.jsonl')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    key_jsons = get_json_list(args.key_file)
    question_jsons = get_json_list(args.question_file)
    skills_jsons = get_json_list(args.skillset_file)
    answer_jsons = get_json_list(args.answer_file) 
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)

    handles = []
    review_jsons = []
    total_len = len(question_jsons)
    question_idx_list = list(range(total_len))
    question_copy = []
    answer_copy = []

    requests = []
    for i in question_idx_list:
        for row in answer_jsons:
            if row.get('question_id') == question_jsons[i]['idx']:
                answer_elem = row
                break
        answer_copy.append(answer_elem)
        assert answer_copy[i]['question_id'] == question_jsons[i]['idx']
        question_copy.append(question_jsons[i])
        sys_prompt, prompt = gen_prompt(reviewer_jsons, prompt_jsons, skills_jsons,answer_copy[i]["text"], question_jsons[i])
        # print(prompt)
        review_id = shortuuid.uuid()
        review_jsons.append({
            'review_id': review_id,
            'question_id': question_jsons[i]['idx'],
            'metadata': {},
        })
        requests.append(
            {
                'review_id': review_id,
                'question_id': question_jsons[i]['idx'],
                'metadata': {},
                'request': {
                    "model": "gpt-4-0613",
                    "messages":[
                        {
                            'role': 'system',
                            'content': sys_prompt
                        },
                        {
                            'role': 'user',
                            'content': prompt,
                        }
                    ]
                },
                # setting temperature 0 for reproducibility
                "temperature": 0,
                "max_tokens": args.max_tokens
            }
        )

    openai_concurrent = OpenAIChatCompletionConcurrent(api_keys=key_jsons["api_keys"], requests_per_minute=180, expected_response_seconds=5)
    responses, fails = openai_concurrent.create_many(requests)

    reviews = [response['response'].choices[0].message.content for response in responses]
    total_tokens = [response['response'].usage.total_tokens for response in responses]
    print("total_token:", sum(total_tokens))

    output_directory = os.path.dirname(args.output_error_file)

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    delete_index = []
    if len(fails)>0:
        with open(f'{args.output_error_file}', 'w') as output_error_file:
            try:
                for idx, fail in enumerate(fails):
                    print("fail:", fail)
                    for index, item in enumerate(question_copy):
                        if int(item.get("question_id")) == int(fail['question_id']):
                            delete_elem_idx = index 
                    delete_index.append(delete_elem_idx)
                    output_error_file.write(json.dumps(question_copy[delete_elem_idx]) + '\n')
            except: 
                print("@@@", delete_index)
                delete_index=[]
    
    question_copy = [item for index, item in enumerate(question_copy) if index not in delete_index]

    with open(f'{args.output_review_file}', 'a') as output_review_file:
        for idx, review in enumerate(reviews):
            num = 3
            scores = parse_score(review, num)
            review_jsons[idx] = question_copy[idx]
            for row in answer_jsons:
                if row.get('question_id') == question_copy[idx]['idx']:
                    review_jsons[idx]['target_txt'] = row["text"]
            review_jsons[idx]['review'] = review
            review_jsons[idx]['score'] = scores
            review_jsons[idx]['total_tokens_step4'] = total_tokens[idx]
            try:
                output_review_file.write(json.dumps(review_jsons[idx]) + '\n')
            except Exception as e:
                output_review_file.write('\n')
                print(review_jsons[idx]['question_id'])
        output_review_file.close()

    with open(f'{args.output_review_file}', 'r') as output_read_file:
        lines = output_read_file.readlines()
        output_read_file.close()
    json_objects = [json.loads(line) for line in lines]
    sorted_objects = sorted(json_objects, key=lambda obj: obj.get('idx'))

    with open(f'{args.output_review_file}', 'w') as output_write_file:
        for obj in sorted_objects:
            try: 
                output_write_file.write(json.dumps(obj) + '\n')
            except Exception as e:
                output_write_file.write('\n')
                print(obj['idx'])
