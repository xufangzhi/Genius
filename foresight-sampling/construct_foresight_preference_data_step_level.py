import json
import numpy as np
import re


def extract_numbers(input_string):
    return re.findall(r'\d+', input_string)


def find_last_number(input_string):
    words = input_string.split()
    numbers = []
    for word in words:
        try:
            number = float(word)
            numbers.append(number)
        except ValueError:
            pass

    if not numbers:
        return ""
    return str(numbers[-1])


def eval_cot_answer(pred, gt):
    boxed_contents = ""
    try:
        if "\\boxed{" in pred:
            boxed_contents = re.findall(r'\\boxed\{(.*?)\}', pred)
            if boxed_contents:
                boxed_contents = boxed_contents[-1]
            else:
                boxed_contents = ""
        elif "<boxed>" in pred:
            boxed_contents = re.findall(r'<boxed>(\d+)<\/boxed>', pred)
            if boxed_contents:
                boxed_contents = boxed_contents[-1].strip()
            else:
                boxed_contents = ""
        elif "The answer is:" in pred:
            boxed_contents = pred.split("The answer is:")[-1].strip()
        else:
            boxed_contents = find_last_number(pred[-50:])  # from last 100 chars to parse the possible answer
    except:
        return False
    
    answer = boxed_contents.strip('\\').replace(",","").strip("$").strip()

    if "." in answer:
        pred_ans = answer
    elif "frac" in answer and len(extract_numbers(answer))==2:
        pred_ans = float(extract_numbers(answer)[0]) / float(extract_numbers(answer)[1])
    elif extract_numbers(answer):
        pred_ans = extract_numbers(answer)[0]
    else:
        return False

    try:
        if abs(float(gt) - float(pred_ans)) < 1e-5:
            return True
    except:
        return False
    return False



with open(f"/cpfs01/user/xufangzhi/o1/data/metamathqa_train.json") as file:
    origin_train_data = json.load(file)

organized_train_data = []
sample_id = 0
correct_num_final, correct_num_pool = 0, 0
for part_id in range(1,11):
    DATA_PATH = f"/cpfs01/user/xufangzhi/o1/infer/results/metamathqa_train_sir_rollout_4_foresight_4_{part_id}.json"

    with open(DATA_PATH, 'r') as f:
        data = [json.loads(line) for line in f]

    for i in range(len(data)):
        gt = data[i]['ground_truth']
        pred = data[i]['response']
        if eval_cot_answer(pred, gt):
            organized_train_data.append({
                "id": sample_id,
                "messages": [
                    {"role": "user", "content": "The question: "  + data[i]['question'] + "\nPlease directly output the reasoning steps.\nThe reasoning steps are:\n\n"},
                    {"role": "assistant", "content": pred},
                ]
            })
            correct_num_final += 1

        traj_pool = data[i]['traj_pool']
        for j in range(len(traj_pool)):
            for k in range(len(traj_pool[j])):
                gt = data[i]['ground_truth']
                pred = traj_pool[j][k]
                if eval_cot_answer(pred, gt):
                    # organized_train_data.append({
                    #     "id": sample_id,
                    #     "messages": [
                    #         {"role": "user", "content": "The question: "  + data[i]['question'] + "\nPlease directly output the reasoning steps.\nThe reasoning steps are:\n\n"},
                    #         {"role": "assistant", "content": pred},
                    #     ]
                    # })
                    correct_num_pool += 1
        sample_id += 1
print(sample_id)
print(correct_num_final / sample_id)
print(correct_num_pool / sample_id / 16)

with open(f"/cpfs01/user/xufangzhi/symbol-llm-omni/open-instruct/data/gsm_train_correct_foresight_step_all_final_rollout_4_foresight_4.jsonl", "w") as file:
    for i in range(len(organized_train_data)):
        file.write(json.dumps(organized_train_data[i]) + '\n')
        file.flush()

print(len(organized_train_data))