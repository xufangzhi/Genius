import json
import numpy as np
import re
import random
from data.math_example import MATH_POT_FEW_SHOT, MATH_COT_FEW_SHOT, GSM_COT_8_SHOT


def extract_numbers(input_string):
    return re.findall(r'\d+', input_string)


def longest_common_prefix(str1, str2):
    i = 0
    while i < len(str1) and i < len(str2) and str1[i] == str2[i]:
        i += 1
    return str1[:i]


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
        return False, None
    
    answer = boxed_contents.strip('\\').replace(",","").strip("$").strip()

    if "." in answer:
        pred_ans = answer
    elif "frac" in answer and len(extract_numbers(answer))==2:
        if float(extract_numbers(answer)[1])!=0:
            pred_ans = float(extract_numbers(answer)[0]) / float(extract_numbers(answer)[1])
    elif extract_numbers(answer):
        pred_ans = extract_numbers(answer)[0]
    else:
        return False, None

    try:
        if abs(float(gt) - float(pred_ans)) < 1e-5:
            return True, pred_ans
        else:
            return False, pred_ans
    except:
        return False, None
    return False, None



with open(f"/cpfs01/user/xufangzhi/o1/data/magpie_reasoning_150K_train.json") as file:
    origin_train_data = json.load(file)

organized_train_data = []
organized_sft_train_data = []
organized_preference_data = []
organized_step_preference_data = []

sample_id = 0
correct_num_final, correct_num_pool = 0, 0
for part_id in range(1,26):
    # DATA_PATH = f"/cpfs01/user/xufangzhi/o1/infer/results/reclor_logiqa_train_llama3.1_step_beam_2_sir_no_replace_rollout_4_foresight_4_{part_id}.json"
    # DATA_PATH = f"/cpfs01/user/xufangzhi/o1/infer/results/gsm_train_llama3.1_step_beam_2_sir_no_replace_rollout_4_foresight_8_{part_id}.json"
    DATA_PATH = f"/cpfs01/user/xufangzhi/o1/infer/results/magpie_reasoning_150K_train_llama3.1_step_beam_2_sir_no_replace_rollout_4_foresight_4_{part_id}.json"
    # DATA_PATH = f"/cpfs01/user/xufangzhi/o1/infer/results/openhermes_48k_train_llama3.1_step_beam_2_sir_no_replace_rollout_4_foresight_4_{part_id}.json"
    system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{GSM_COT_8_SHOT}"
    data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue

    # data = data[:3000]
    print(part_id, len(data))
    foresight_correct = False
    for i in range(len(data)):
        gt = data[i]['ground_truth']
        pred = data[i]['response']
        if eval_cot_answer(pred, gt)[0] and "\\boxed{" in pred:
        # if "\\boxed{" in pred:
            # organized_train_data.append({
            #     "id": sample_id,
            #     "messages": [
            #         {'role': 'system', 'content': system_prompt},
            #         {"role": "user", "content": "The question: "  + data[i]['question'] + "\nPlease directly output the reasoning steps.\nThe reasoning steps are:\n\n"},
            #         {"role": "assistant", "content": pred},
            #     ]
            # })
            correct_num_final += 1
            foresight_correct = True

        traj_pool = data[i]['traj_pool']
        logp_pool = data[i]['prob_pool']
        step_prob = data[i]['step_prob']
        adv_pool = data[i]['adv_pool']
        sample_id_pool = data[i]['sample_id_pool']

        candidate_list = []
        for j in range(len(traj_pool)):
            step_candidate_list = []
            step_candidate_list_by_beam = [[] for _ in range(2)]   # beam size 2
            step_candidate_list_positive, step_candidate_list_negative = [], []
            for k in range(len(traj_pool[j])):
                gt = data[i]['ground_truth']
                pred = traj_pool[j][k]
                if len(pred)<10000:
                    step_candidate_list.append((j,k,logp_pool[j][k], adv_pool[j][k], step_prob[j][k]))
                    step_candidate_list_by_beam[k//4].append((j,k,logp_pool[j][k], adv_pool[j][k], step_prob[j][k]))
                # if eval_cot_answer(pred, gt)[0] and "\\boxed{" in pred:
                if eval_cot_answer(pred, gt)[0] and "\\boxed{" in pred and j==0:
                    organized_train_data.append({
                        "id": sample_id,
                        "messages": [
                            {'role': 'system', 'content': system_prompt},
                            {"role": "user", "content": "The question: "  + data[i]['question'] + "\nPlease directly output the reasoning steps.\nThe reasoning steps are:\n\n"},
                            {"role": "assistant", "content": pred},
                        ]
                    })
                    correct_num_pool += 1
                    step_candidate_list_positive.append((j,k,logp_pool[j][k], adv_pool[j][k]))
                elif not eval_cot_answer(pred, gt)[0] and eval_cot_answer(pred, gt)[1] and len(pred)<10000:
                    candidate_list.append((j,k,logp_pool[j][k]))
                    step_candidate_list_negative.append((j,k,logp_pool[j][k], adv_pool[j][k]))

            """ supervised setting """
            # if len(step_candidate_list_positive)>=1 and len(step_candidate_list_negative)>=1:
            #     step_candidate_list_positive = sorted(step_candidate_list_positive, key=lambda x: x[2], reverse=True)
            #     step_candidate_list_negative = sorted(step_candidate_list_negative, key=lambda x: x[2], reverse=True)
            #     organized_step_preference_data.append({
            #         "prompt": "The question: " + data[i]['question'] + "\nPlease directly output the reasoning steps.\n",
            #         "chosen": [{"role": "assistant", "content": traj_pool[step_candidate_list_positive[0][0]][step_candidate_list_positive[0][1]]}],
            #         "rejected": [{"role": "assistant", "content": traj_pool[step_candidate_list_negative[-1][0]][step_candidate_list_negative[-1][1]]}]
            #     })
            # elif len(step_candidate_list_positive)==0 and len(step_candidate_list_negative)>=1:
            #     step_candidate_list_negative = sorted(step_candidate_list_negative, key=lambda x: x[2], reverse=True)
            #     organized_step_preference_data.append({
            #         "prompt": "The question: " + data[i]['question'] + "\nPlease directly output the reasoning steps.\n",
            #         "chosen": [{"role": "assistant", "content": data[i]['response']}],
            #         "rejected": [{"role": "assistant", "content": traj_pool[step_candidate_list_negative[-1][0]][step_candidate_list_negative[-1][1]]}]
            #     })

            """ un-supervised setting """
            if len(step_candidate_list)>=2:
                step_candidate_list = sorted(step_candidate_list, key=lambda x: x[2], reverse=True)
                step_candidate_list_by_beam = [sorted(sub_list, key=lambda x: x[2], reverse=True) for sub_list in step_candidate_list_by_beam]
                # max_id = step_candidate_list[0][1]

                # print(step_candidate_list_by_beam)
                # # print(step_prob)
                # input()

                # for beam in range(len(step_candidate_list_by_beam)):
                #     if len(step_candidate_list_by_beam[beam])>=2 and step_candidate_list_by_beam[beam][0][3]>0:    # 保证adv都是大于0
                #         try:
                #             random_numbers = random.sample(range(1, len(step_candidate_list_by_beam[beam])), 1)
                #         except:
                #             random_numbers = []
                #         for k in random_numbers:
                #             organized_step_preference_data.append({
                #                 "prompt": "The question: " + data[i]['question'] + "\nPlease directly output the reasoning steps.\n",
                #                 "chosen": [{"role": "assistant", "content": traj_pool[step_candidate_list_by_beam[beam][0][0]][step_candidate_list_by_beam[beam][0][1]]}],
                #                 "rejected": [{"role": "assistant", "content": traj_pool[step_candidate_list_by_beam[beam][k][0]][step_candidate_list_by_beam[beam][k][1]]}],
                #                 "chosen_weights": step_candidate_list_by_beam[beam][0][3],
                #                 "rejected_weights": step_candidate_list_by_beam[beam][k][3],
                #             })



                # if step_candidate_list[0][3]>0:  # adv > 0
                try:
                    random_numbers = random.sample(range(1, len(step_candidate_list)), 1)
                except:
                    random_numbers = []
                for k in random_numbers:                            
                    organized_step_preference_data.append({
                        # "prompt": "The question: " + data[i]['question'] + "\nPlease directly output the reasoning steps.\n",
                        "prompt": data[i]['question'],
                        "chosen": [{"role": "assistant", "content": traj_pool[step_candidate_list[0][0]][step_candidate_list[0][1]]}],
                        "rejected": [{"role": "assistant", "content": traj_pool[step_candidate_list[k][0]][step_candidate_list[k][1]]}],
                        "chosen_weights": step_candidate_list[0][3],
                        "rejected_weights": step_candidate_list[k][3],
                        "chosen_average_weights": np.mean([step_candidate_list[idx][3] for idx in range(len(step_candidate_list))]),
                    })

                #     organized_sft_train_data.append({
                #         "messages": [
                #             # {'role': 'system', 'content': system_prompt},
                #             {"role": "user", "content": "The question: "  + data[i]['question'] + "\nPlease directly output the reasoning steps.\n"},
                #             {"role": "assistant", "content": traj_pool[step_candidate_list[0][0]][step_candidate_list[0][1]]},
                #         ]
                #     })

                # try:
                #     random_numbers = random.sample(range(2, len(step_candidate_list)), 2)
                # except:
                #     random_numbers = []
                # for neg_id, k in enumerate(random_numbers):
                #     if neg_id <= 1:                           
                #         organized_step_preference_data.append({
                #             # "prompt": "The question: " + data[i]['question'] + "\nPlease directly output the reasoning steps.\n",
                #             "prompt": data[i]['question'],
                #             "chosen": [{"role": "assistant", "content": traj_pool[step_candidate_list[0][0]][step_candidate_list[0][1]]}],
                #             "rejected": [{"role": "assistant", "content": traj_pool[step_candidate_list[k][0]][step_candidate_list[k][1]]}],
                #             "chosen_weights": step_candidate_list[0][3],
                #             "rejected_weights": step_candidate_list[k][3],
                #             "chosen_average_weights": np.mean([step_candidate_list[idx][3] for idx in range(len(step_candidate_list))]),
                #         })
                #     else:
                #         organized_step_preference_data.append({
                #             # "prompt": "The question: " + data[i]['question'] + "\nPlease directly output the reasoning steps.\n",
                #             "prompt": data[i]['question'],
                #             "chosen": [{"role": "assistant", "content": traj_pool[step_candidate_list[1][0]][step_candidate_list[1][1]]}],
                #             "rejected": [{"role": "assistant", "content": traj_pool[step_candidate_list[k][0]][step_candidate_list[k][1]]}],
                #             "chosen_weights": step_candidate_list[1][3],
                #             "rejected_weights": step_candidate_list[k][3],
                #             "chosen_average_weights": np.mean([step_candidate_list[idx][3] for idx in range(len(step_candidate_list))]),
                #         })

                # print(traj_pool[step_candidate_list[0][0]][step_candidate_list[0][1]])
                # print("=======")
                # print(traj_pool[step_candidate_list[k][0]][step_candidate_list[k][1]])
                # print("---------------------------")
                # input()
                    


                # print(step_candidate_list)
                # input()
                # the first and the last
                # organized_step_preference_data.append({
                #     "prompt": "The question: " + data[i]['question'] + "\nPlease directly output the reasoning steps.\n",
                #     "chosen": [{"role": "assistant", "content": traj_pool[step_candidate_list[0][0]][step_candidate_list[0][1]]}],
                #     "rejected": [{"role": "assistant", "content": traj_pool[step_candidate_list[-1][0]][step_candidate_list[-1][1]]}],
                #     "chosen_weights": step_candidate_list[0][3],
                #     "rejected_weights": step_candidate_list[-1][3],
                # })
                
                # print(step_candidate_list)
                # input()
                # organized_train_data.append({
                #     "id": sample_id,
                #     "messages": [
                #         {'role': 'system', 'content': system_prompt},
                #         {"role": "user", "content": "The question: "  + data[i]['question'] + "\nPlease directly output the reasoning steps.\n"},
                #         {"role": "assistant", "content": traj_pool[step_candidate_list[0][0]][step_candidate_list[0][1]]},
                #     ]
                # })

        for j in range(len(data[i]['response_all_beams'])):
            organized_train_data.append({
                "id": sample_id,
                "messages": [
                    # {'role': 'system', 'content': system_prompt},
                    {"role": "user", "content": "The question: "  + data[i]['question'] + "\nPlease directly output the reasoning steps.\n"},
                    {"role": "assistant", "content": data[i]['response_all_beams'][j]},
                ]
            })

        sample_id += 1

        candidate_list = sorted(candidate_list, key=lambda x: x[2], reverse=True)
    
        # if candidate_list and foresight_correct:
        #     for m in range(min(2,len(candidate_list))):
        #         organized_preference_data.append({
        #             "prompt": "The question: " + data[i]['question'] + "\nPlease directly output the reasoning steps.\n",
        #             "chosen": [{"role": "assistant", "content": data[i]['response']}],
        #             "rejected": [{"role": "assistant", "content": traj_pool[candidate_list[m][0]][candidate_list[m][1]]}]
        #         })


# print(sample_id)
# print(correct_num_final / sample_id)
# print(correct_num_pool / sample_id / 8)

# organized_step_preference_data = organized_step_preference_data[:128]
# with open(f"/cpfs01/user/xufangzhi/symbol-llm-omni/open-instruct/data/250204-2.jsonl", "w") as file:
#     for i in range(len(organized_step_preference_data)):
#         file.write(json.dumps(organized_step_preference_data[i]) + '\n')
#         file.flush()

print(len(organized_step_preference_data))
