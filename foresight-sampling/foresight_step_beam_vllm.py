import json
import random
import numpy as np
import torch
import os
import argparse
from data.math_example import MATH_POT_FEW_SHOT, MATH_COT_FEW_SHOT, GSM_COT_8_SHOT, MATH_COT_4_SHOT
from data.logic_example import LOGIC_MRC_COT_4_SHOT
from vllm import LLM, SamplingParams
from transformers import (
    StoppingCriteriaList,
    StoppingCriteria,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='math')
    parser.add_argument('--model_id', type=str, default='llama3.1')
    parser.add_argument('--data_path', type=str, default='/cpfs01/user/xufangzhi/o1/data/gsm_test.json')
    parser.add_argument('--output_path', type=str, default='/cpfs01/user/xufangzhi/o1/infer/results/gsm_test_adv_no_replace_step_beam_4_rollout_4_foresight_4.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--step_beam_size', type=int, default=4)
    parser.add_argument('--num_rollout', type=int, default=4)
    parser.add_argument('--num_foresight', type=int, default=4)
    parser.add_argument('--record_process', type=bool, default=True)
    parser.add_argument('--strategy', type=str, default='adv_no_replace')
    parser.add_argument('--mode', type=str, default='no_light')
    args = parser.parse_args()


    if args.model_id=="llama3.1":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    elif args.model_id=="mistral-v0.3":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de/"
    elif args.model_id=="qwen2.5":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/"
    elif args.model_id=="qwen2.5-3b":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1/"
    elif args.model_id=="gemma2":
        # no_system_prompt = True
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819/"
    elif args.model_id=="deepseek-v1.5":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    elif args.model_id=="phi3":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/38143357bf52ce57009ecbd58cf9f0b0029cb393"
    else:
        if os.path.exists(f"/nas/shared/NLP_A100/xufangzhi/symbol-llm-omni/open-instruct/output/{args.model_id}_mistral-v0.3_8B"):
            PATH_TO_CONVERTED_WEIGHTS = f"/nas/shared/NLP_A100/xufangzhi/symbol-llm-omni/open-instruct/output/{args.model_id}_mistral-v0.3_8B"
        else:
            PATH_TO_CONVERTED_WEIGHTS = f"/nas/shared/NLP_A100/xufangzhi/symbol-llm-omni/open-instruct/output/{args.model_id}_llama3.1_8B"
    # PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    # PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-omni/open-instruct/output/metamathqa_train_correct_foresight_step_all_final_rollout_4_foresight_4_llama3.1_8B"


    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, max_length=2048, trust_remote_code=True)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    stop_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained(
    #     PATH_TO_CONVERTED_WEIGHTS, 
    #     trust_remote_code=True
    # )
    device = "cuda:0"

    model = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)


    num_rollout = args.num_rollout
    num_foresight = args.num_foresight
    step_beam_size = args.step_beam_size
    
    DATA_PATH = args.data_path
    with open(DATA_PATH) as file:
        test_data = json.load(file)
    if args.record_process:
        idx = int(args.output_path.split("_")[-1].split(".json")[0])
        # origin_num = 0
        # with open(args.output_path) as file:
        #     for line in file:
        #         origin_num += 1
        # test_data = test_data[(idx-1)*3000:idx*3000]
        # test_data = test_data[(idx-1)*3000:(idx-1)*3000+2000]
        test_data = test_data[(idx-1)*4800 : idx * 4800]

    OUTPUT_PATH = args.output_path
    if "gsm" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{GSM_COT_8_SHOT}"
    elif "math" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{MATH_COT_4_SHOT}"
    elif "reclor" in args.data_path or "logiqa" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D'. Read the question and options thoroughly and select the correct answer from the four answer labels. Please finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{LOGIC_MRC_COT_4_SHOT}\n"
    else:
        system_prompt = ""
    
    with open(OUTPUT_PATH, "w") as f:
        for i in range(len(test_data)):
            traj_pool = [[] for _ in range(num_foresight)]
            step_pool = [[] for _ in range(num_foresight)]
            prob_pool = [[] for _ in range(num_foresight+1)]
            adv_pool = [[] for _ in range(num_foresight+1)]
            step_prob = [[] for _ in range(num_foresight)]
            sample_id_pool = []
            
            traj_complete = False
            previous_steps_list = ["The reasoning steps are:\n\n" for _ in range(step_beam_size)]
            previous_q_value_list = [0.0 for _ in range(step_beam_size)]
            T = 0
            for T in range(num_foresight):
                skip_foresight = False
                reasoning_steps_list = previous_steps_list

                if "gsm" in args.data_path or "math" in args.data_path:
                    question = test_data[i]['input']
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': 'The question: ' + question + '\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n'},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "reclor" in args.data_path or "logiqa" in args.data_path:
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "magpie" in args.data_path:
                    chat = [
                        {'role': 'user', 'content': test_data[i]['instruction']},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "hermes" in args.data_path:
                    chat = [
                        {'role': 'user', 'content': test_data[i]['instruction']},
                        {'role': 'assistant', 'content': ''}
                    ]
                inputs = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                )
                # print(inputs)
                inputs = inputs.replace(stop_token,"").strip()
                
                inputs_list = [inputs + reasoning_steps_list[beam_idx] for beam_idx in range(step_beam_size)]

                sampling_params = SamplingParams(max_tokens=1024, n=num_rollout, logprobs=1, temperature=0.6, stop=["\n", "<end_of_reasoning>"])
                # sampling_params = SamplingParams(max_tokens=1024 ,n=num_rollout, logprobs=0, best_of=4, temperature=0, use_beam_search=True, stop=["\n", "<end_of_reasoning>"])
                outputs = model.generate(inputs_list, sampling_params)
                

                selected_steps = []
                inputs_list = []
                candidates_list = []

                for beam_idx in range(step_beam_size):
                    for j in range(num_rollout):
                        output = outputs[beam_idx].outputs[j]
                        response = output.text.strip()

                        step_prob[T].append(output.cumulative_logprob / (len(output.token_ids)+1e-8))

                        selected_steps.append(response)

                        reasoning_steps_candidate = reasoning_steps_list[beam_idx] + "\n" + response
                        candidates_list.append(response)

                        if "gsm" in args.data_path or "math" in args.data_path:
                            question = test_data[i]['input']
                            chat = [
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': 'The question: ' + question + '\nPlease directly output the reasoning steps.\n'},
                                {'role': 'assistant', 'content': ''}
                            ]
                        elif "reclor" in args.data_path or "logiqa" in args.data_path:
                            chat = [
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}\n"},
                                {'role': 'assistant', 'content': ''}
                            ]
                        elif "magpie" in args.data_path:
                            chat = [
                                {'role': 'user', 'content': test_data[i]['instruction']},
                                {'role': 'assistant', 'content': ''}
                            ]
                        elif "hermes" in args.data_path:
                            chat = [
                                {'role': 'user', 'content': test_data[i]['instruction']},
                                {'role': 'assistant', 'content': ''}
                            ]
                        # chat = system_prompt + '\nThe question: ' + question + '\nPlease directly output the reasoning steps.\nThe reasoning steps are:\n' + reasoning_steps_candidate
                        inputs_list.append(tokenizer.apply_chat_template(
                            chat,
                            tokenize=False,
                            # add_generation_prompt=True
                        ).rstrip(stop_token).rstrip() + reasoning_steps_candidate)
                
                if args.mode=="light" and min(step_prob[T])>-0.25:
                    skip_foresight = True

                if not skip_foresight:
                    sampling_params = SamplingParams(max_tokens=1024 ,n=1, logprobs=1, stop="<end_of_reasoning>")
                    # sampling_params = SamplingParams(max_tokens=1024 ,n=1, logprobs=1, best_of=4, temperature=0, use_beam_search=True, stop=["<end_of_reasoning>"])
                    outputs = model.generate(inputs_list, sampling_params)

                    normalized_logp_list = []
                    advantages_list = []
                    for j in range(num_rollout * step_beam_size):
                        output = outputs[j].outputs[0]
                        response = output.text.strip()
                        
                        step_pool[T].append(response)
                        # print(len(previous_steps_list), j)
                        traj_pool[T].append(previous_steps_list[j//num_rollout] + candidates_list[j].strip() + "\n" + response)
                        prob_pool[T].append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                        adv_pool[T].append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[j//num_rollout])
                        normalized_logp_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                        advantages_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[j//num_rollout])

                if args.strategy == "highest":  # select the step with highest rewards
                    selected_index = normalized_logp_list.index(max(normalized_logp_list))
                elif args.strategy == "random" or skip_foresight:  # select the step with random rewards
                    selected_index_list = random.sample(range(num_rollout*step_beam_size), step_beam_size).tolist()
                elif args.strategy == "sir":
                    temp = 0.1
                    def softmax(x):
                        e_x = np.exp(np.array(x))
                        return e_x / e_x.sum(axis=0)
                    weights = softmax([logp/temp for logp in normalized_logp_list])
                    selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size).tolist()
                elif args.strategy == "sir_mask":
                    temp = 0.1
                    def masked_softmax(x):
                        mean_value = np.mean(x)
                        mask = [True if xx<mean_value else False for xx in x]
                        # 将需要屏蔽的位置设为极大负数
                        masked_x = np.where(mask, x, -np.inf)
                        # 计算 softmax
                        exp_x = np.exp(masked_x - np.max(masked_x, axis=-1, keepdims=True))
                        softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
                        return softmax_x

                    normalized_logp_list_filter = []
                    weights = masked_softmax([logp/temp for logp in normalized_logp_list])
                    selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size).tolist()
                elif args.strategy == "sir_no_replace":
                    temp = 0.1
                    def softmax(x):
                        e_x = np.exp(np.array(x))
                        return e_x / e_x.sum(axis=0)
                    weights = softmax([logp/temp for logp in normalized_logp_list])
                    selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()

                elif args.strategy == "adv":
                    temp = 0.1
                    def softmax(x):
                        e_x = np.exp(np.array(x))
                        return e_x / e_x.sum(axis=0)
                    weights = softmax([logp/temp for logp in advantages_list])
                    selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size).tolist()

                elif args.strategy == "adv_no_replace":
                    temp = 0.1
                    def softmax(x):
                        e_x = np.exp(np.array(x))
                        return e_x / e_x.sum(axis=0)
                    weights = softmax([logp/temp for logp in advantages_list])
                    selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size).tolist()

                sample_id_pool.append(selected_index_list)

                previous_steps_list_updated, previous_q_value_list = [], []
                for m, selected_index in enumerate(selected_index_list):
                    previous_steps_list_updated.append(previous_steps_list[selected_index//num_rollout] + candidates_list[selected_index].strip() + "\n")
                    previous_q_value_list.append(normalized_logp_list[selected_index])

                    if "\\boxed{" in candidates_list[selected_index] or "<boxed>" in candidates_list[selected_index] or "<end_of_reasoning>" in candidates_list[selected_index]:
                        traj_complete = True
                previous_steps_list = previous_steps_list_updated

            
            if "gsm" in args.data_path or "math" in args.data_path:
                question = test_data[i]['input']
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'The question: ' + question + '\nPlease directly output the reasoning steps.\n'},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "reclor" in args.data_path or "logiqa" in args.data_path:
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}\n"},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "magpie" in args.data_path:
                chat = [
                    {'role': 'user', 'content': test_data[i]['instruction']},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "hermes" in args.data_path:
                chat = [
                    {'role': 'user', 'content': test_data[i]['instruction']},
                    {'role': 'assistant', 'content': ''}
                ]
            inputs = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            ).rstrip(stop_token).rstrip()

            inputs_list = [inputs + previous_steps_list[beam_idx] for beam_idx in range(step_beam_size)]
            # chat = system_prompt + '\nThe question: ' + question + '\nPlease directly output the reasoning steps.\nThe reasoning steps are:\n' + previous_steps
            # sampling_params = SamplingParams(max_tokens=3000 ,n=step_beam_size, logprobs=0, best_of=4, temperature=0, use_beam_search=True, stop="<end_of_reasoning>")
            sampling_params = SamplingParams(max_tokens=3000 ,n=step_beam_size, logprobs=0, stop="<end_of_reasoning>")

            outputs = model.generate(inputs_list, sampling_params)

            normalized_logp_list = []
            advantages_list = []
            candidates_list = []
            for j in range(step_beam_size):
                output = outputs[j].outputs[0]
                response = output.text.strip()
                candidates_list.append(response)
                normalized_logp_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                advantages_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[j//num_rollout])
                prob_pool[T+1].append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                adv_pool[T+1].append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[j//num_rollout])

            if args.strategy == "sir":
                temp = 0.1
                def softmax(x):
                    e_x = np.exp(np.array(x))
                    return e_x / e_x.sum(axis=0)
                weights = softmax([logp/temp for logp in normalized_logp_list])
                selected_index_final = np.random.choice(len(weights), p=weights)
            if args.strategy == "sir_mask":
                temp = 0.1
                def masked_softmax(x):
                    mean_value = np.mean(x)
                    mask = [True if xx<mean_value else False for xx in x]
                    # 将需要屏蔽的位置设为极大负数
                    masked_x = np.where(mask, x, -np.inf)
                    # 计算 softmax
                    exp_x = np.exp(masked_x - np.max(masked_x, axis=-1, keepdims=True))
                    softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
                    return softmax_x

                normalized_logp_list_filter = []
                weights = masked_softmax([logp/temp for logp in normalized_logp_list])
                selected_index_final = np.random.choice(len(weights), p=weights)
            if args.strategy == "sir_no_replace":
                temp = 0.1
                def softmax(x):
                    e_x = np.exp(np.array(x))
                    return e_x / e_x.sum(axis=0)
                weights = softmax([logp/temp for logp in normalized_logp_list])
                selected_index_final = np.random.choice(len(weights), p=weights, replace=False)
            elif args.strategy == "adv":
                temp = 0.1
                def softmax(x):
                    e_x = np.exp(np.array(x))
                    return e_x / e_x.sum(axis=0)
                weights = softmax([logp/temp for logp in advantages_list])
                selected_index_final = np.random.choice(len(weights), p=weights)

            elif args.strategy == "adv_no_replace":
                temp = 0.1
                def softmax(x):
                    e_x = np.exp(np.array(x))
                    return e_x / e_x.sum(axis=0)
                weights = softmax([logp/temp for logp in advantages_list])
                selected_index_final = np.random.choice(len(weights), p=weights, replace=False)

            sample_id_pool.append([selected_index_final for _ in range(step_beam_size)])
            whole_traj = previous_steps_list[selected_index_final] + "\n" + candidates_list[selected_index_final]
            whole_traj_list = [previous_steps_list[beam_idx] + "\n" + candidates_list[beam_idx] for beam_idx in range(step_beam_size)]

            ###########################################
            #           Write to result file          #
            ###########################################
            result = {}
            result['id'] = i
            if "gsm" in args.data_path or "math" in args.data_path:
                result['question'] = question
                result['ground_truth'] = test_data[i]['target']
            elif "reclor" in args.data_path or "logiqa" in args.data_path:
                result['question'] = 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"
                result['ground_truth'] = test_data[i]['label'] if "label" in test_data[i] else None
            elif "magpie" in args.data_path:
                result['question'] = test_data[i]['instruction']
                result['ground_truth'] = test_data[i]['response']
            elif "hermes" in args.data_path:
                result['question'] = test_data[i]['instruction']
                result['ground_truth'] = test_data[i]['response']

            result['response'] = whole_traj
            result['response_all_beams'] = whole_traj_list


            if args.record_process:
                result['foresight_steps'] = T + 1
                result['traj_pool'] = traj_pool
                result['step_pool'] = step_pool
                result['prob_pool'] = prob_pool
                result['adv_pool'] = adv_pool
                result['step_prob'] = step_prob
                result['sample_id_pool'] = sample_id_pool
            f.write(json.dumps(result) + '\n')
            f.flush()