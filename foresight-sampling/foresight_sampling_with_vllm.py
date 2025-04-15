import os
import json
import random
import argparse
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def softmax(x, temp=1.0):
    x = np.array(x) / temp
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def build_chat_prompt(instruction: str, reasoning_steps: str, tokenizer, stop_token: str) -> str:
    chat = [
        {'role': 'user', 'content': instruction},
        {'role': 'assistant', 'content': ''}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return prompt.replace(stop_token, '').strip() + reasoning_steps

def normalize_logprob(logprob: float, length: int) -> float:
    return logprob / (length + 1e-8)

def select_indices(strategy: str, values: List[float], step_beam_size: int, num_rollout: int) -> List[int]:
    if strategy == "highest":
        return [values.index(max(values))]
    elif strategy == "random":
        return random.sample(range(len(values)), step_beam_size)
    elif strategy in {"predictive_decoding", "foresight_sampling"}:
        weights = softmax(values, temp=0.1)
        return np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def load_data(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, 'r') as f:
        return json.load(f)

def save_result(output_path: str, result: Dict[str, Any]):
    with open(output_path, "a") as f:
        f.write(json.dumps(result) + '\n')
        f.flush()

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, max_length=2048, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    stop_token = tokenizer.eos_token

    model = LLM(model=args.model_path, tensor_parallel_size=1, trust_remote_code=True)

    test_data = load_data(args.data_path)

    for i, item in enumerate(tqdm(test_data)):
        instruction = item['instruction']
        ground_truth = item['response']
        step_beam_size = args.step_beam_size
        num_rollout = args.num_rollout
        num_foresight = args.num_foresight

        traj_pool, step_pool, prob_pool, adv_pool = [[] for _ in range(num_foresight+1)], [[] for _ in range(num_foresight)], [[] for _ in range(num_foresight+1)], [[] for _ in range(num_foresight+1)]
        step_prob, sample_id_pool = [[] for _ in range(num_foresight)], []

        previous_steps_list = ["The reasoning steps are:\n\n" for _ in range(step_beam_size)]
        previous_q_value_list = [0.0] * step_beam_size
        traj_complete = False

        for T in range(num_foresight):
            reasoning_steps_list = previous_steps_list
            inputs_list = [build_chat_prompt(instruction, rs, tokenizer, stop_token) for rs in reasoning_steps_list]

            sampling_params = SamplingParams(max_tokens=1024, n=num_rollout, logprobs=1, temperature=0.6, stop=["\n", "<end_of_reasoning>"])
            outputs = model.generate(inputs_list, sampling_params)

            candidates_list, full_inputs_list = [], []
            for beam_idx in range(step_beam_size):
                for j in range(num_rollout):
                    output = outputs[beam_idx].outputs[j]
                    response = output.text.strip()
                    prob = normalize_logprob(output.cumulative_logprob, len(output.token_ids))
                    step_prob[T].append(prob)

                    reasoning_step = reasoning_steps_list[beam_idx] + "\n" + response
                    candidates_list.append(response)
                    full_inputs_list.append(build_chat_prompt(instruction, reasoning_step, tokenizer, stop_token))

            skip_foresight = args.mode == "light" and min(step_prob[T]) > -0.25
            normalized_logp_list, advantages_list = [], []

            if not skip_foresight:
                foresight_params = SamplingParams(max_tokens=1024, n=1, logprobs=1, stop="<end_of_reasoning>")
                foresight_outputs = model.generate(full_inputs_list, foresight_params)

                for j, output in enumerate(foresight_outputs):
                    res = output.outputs[0]
                    prob = normalize_logprob(res.cumulative_logprob, len(res.token_ids))
                    advantage = prob - previous_q_value_list[j // num_rollout]

                    step_pool[T].append(res.text.strip())
                    traj_pool[T].append(previous_steps_list[j // num_rollout] + candidates_list[j].strip() + "\n" + res.text.strip())
                    prob_pool[T].append(prob)
                    adv_pool[T].append(advantage)
                    normalized_logp_list.append(prob)
                    advantages_list.append(advantage)

            strategy_input = advantages_list if args.strategy.startswith("adv") else normalized_logp_list
            selected_index_list = select_indices(args.strategy, strategy_input, step_beam_size, num_rollout)
            sample_id_pool.append(selected_index_list)

            new_steps_list, new_q_values = [], []
            for idx in selected_index_list:
                beam_id = idx // num_rollout
                new_steps = previous_steps_list[beam_id] + candidates_list[idx].strip() + "\n"
                new_steps_list.append(new_steps)
                new_q_values.append(normalized_logp_list[idx])

                if any(tag in candidates_list[idx] for tag in ["\\boxed{", "<boxed>", "<end_of_reasoning>"]):
                    traj_complete = True

            previous_steps_list, previous_q_value_list = new_steps_list, new_q_values

        # Final answer generation
        final_inputs = [build_chat_prompt(instruction, rs, tokenizer, stop_token) for rs in previous_steps_list]
        final_params = SamplingParams(max_tokens=3000, n=step_beam_size, logprobs=0, stop="<end_of_reasoning>")
        final_outputs = model.generate(final_inputs, final_params)

        final_responses, final_probs, final_advs = [], [], []
        for j in range(step_beam_size):
            output = final_outputs[j].outputs[0]
            response = output.text.strip()
            prob = normalize_logprob(output.cumulative_logprob, len(output.token_ids))
            adv = prob - previous_q_value_list[j]

            final_responses.append(response)
            final_probs.append(prob)
            final_advs.append(adv)
            prob_pool[num_foresight].append(prob)
            adv_pool[num_foresight].append(adv)

        strategy_input = final_advs if args.strategy.startswith("adv") else final_probs
        selected_final = select_indices(args.strategy, strategy_input, 1, 1)[0]
        sample_id_pool.append([selected_final for _ in range(step_beam_size)])

        whole_traj = previous_steps_list[selected_final] + "\n" + final_responses[selected_final]
        all_beams_traj = [previous_steps_list[beam] + "\n" + final_responses[beam] for beam in range(step_beam_size)]

        result = {
            'id': i,
            'question': instruction,
            'ground_truth': ground_truth,
            'response': whole_traj,
            'response_all_beams': all_beams_traj
        }

        if args.record_process:
            result.update({
                'foresight_steps': num_foresight,
                'traj_pool': traj_pool,
                'step_pool': step_pool,
                'prob_pool': prob_pool,
                'adv_pool': adv_pool,
                'step_prob': step_prob,
                'sample_id_pool': sample_id_pool
            })

        save_result(args.output_path, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='math')
    parser.add_argument('--model_id', type=str, default='llama3.1')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--step_beam_size', type=int, default=4)
    parser.add_argument('--num_rollout', type=int, default=4)
    parser.add_argument('--num_foresight', type=int, default=4)
    parser.add_argument('--record_process', type=bool, default=True)
    parser.add_argument('--strategy', type=str, default='foresight_sampling')
    parser.add_argument('--mode', type=str, default='no_light')

    args = parser.parse_args()
    main(args)
