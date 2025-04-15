import json
import numpy as np
import random
import argparse
from typing import List, Tuple


def load_json_lines(path: str) -> List[dict]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def save_jsonl(data: List[dict], path: str):
    with open(path, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')


def resampling_data(input_path: str) -> List[dict]:
    data = load_json_lines(input_path)
    organized_step_preference_data = []

    for entry in data:
        traj_pool = entry.get('traj_pool')
        logp_pool = entry.get('prob_pool')
        step_prob = entry.get('step_prob')
        adv_pool = entry.get('adv_pool')
        question = entry.get('question')

        if not (traj_pool and logp_pool and step_prob and adv_pool and question):
            continue
        if not (len(traj_pool) == len(logp_pool) == len(step_prob) == len(adv_pool)):
            continue

        for j in range(len(traj_pool)):
            try:
                step_candidates = [
                    (j, k, logp_pool[j][k], adv_pool[j][k], step_prob[j][k])
                    for k in range(len(traj_pool[j]))
                ]
            except (IndexError, TypeError):
                continue

            if len(step_candidates) < 2:
                continue

            step_candidates.sort(key=lambda x: x[2], reverse=True)

            try:
                k = random.choice(range(1, len(step_candidates)))
                chosen = traj_pool[step_candidates[0][0]][step_candidates[0][1]]
                rejected = traj_pool[step_candidates[k][0]][step_candidates[k][1]]

                organized_step_preference_data.append({
                    "prompt": question,
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                    "chosen_weights": step_candidates[0][3],
                    "rejected_weights": step_candidates[k][3],
                    "chosen_average_weights": np.mean([c[3] for c in step_candidates]),
                })
            except Exception:
                continue

    return organized_step_preference_data


def main():
    parser = argparse.ArgumentParser(description="Process step preference data")
    parser.add_argument("--input_path", type=str, required=True, help="<path_to_input_file>")
    parser.add_argument("--output_path", type=str, required=True, help="<path_to_output_file>")
    args = parser.parse_args()

    random.seed(42)

    step_pref_data = resampling_data(args.input_path)
    save_jsonl(step_pref_data, args.output_path)

    print(f"Step preference examples saved: {len(step_pref_data)}")


if __name__ == "__main__":
    main()
