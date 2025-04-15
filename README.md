<h1 align="center">
<img src="./assets/logo.png" width="100" alt="Genius" />
<br>
Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning
</h1>



<p align="center">
  <a href="https://arxiv.org/abs/2504.08672"><b>[📜 Paper]</b></a> •
  <a href="https://huggingface.co/collections/xufangzhi/genius-67b6be9b65ac174668739270"><b>[🤗 HF Models]</b></a> •
  <a href="https://github.com/xufangzhi/Genius"><b>[🐱 GitHub]</b></a>
  
</p>


<p align="center">
Repo for "<a href="https://arxiv.org/abs/2311.09278" target="_blank">Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning</a>"
</p>


## 🔥 News

- [2025/02/16] 🔥🔥🔥 Genius is under review !


## 🔍 Core Implementation

The core implementation of Advantage Calibrated Optimization (ACO) loss function is presented below:

```python
import torch
import torch.nn.functional as F
from typing import Tuple

def aco_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    chosen_weights: torch.FloatTensor,
    rejected_weights: torch.FloatTensor,
    average_weights: torch.FloatTensor,
    beta: float,
    alpha: float = 1.0,
    reference_free: bool = False
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Adaptive Comparison Optimization (ACO) Loss.

    Args:
        policy_chosen_logps (FloatTensor): Log-probs from policy model for chosen responses. Shape: (batch,)
        policy_rejected_logps (FloatTensor): Log-probs from policy model for rejected responses. Shape: (batch,)
        reference_chosen_logps (FloatTensor): Log-probs from reference model for chosen responses. Shape: (batch,)
        reference_rejected_logps (FloatTensor): Log-probs from reference model for rejected responses. Shape: (batch,)
        chosen_weights (FloatTensor): Preference weights for chosen responses.
        rejected_weights (FloatTensor): Preference weights for rejected responses.
        beta (float): Temperature parameter to scale loss sharpness.
        alpha (float): Relaxation scaling factor in adaptive weighting.
        reference_free (bool): If True, ignores reference model by assuming uniform logits.

    Returns:
        Tuple of (losses, chosen_rewards, rejected_rewards), each of shape (batch,)
    """

    if reference_free:
        reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
        reference_rejected_logps = torch.zeros_like(policy_rejected_logps)

    # Log-ratio between policy and reference
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    # Adaptive weighting based on preference gap
    adjustment_factor = torch.max(
        torch.tensor(1.0, device=chosen_logratios.device),
        torch.exp(-(rejected_weights - chosen_weights) / alpha)
    )
    rejected_logratios_weighted = adjustment_factor * rejected_logratios

    logits = chosen_logratios - rejected_logratios_weighted

    # Final ACO loss
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    return losses, chosen_rewards, rejected_rewards
```


## 🚀 Quick Start

To implement the *foresight re-sampling*, you can first use the following command to conduct the stepwise self-exploration:

```python
python foresight-sampling/foresight_sampling_with_vllm.py \
  --model_path <path_to_your_model> \
  --data_path <path_to_your_queries> \
  --output_path <path_to_your_output_file> \
  --step_beam_size 2 \
  --num_rollout 4 \
  --num_foresight 4
```

After exploration, you can exploit and construct the preference data through resampling technique:

```python
python foresight-sampling/construct_foresight_preference_data.py \
  --input_path <path_to_the_exploration_outputs> \
  --output_path <path_to_output_data_file>
```

To train the policy model with Advantage Calibrated Optimization (ACO) loss function, please refer to the following command:

```bash
# execute
bash ./aco-training/scripts/aco_train_with_accelerate.sh
```


## 📒 Note
This work is still under review. We are open-sourcing the model weights and the code. The training scripts are based on [open-instruct](https://github.com/allenai/open-instruct) and the evaluation is supported and accelerated by [vLLM](https://docs.vllm.ai/en/latest/) engine.


## Citation
If you find it helpful, please kindly cite our paper as well as the inference-time decoding algorithm $\phi$-Decoding:

```
@article{xu2025genius,
  title={Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning},
  author={Xu, Fangzhi and Yan, Hang and Ma, Chang and Zhao, Haiteng and Sun, Qiushi and Cheng, Kanzhi and He, Junxian and Liu, Jun and Wu, Zhiyong},
  journal={arXiv preprint arXiv:2504.08672},
  year={2025}
}
```

```
@article{xu2025phi,
  title={$\phi$-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation},
  author={Xu, Fangzhi and Yan, Hang and Ma, Chang and Zhao, Haiteng and Liu, Jun and Lin, Qika and Wu, Zhiyong},
  journal={arXiv preprint arXiv:2503.13288},
  year={2025}
}
```