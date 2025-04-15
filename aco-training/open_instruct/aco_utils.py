'''
ACO / DPO utils
Adapted from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py
'''
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, List, Union, Tuple
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


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
    Compute the ACO (Adaptive Comparison Optimization) loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps (FloatTensor): Log-probabilities from the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps (FloatTensor): Log-probabilities from the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps (FloatTensor): Log-probabilities from the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps (FloatTensor): Log-probabilities from the reference model for the rejected responses. Shape: (batch_size,)
        chosen_weights (FloatTensor): Weights for chosen responses, typically representing preference confidence.
        rejected_weights (FloatTensor): Weights for rejected responses, typically representing preference confidence.
        average_weights (FloatTensor): Average weights across comparisons (not used in this implementation).
        beta (float): Temperature parameter; scales the sharpness of the preference.
        alpha (float): Hyper-parameter to control the scale of the relaxation term.
        reference_free (bool): If True, the reference model is assumed uniform (i.e., ignored).

    Returns:
        Tuple[FloatTensor, FloatTensor, FloatTensor]: A tuple containing:
            - losses: ACO loss for each example in the batch.
            - chosen_rewards: Estimated reward signal for chosen responses.
            - rejected_rewards: Estimated reward signal for rejected responses.
    """

    if reference_free:
        reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
        reference_rejected_logps = torch.zeros_like(policy_rejected_logps)

    # Compute log-ratios between policy and reference
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    # Adaptive weighting of rejected log-ratios
    adjustment_factor = torch.max(
        torch.tensor(1.0, device=chosen_logratios.device),
        torch.exp(-(rejected_weights - chosen_weights) / alpha)
    )
    rejected_logratios_weighted = adjustment_factor * rejected_logratios

    logits = chosen_logratios - rejected_logratios_weighted

    # Compute loss
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    return losses, chosen_rewards, rejected_rewards



def simpo_loss(average_policy_chosen_logps: torch.FloatTensor,
             average_policy_rejected_logps: torch.FloatTensor,
             average_reference_chosen_logps: torch.FloatTensor,
             average_reference_rejected_logps: torch.FloatTensor,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             alpha: float,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the RPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = average_policy_chosen_logps - average_policy_rejected_logps
    # ref_logratios = reference_chosen_logps - reference_rejected_logps


    logits = pi_logratios
    gamma = 1
    losses = -F.logsigmoid(beta * logits - gamma)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def ipo_loss(average_policy_chosen_logps: torch.FloatTensor,
             average_policy_rejected_logps: torch.FloatTensor,
             average_reference_chosen_logps: torch.FloatTensor,
             average_reference_rejected_logps: torch.FloatTensor,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             alpha: float,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the RPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios

    tau = 0.5
    losses = (logits - 1 / (2*tau) ) ** 2
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards



def ropo_loss(average_policy_chosen_logps: torch.FloatTensor,
             average_policy_rejected_logps: torch.FloatTensor,
             average_reference_chosen_logps: torch.FloatTensor,
             average_reference_rejected_logps: torch.FloatTensor,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             alpha: float,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the RPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    dpo_logits = pi_logratios - ref_logratios
    ropo_logits = ref_logratios - pi_logratios

    alpha, gamma = 0.2, 0.1
    
    dpo_losses = - gamma * F.logsigmoid(beta * dpo_logits)
    ropo_losses = alpha * F.sigmoid(beta * ropo_logits)

    losses = ropo_losses + dpo_losses
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards




def rpo_loss(average_policy_chosen_logps: torch.FloatTensor,
             average_policy_rejected_logps: torch.FloatTensor,
             average_reference_chosen_logps: torch.FloatTensor,
             average_reference_rejected_logps: torch.FloatTensor,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             alpha: float,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the RPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    beta, alpha = 0.5, 1
    losses = -F.logsigmoid(beta * logits) - alpha * average_policy_chosen_logps
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards



def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), (per_token_logps * loss_mask).sum(-1)

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]], has_weights=False) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and "weights" not in k and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

    for k in batch:
        if k.startswith('rejected') and "weights" not in k and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    if has_weights:
        concatenated_batch['chosen_weights'] = batch['chosen_weights']
        concatenated_batch['rejected_weights'] = batch['rejected_weights']
        try:
            concatenated_batch['chosen_average_weights'] = batch['chosen_average_weights']
        except:
            concatenated_batch['chosen_average_weights'] = batch['chosen_weights']
    return concatenated_batch


def concatenated_forward(model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], has_weights=False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch, has_weights)
        all_logits = model(
            input_ids=concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask']
        ).logits.to(torch.float32)
        average_logps, all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'])
        average_chosen_logps = average_logps[:batch['chosen_input_ids'].shape[0]]
        average_rejected_logps = average_logps[batch['chosen_input_ids'].shape[0]:]
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]

        if has_weights:
            return average_chosen_logps, average_rejected_logps, chosen_logps, rejected_logps, concatenated_batch['chosen_weights'], concatenated_batch['rejected_weights'], concatenated_batch['chosen_average_weights']
        else:
            return average_chosen_logps, average_rejected_logps, chosen_logps, rejected_logps, None, None, None


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)

@dataclass
class DataCollatorForSeq2SeqDPO(DataCollatorForSeq2Seq):
    """
    Alternate version of the hf DataCollatorForSeq2Seq for use with DPO.
    adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L517C1
    """
    def __call__(self, features, return_tensors=None):
        # call the original collator on chosen and rejected separately, then combine
        def filter_batch(match_string, features):
            return [
                {k.replace(match_string, ''): v for k, v in f.items() if match_string in k}
                for f in features
            ]
        chosen_features = super().__call__(
            filter_batch('chosen_', features), 
            return_tensors=return_tensors
        )
        rejected_features = super().__call__(
            filter_batch('rejected_', features),
            return_tensors=return_tensors
        )
        result = {}
        for k in chosen_features:
            result['chosen_' + k] = chosen_features[k]
        for k in rejected_features:
            result['rejected_' + k] = rejected_features[k]
        return result
