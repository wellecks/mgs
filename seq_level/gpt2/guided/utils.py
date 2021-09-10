import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import editdistance
import torch
import numpy as np
import hashlib
from copy import deepcopy

from nltk import ngrams
from collections import Counter
from seq_level.gpt2.utils import generate_batch


def task_distance(
        target_trim, model_trim, outputs, score_model,
        kind='edit',
        eos_id=None,
        average=True,
):
    if kind == 'edit':
        edits = []
        for actual_, predicted_ in zip(target_trim, model_trim):
            edit_dist = editdistance.eval(actual_, predicted_)
            edit_dist = min(edit_dist / len(actual_), 1)

            edits.append(edit_dist)
        if average:
            distance = sum(edits) / len(edits)
        else:
            distance = torch.tensor(edits, device=outputs.device, dtype=torch.float)
    elif kind == 'nonterm':
        diffs = []
        for actual_, predicted_ in zip(target_trim, model_trim):
            if eos_id in predicted_:
                diffs.append(0.0)
            else:
                diffs.append(1.0)
        if average:
            distance = sum(diffs) / len(diffs)
        else:
            distance = diffs
    elif kind == 'repeat-4':
        distances = []
        for actual_, predicted_ in zip(target_trim, model_trim):
            if len(predicted_) >= 4:
                ngs = [ng for ng in ngrams(predicted_, 4)]
                counter = Counter(ngs)
                distances.append(1.0 - (1.0 * len(counter) / max(len(ngs), 1)))
            else:
                distances.append(1.0)
        if average:
            distance = np.mean(distances)
        else:
            distance = torch.tensor(distances, device=outputs.device)
    elif kind == 'lm':
        score_model.eval()
        with torch.no_grad():
            # Compute log probs using the score model
            log_probs = torch.log_softmax(score_model(outputs)[0], -1)
            log_probs = log_probs[:, :-1, :].gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)

            sequence_mask = (outputs[:, 1:].eq(eos_id).cumsum(1).cumsum(1) <= 1).float()
            # Compute distance
            distances = -(log_probs * sequence_mask).sum(1)
            if average:
                distance = distances.mean().item()
            else:
                distance = distances
    else:
        raise NotImplementedError(kind)
    return distance


def max_length(target, eos_token_id, args):
    target_ = target.tolist()[0]  # longest sequence in the batch
    if eos_token_id in target_[args.context_length:]:
        max_length = int(
            args.decode_len_multiplier *
            (target_.index(eos_token_id) + 2)
        )
    else:
        max_length = int(args.decode_len_multiplier * (target.size(1) + 2))
    max_length = max(1, max_length)
    max_length = min(args.decode_max_length, max_length)
    return max_length


def decode_and_distance(model, tokenizer, batch, score_model, max_len, device, args, average_distance=True):
    model.eval()
    _, _, bpes, texts, outputs = generate_batch(
        model, tokenizer, batch, args.context_length, device,
        max_length=max_len,
        decoder=args.train_decoder,
        fixed_length=args.fixed_length
    )
    bpes = [b[args.context_length:] for b in bpes]  # original has prefix
    model.train()

    prefix_, target_trim, model_curr_trim = trim(
        bpes, batch, args.context_length, tokenizer.eos_token_id
    )

    distance = task_distance(
        target_trim, model_curr_trim, outputs, score_model, args.ggs_metric,
        eos_id=tokenizer.eos_token_id,
        average=average_distance,
    )
    return bpes, outputs, distance


def trim(decoded, target, context_length, eos_id):
    prefix = target[:, :context_length].tolist()
    cont_target = target[:, context_length:]
    target_trim = []
    model_trim = []
    if not isinstance(decoded, list):
        decoded = decoded.tolist()
    if not isinstance(cont_target, list):
        cont_target = cont_target.tolist()

    for data_cont, model_cont in zip(cont_target, decoded):
        if eos_id in data_cont:
            data_cont_ = data_cont[:data_cont.index(eos_id) + 1]
        else:
            data_cont_ = data_cont
        if eos_id in model_cont:
            model_cont_ = model_cont[:model_cont.index(eos_id) + 1]
        else:
            model_cont_ = model_cont
        target_trim.append(data_cont_)
        model_trim.append(model_cont_)
    return prefix, target_trim, model_trim


def mle_grad(model, inp, target, pad, clip_grad_norm=1.0):
    model.zero_grad()

    inp_ = inp.clone()
    inp_[inp == pad] = 0

    target_ = target.clone()
    target_[target == pad] = 0

    inp_mask = inp.ne(pad).float()
    target_mask = target.ne(pad).float()

    model_output = model(inp_, attention_mask=inp_mask)
    logits = model_output[0]
    lprobs = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(
        lprobs.view(-1, lprobs.size(-1)),
        target_.reshape(-1),
        reduction='none'
    )
    loss = loss * target_mask.view(loss.size())
    loss_sum = loss.sum()
    ntokens = target_mask.sum()
    loss = loss_sum / ntokens
    loss.backward()

    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    return model, loss_sum


def perturb(
        model, model_with_grad, num_samples, noise, noise_scale,
        zero_dist_only=False, mle_dist_only=False, include_mle_gradient=False,
):
    models = []
    log_rhos = []
    noise_magnitudes = []  # diagnostic metric
    # If we are including mle_gradients, we will sample num_directions + 1 samples.
    # The extra sample will correspond to MLE gradient.
    n = num_samples + 1 if include_mle_gradient else num_samples

    for i in range(n):
        model_ = deepcopy(model)
        noise_mag = 0
        eps_eps = 0
        eps_nabla = 0
        nabla_nabla = 0

        for param, (name, param_with_grad) in zip(model_.parameters(), model_with_grad.named_parameters()):
            g = -param_with_grad.grad.data
            # Generate the noise
            if include_mle_gradient and i == 0:
                noise = 0

            if noise_scale == 'uniform':
                noise_ = noise * torch.randn_like(param.data) * (g.abs().sum() / g.numel())
            else:
                noise_ = noise * torch.randn_like(param.data)

            # Choose the mixture component (assume 0.5 mixture proportion)
            if zero_dist_only:
                epsilon = noise_
            elif (include_mle_gradient and i == 0) or mle_dist_only:
                epsilon = g + noise_
            else:
                if i % 2 == 0:
                    epsilon = g + noise_
                else:
                    epsilon = noise_

            noise_mag += torch.sqrt((noise_ ** 2).sum()).item()

            eps_eps += (epsilon.data.view(-1) * epsilon.data.view(-1)).sum()
            eps_nabla += (g.view(-1) * epsilon.data.view(-1)).sum()
            nabla_nabla += (g.view(-1) * g.view(-1)).sum()
            param.data = param.data + epsilon

        noise_magnitudes.append(noise_mag)

        q = (0.5 * torch.exp(-0.5 * eps_eps) + 0.5 * torch.exp(-0.5 * eps_eps + eps_nabla + - 0.5 * nabla_nabla))
        log_rhos.append(torch.log(q))
        models.append(model_)
    log_rhos = torch.stack(log_rhos).cpu()
    return models, log_rhos, noise_magnitudes


def parameter_weighted_average(model, perturbed_models, log_weights):
    update_directions = {}
    for name, param in model.named_parameters():
        epsilons = []
        for i, model_ in enumerate(perturbed_models):
            epsilon = (model_.state_dict()[name] - param).data
            epsilon = torch.exp(log_weights[i]) * epsilon
            epsilons.append(epsilon)
        averaged = torch.stack(epsilons, 0).sum(0)
        update_directions[name] = averaged.data
    return update_directions


def compute_weight(distance, perturbed_distances, log_rhos, beta):
    ws = torch.tensor([
            beta * (distance - perturbed_distance)
                for perturbed_distance in perturbed_distances])\
            .clamp(max=1e16)
    ws = ws - log_rhos
    log_ws = torch.log_softmax(ws, 0)
    return log_ws

def get_model_id(model):
    return hashlib.sha1(next(model.parameters()).detach().cpu().numpy()).hexdigest()


def update(model, update_directions, optimizer, clip_grad_norm=1.0):
    optimizer.zero_grad()
    for name, param in model.named_parameters():
        param.grad = -update_directions[name]

    if clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    optimizer.step()
    return get_model_id(model)



def load_model(args, device):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir)
    if args.model_load_dir:
        model = GPT2Wrapper.from_pretrained(args.model_load_dir, cache_dir=args.cache_dir)
    else:
        model = GPT2Wrapper.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    model.to(device)
    return model, tokenizer


class GPT2Wrapper(GPT2LMHeadModel):
    pass

class RNG(object):
    def __init__(self, rng_state=None, device=None):
        self.rng = torch.Generator(device=device)
        self.rng_state = rng_state
        self.current_rng_state = self.rng.seed()

    def __enter__(self):
        if self.rng_state is not None:
            self.rng.manual_seed(self.rng_state)
        else:
            self.rng.manual_seed(self.current_rng_state)
        return self.rng, self.rng.initial_seed()

    def __exit__(self, type, value, traceback):
        self.rng.manual_seed(self.current_rng_state)
        return True