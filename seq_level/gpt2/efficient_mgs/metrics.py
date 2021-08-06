import torch
import numpy as np
from collections import defaultdict, Counter
from nltk import ngrams
import os
import json


class GuidedMetrics(object):
    def __init__(self):
        self._stats_cache = defaultdict(list)

    def step(self,
        loss, distance, decoded, target, context_length, pad_token, eos_token,
        model_with_grad=None, update_directions=None, log_rhos=None, log_weights=None,
        noise_magnitudes=None, distances=None
    ):
        with torch.no_grad():
            self._stats_cache['distance'].append(distance)
            self._stats_cache['total_loss'].append(loss)
            self._stats_cache['non_pad_tokens'].append(
                target.reshape(-1).ne(pad_token).nonzero().numel() -
                (target.size(0)*context_length)
            )
            len_diffs, target_lens, model_lens = len_stats(
                decoded, target, context_length, eos_token
            )
            self._stats_cache['len_diff'].extend(len_diffs)
            self._stats_cache['target_lens'].extend(target_lens)
            self._stats_cache['model_lens'].extend(model_lens)
            self._stats_cache['non_term'].extend(nonterm_metrics(decoded, eos_token))
            for k, vs in ngram_metrics(target, eos_token).items():
                self._stats_cache['target/%s' % k].extend(vs)
            for k, vs in ngram_metrics(decoded, eos_token).items():
                self._stats_cache['model/%s' % k].extend(vs)
            for k, v in grad_metrics(model_with_grad, update_directions).items():
                self._stats_cache['model/%s' % k].append(v)
            for k, v in weight_metrics(log_rhos, log_weights, noise_magnitudes, distances).items():
                self._stats_cache['model/%s' % k].append(v)

    def normalize(self, prefix='train', loss_normalize='num_tokens'):
        output = {}
        output['%s/distance' % prefix] = np.mean(self._stats_cache['distance'])
        if loss_normalize == 'num_tokens':
            output['%s/mle_loss' % prefix] = (
                np.sum(self._stats_cache['total_loss']) /
                np.sum(self._stats_cache['non_pad_tokens'])
            )
        else:
            output['%s/loss' % prefix] = (
                np.mean(self._stats_cache['total_loss'])
            )
        for key in ['len_diff', 'target_lens', 'model_lens', 'non_term']:
            output['%s/%s' % (prefix, key)] = np.mean(self._stats_cache[key])
        for key in self._stats_cache:
            if ('grad_cosine' in key or
                'rho' in key or
                'tilde_w' in key or
                'pct_repeat' in key or
                'highest' in key or
                'mle_weight' in key):
                output['%s/%s' % (prefix, key)] = np.mean(self._stats_cache[key])
        return output

    def add_generations(self, prefix, model_trim, target_trim, vocab):
        for i in range(len(prefix)):
            self._stats_cache['generations'].append({
                'prefix': vocab.decode_idx_seq(prefix[i]),
                'model': vocab.decode_idx_seq(model_trim[i]),
                'target': vocab.decode_idx_seq(target_trim[i]),
            })

    def save_generations(self, save_dir, train_step):
        generations = self._stats_cache['generations']
        fp = open(os.path.join(save_dir, "model_generations_%d.json" % train_step), "w")
        json.dump({'generations': generations}, fp)

    def reset(self):
        self._stats_cache = defaultdict(list)


def grad_metrics(model_with_grad, update_directions):
    if model_with_grad is None or update_directions is None:
        return {}
    similarities = []
    for name, param in model_with_grad.named_parameters():
        sim = torch.cosine_similarity(
            param.grad.data.view(1, -1),
            update_directions[name].view(1, -1)
        ).item()
        similarities.append(sim)
    avg = np.mean(similarities)
    std = np.std(similarities)
    output = {
        'avg_grad_cosine': avg,
        'std_grad_cosine': std
    }
    return output


def weight_metrics(log_rhos, log_weights, noise_magnitudes, distances):
    if log_rhos is None or log_weights is None or distances is None or noise_magnitudes is None:
        return {}
    rhos = torch.exp(log_rhos)
    weights = torch.exp(log_weights)

    argmax = weights.argmax().item()
    mle_highest = int(argmax % 2 == 0)
    zero_highest = int(argmax % 2 == 1)

    mle_weight = (weights[0] + weights[2]).item()
    output = {
        'avg_rho': rhos.mean().item(),
        'std_rho': rhos.std().item(),
        'avg_tilde_w': weights.mean().item(),
        'std_tilde_w': weights.std().item(),
        'std_distance': np.std(distances).item(),
        'avg_noise_mag': np.mean(noise_magnitudes),
        'zero_dist_highest': zero_highest,
        'mle_dist_highest': mle_highest,
        'mle_weight': mle_weight
    }

    return output


def len_stats(decoded, target, context_length, eos_token):
    if not isinstance(target, list):
        target = target.tolist()
    if not isinstance(decoded, list):
        decoded = decoded.tolist()
    cont_target = [t[context_length:] for t in target]
    diffs = []
    target_lens = []
    model_lens = []

    for data_cont, model_cont in zip(cont_target, decoded):
        if eos_token in data_cont:
            data_cont_ = data_cont[:data_cont.index(eos_token)+1]
        else:
            data_cont_ = data_cont
        if eos_token in model_cont:
            model_cont_= model_cont[:model_cont.index(eos_token)+1]
        else:
            model_cont_ = model_cont

        diff = np.abs(len(data_cont_) - len(model_cont_))
        diffs.append(diff)
        target_lens.append(len(data_cont_))
        model_lens.append(len(model_cont_))
    return diffs, target_lens, model_lens


def ngram_metrics(sequences, eos_token):
    stats = defaultdict(list)
    if not isinstance(sequences, list):
        sequences = sequences.tolist()

    for sequence in sequences:
        if eos_token in sequence:
            sequence_ = sequence[:sequence.index(eos_token)+1]
        else:
            sequence_ = sequence

        for n in [1, 4]:
            if len(sequence_) >= n:
                ngs = [ng for ng in ngrams(sequence_, n)]
                counter = Counter([ng for ng in ngrams(sequence_, n)])
                stats['pct_repeat_%dgrams' % n].append(
                    1.0 - len(counter)/max(len(ngs), 1)
                )
    return stats


def nonterm_metrics(sequences, eos_token):
    nonterm = []
    if not isinstance(sequences, list):
        sequences = sequences.tolist()
    for sequence in sequences:
        nonterm.append(float(eos_token not in sequence))
    return nonterm