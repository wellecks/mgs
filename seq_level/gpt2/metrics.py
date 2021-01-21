from collections import defaultdict
from seq_level.gpt2.guided.utils import task_distance
from seq_level.gpt2.guided.metrics import (
    len_stats, ngram_metrics, nonterm_metrics
)
import torch
import numpy as np


class GenerationMetrics(object):
    def __init__(self, distances=('edit', 'nonterm', 'repeat-4', 'lm')):
        self.distances = distances
        self._stats_cache = defaultdict(list)

    def step(self, preds_trim, targets_trim, outputs, score_model, eos_token, context_length):
        with torch.no_grad():
            for distance in self.distances:
                dist = task_distance(
                    target_trim=targets_trim,
                    model_trim=preds_trim,
                    outputs=outputs,
                    score_model=score_model,
                    kind=distance,
                    eos_id=eos_token,
                )
                self._stats_cache['distance-%s' % distance].append(dist)

            len_diffs, target_lens, model_lens = len_stats(
                preds_trim, targets_trim, context_length, eos_token
            )
            self._stats_cache['len_diff'].extend(len_diffs)
            self._stats_cache['target/len'].extend(target_lens)
            self._stats_cache['model/len'].extend(model_lens)
            self._stats_cache['non_term'].extend(
                nonterm_metrics(preds_trim, eos_token)
            )
            for k, vs in ngram_metrics(targets_trim, eos_token).items():
                self._stats_cache['target/%s' % k].extend(vs)
            for k, vs in ngram_metrics(preds_trim, eos_token).items():
                self._stats_cache['model/%s' % k].extend(vs)

    def normalize(self, prefix='valid'):
        output = {}
        for key in self._stats_cache:
            output['%s/%s' % (prefix, key)] = np.mean(self._stats_cache[key])
        return output
