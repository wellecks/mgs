import torch


class AvgBaseline(object):
    def __init__(self, alpha=0.95):
        self.val = 0
        self.alpha = alpha

    def update(self, distances):
        self.val = (1.0 - self.alpha)*self.val + self.alpha*distances.mean().item()

    def __call__(self, *args, **kwargs):
        return self.val


def pg_loss(model, batch, baseline, candidates, distances, pad_token_id, eos_token_id, num_candidates, normalize_distance):
    log_probs = torch.log_softmax(model(candidates)[0], -1)
    log_probs = log_probs[:, :-1, :].gather(2, candidates[:, 1:].unsqueeze(-1)).squeeze(-1)
    sequence_mask = (candidates[:, 1:].eq(eos_token_id).cumsum(1).cumsum(1) <= 1).float()

    # Reshape to B x num_candidates x -1
    log_probs = log_probs.view(batch.size(0), num_candidates, -1)
    sequence_mask = sequence_mask.view(batch.size(0), num_candidates, -1)

    distances = distances.view(batch.size(0), num_candidates)

    # For large-magnitude distances, e.g. LM
    if normalize_distance:
        distances = distances / distances.max()

    reward = -distances

    baseline.update(reward)
    b = baseline(model, batch)
    adv = reward - b

    scores = (log_probs*sequence_mask).sum(2) / sequence_mask.sum(2).clamp(min=1)
    loss = -(adv*scores).sum(1).mean()

    metrics = dict(
        loss=loss.item()
    )
    return loss, metrics