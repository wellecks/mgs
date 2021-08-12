import torch
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import GPT2Config
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler
import seq_level.gpt2.guided.utils as ggs_utils
import seq_level.gpt2.utils as utils
import seq_level.gpt2.train as train_utils
import os
from seq_level.gpt2.guided.metrics import GuidedMetrics


def aggregate_score_data(train_dataset, model, score_model, tokenizer, args, count, device):
    """ This method does a forward pass over the original model and 
        the perturbed model to compute the yo_i, the decoded output corresponding
        to the input x using the original model, and yp_i, the decoding output corresponding
        to the perturbed model. 
        The perturbations are sampled from $\Deta \sim Q_{MGS}$.

        It returns a set of tuples with each tuple of the form (x_i, y_i, yo_i, yp_i, \Delta).
    """

    print('=' * 150)
    print('Data aggregation.\n')

    buffer = []

    for i in range(count * args.aggregation_size, (count + 1) * args.aggregation_size):

        batch = train_dataset[i]
        batch = batch.squeeze(0)
        batch = batch.to(device)
        assert batch.size(1) >= args.context_length + 1

        inp, target = batch[:, :-1], batch[:, 1:]
        max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)

        model.eval()
        _, cur_decodings, cur_distances = ggs_utils.decode_and_distance(
            model, tokenizer, batch, score_model, max_length, device, args
        )

        # Get the current MLE gradients
        model.train()
        cur_model = deepcopy(model)
        model_with_grad, _ = ggs_utils.mle_grad(
            cur_model, inp, target, tokenizer.pad_token_id, args.max_grad_norm
        )

        per_model = deepcopy(model)

        for param, (name, param_with_grad) in zip(per_model.parameters(), model_with_grad.named_parameters()):

            gradient = -param_with_grad.grad.data

            if args.noise_scale == 'uniform':
                noise_ = args.noise * torch.randn_like(param.data) * (gradient.abs().sum() / gradient.numel())
            else:
                noise_ = args.noise * torch.randn_like(param.data)

            if i % 2 == 0:
                epsilon = noise_ + gradient
            else:
                epsilon = noise_

            param.data = param.data + epsilon

        _, per_decodings, per_distances = ggs_utils.decode_and_distance(
            per_model, tokenizer, batch, score_model, max_length, device, args
        )

        buffer.append((batch, cur_decodings, per_decodings, cur_distances, per_distances, per_model))

    return buffer


def train_score_network(buffers, model, phi_network, phi_optimizer, args):
    """ This method takes in the scoring data (B) and learns a parameterized scoring model (S) to 
        mimic the original cost function (C) by minimizing the L2 loss.
        $C$ is defined as
            C(\theta) = 1/|B| \sum_{i=1}^{B} c(y_i, F(x_i, \theta))

        The scoring function optimizes the following equation. 
            min_{W,b} \sum_{(x_i, y_i, yo_i, yp_i, \Delta_i) in B} || S(x_i, \theta, \Delta; W, b) - (C(\theta) - C(\theta + \Delta))||^2
            where S(x_i, \theta, \Delta; W, b) = W^T[R(x;\theta) - R(x;\theta+\Delta)] + b
    """

    print('=' * 150)
    print('Start training the score network.\n')

    train_loss = 0.
    num_docs = 0

    for step, (batch, cur_decodings, per_decodings, cur_distances, per_distances, per_model) in enumerate(buffers):

        model.eval()
        per_model.eval()

        context_batch = utils.wrap_context_batch(batch, args)
        context_batch = context_batch.to(batch.device)

        cur_emb = model(context_batch, output_hidden_states=True).hidden_states[-1][:, -1, :].detach()
        per_emb = per_model(context_batch, output_hidden_states=True).hidden_states[-1][:, -1, :].detach()

        phi_network.train()
        phi_optimizer.zero_grad()

        outputs = phi_network(cur_emb - per_emb)

        loss = F.mse_loss(
            outputs,
            (per_distances - cur_distances).view(-1, 1),
            reduction='none'
        )
        loss = loss.sum()

        loss.backward()
        phi_optimizer.step()

        train_loss += loss.item()
        num_docs += batch.size(0)

        if step % 5 == 0 and step > 0:
            cur_loss = train_loss / num_docs
            print('Step: %d, Loss: %.2f' % (step, cur_loss))


def score_function(batch, model):
    """ This methods takes in the batch and the model and
         returns the estimated scores for batch input according to the model.
    """
    model.eval()
    embed = model(batch, output_hidden_states=True).hidden_states[-1][:, -1, :]
    return embed


def MGS(batch, model, score_model, phi_network, tokenizer, args, device, metrics, optimizer, efficient=False):
    """ MGS algorithm parameterized to work in original as well as efficient mode.
    """
    inp, target = batch[:, :-1].to(device), batch[:, 1:].to(device)

    # -- Decode with current model (required for computing the 'weights' later).
    max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)

    decoded = defaultdict(list)
    if efficient:
        distance_curr = score_function(batch, model)
    else:
        bpes_curr, _, distance_curr = ggs_utils.decode_and_distance(
            model, tokenizer, batch, score_model, max_length, device, args
        )
        for i, idxs in enumerate(bpes_curr):
            decoded[i].append(
                tokenizer.decode(idxs)
            )

    # -- Obtain MLE gradients
    model_ = deepcopy(model)
    model_with_grad, mle_loss = ggs_utils.mle_grad(
        model_, inp, target, tokenizer.pad_token_id, args.max_grad_norm
    )

    # -- Perturb
    perturbed_models, log_rhos, noise_magnitudes = ggs_utils.perturb(
        model, model_with_grad, args.ggs_num_samples, args.ggs_noise,
        noise_scale=args.noise_scale,
        zero_dist_only=args.zero_dist_only,
        mle_dist_only=args.mle_dist_only,
        include_mle_gradient=args.include_mle_gradient,
    )

    # -- Decode with perturbed models and compute task metric
    distances = []
    for p_model in perturbed_models:
        if efficient:
            distance = score_function(batch, p_model)
        else:
            bpes_, _, distance = ggs_utils.decode_and_distance(
                p_model, tokenizer, batch, score_model, max_length, device, args
            )
            for i, idxs in enumerate(bpes_):
                decoded[i].append(
                    tokenizer.decode(idxs)
                )
        distances.append(distance)

    # -- Compute weights
    # Kushal: please revise phi_network(distance_curr - distances), where the score_function's output is embedding.

    log_weights = ggs_utils.compute_weight(distance_curr, distances, log_rhos, args.ggs_beta)

    # -- Compute weighted average of the directions
    update_directions = ggs_utils.parameter_weighted_average(
        model, perturbed_models, log_weights
    )

    # -- Perform update
    ggs_utils.update(model, update_directions, optimizer, args.max_grad_norm)

    # -- Record statistics
    metrics.step(
        mle_loss.item(), distance_curr, bpes_curr, target, args.context_length,
        tokenizer.pad_token_id, tokenizer.eos_token_id,
        model_with_grad, update_directions, log_rhos, log_weights,
        noise_magnitudes, distances
    )

    return decoded


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()

        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        output = self.fc(x)
        return output


def train(model, tokenizer, dataset_tensor_dict, args, device):
    model.train()
    train_sampler = RandomSampler(dataset_tensor_dict['train'])
    train_dataloader = DataLoader(
        dataset_tensor_dict['train'],
        sampler=train_sampler,
        batch_size=1
    )

    optimizer, scheduler = utils.get_optimizer(model, len(train_dataloader), args)

    config = GPT2Config()
    phi_network = MLP(input_size=config.hidden_size).to(device)
    phi_optimizer = optim.Adam(phi_network.parameters(), lr=0.001)

    best_val_loss = 10000
    patience = args.patience
    stats_cache = defaultdict(list)
    count = 0

    score_model = deepcopy(model)

    for epoch_number in range(args.num_train_epochs):

        metrics = GuidedMetrics()

        buffers = aggregate_score_data(dataset_tensor_dict['train'], model, score_model, tokenizer, args, count, device)
        train_score_network(buffers, model, phi_network, phi_optimizer, args)
        count += 1

        for step, batch in enumerate(train_dataloader):

            batch = batch.squeeze(0)
            assert batch.size(1) >= args.context_length + 1

            decoded = MGS(batch=batch,
                          model=model,
                          score_model=score_model,
                          phi_network=phi_network,
                          tokenizer=tokenizer,
                          args=args,
                          device=device,
                          metrics=metrics,
                          optimizer=optimizer,
                          efficient=False
                          )

            if step % args.print_every == 0:
                metrics_ = metrics.normalize('train')
                metrics.reset()
                print("Epoch %d   \t Step %d   \tmle: %.3f\tdist: %.3f\tnon_term: %.3E\tmle_weight: %.3E" % (
                    epoch_number,
                    step,
                    metrics_['train/mle_loss'],
                    metrics_['train/distance'],
                    metrics_['train/non_term'],
                    metrics_['train/model/mle_weight']
                ))

                utils.log_tensorboard(metrics_, args.log_step)
                stats_cache['train/mle_loss'].append(metrics_['train/mle_loss'])
                stats_cache['train/distance'].append(metrics_['train/distance'])

            if args.log_step % args.valid_every == 0 or step == 0:
                val_loss, val_metrics, decodings = train_utils.valid_iteration(
                    dataset_tensor_dict['valid'], model, score_model, train_utils.get_mle_loss, tokenizer, device,
                    context_length=args.eval_context_length,
                    num_decodings=250,
                    args=args
                )
                if args.save_all == 1:
                    save_dir = os.path.join(args.save_base_dir, str(args.log_step))
                    os.makedirs(save_dir)
                    utils.save(model, save_dir)
                if val_metrics['valid/distance-%s' % args.ggs_metric] < best_val_loss:
                    print('Best distance achieved (%.5f)' % val_metrics['valid/distance-%s' % args.ggs_metric])
                    if not args.no_checkpoint:
                        utils.save(model, args.save_base_dir)
                    utils.save_metrics(val_metrics, args)
                    utils.save_decodings(decodings, args)
                    best_val_loss = val_metrics['valid/distance-%s' % args.ggs_metric]
                    patience = args.patience
                else:
                    patience = patience - 1

                model.train()
                utils.log_tensorboard(val_metrics, args.log_step)

                if patience == 0:
                    return

            if args.max_train_steps > 0 and args.log_step >= args.max_train_steps:
                return

            args.log_step += 1
            torch.cuda.empty_cache()


def add_args(parser):
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument("--ggs-noise", type=float, default=1.0)
    parser.add_argument("--ggs-beta", type=float, default=100.0)
    parser.add_argument("--ggs-num-samples", type=int, default=4)
    parser.add_argument("--decode-len-multiplier", type=float, default=1.3)
    parser.add_argument(
        "--ggs-metric",
        choices=['edit', 'lm'],
        default='lm'
    )
    parser.add_argument(
        "--bleu-smoothing",
        choices=['method%d' % i for i in range(1, 8)],
        default='method2'
    )
    parser.add_argument(
        "--noise-scale", choices=['uniform', 'constant'], default='uniform'
    )
    parser.add_argument(
        "--zero-dist-only", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--mle-dist-only", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--save-all", type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        "--max-train-steps", type=int, default=-1
    )
    parser.add_argument('--include-mle-gradient', action='store_true')

    parser.add_argument(
        "--aggregation_size", type=int, default=100
    )

    return parser
