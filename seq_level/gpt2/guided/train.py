import torch
from copy import deepcopy
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler
import seq_level.gpt2.guided.utils as ggs_utils
import seq_level.gpt2.utils as utils
import seq_level.gpt2.train as train_utils
import os
from seq_level.gpt2.guided.metrics import GuidedMetrics


def aggregte_scoring_data(batch, model):
    """ This method does a forward pass over the original model and 
        the perturbed model to compute the yo_i, the decoded output corresponding
        to the input x using the original model, and yp_i, the decoding output corresponding
        to the perturbed model. 
        The pertubrations are sampled from $\Deta \sim Q_{MGS}$. 

        It returns a set of tuples with each tuple of the form (x_i, y_i, yo_i, yp_i, \Delta).
    """
    pass

def train_scoring_function(B, S, R, scoring_opt, scoring_scheduler, c):
    """ This method takes in the scoring data (B) and learns a parameterized scoring model (S) to 
        mimic the original cost function (C) by minimizing the L2 loss.
        $C$ is defined as
            C(\theta) = 1/|B| \sum_{i=1}^{B} c(y_i, F(x_i, \theta))

        The scoring function optimizes the following equation. 
            min_{W,b} \sum_{(x_i, y_i, yo_i, yp_i, \Delta_i) in B} || S(x_i, \theta, \Delta; W, b) - (C(\theta) - C(\theta + \Delta))||^2
            where S(x_i, \theta, \Delta; W, b) = W^T[R(x;\theta) - R(x;\theta+\Delta)] + b
    """
    pass

def MGS(batch, scoring_function, model):
    """ MGS algorithm parameterized to work in original as well as efficient mode 
    """
    pass


def train(model, tokenizer, dataset_tensor_dict, args, device):
    model.train()
    train_sampler = RandomSampler(dataset_tensor_dict['train'])
    train_dataloader = DataLoader(
        dataset_tensor_dict['train'],
        sampler=train_sampler,
        batch_size=1  # batching is handled by the Line Dataset class
    )

    optimizer, scheduler = utils.get_optimizer(model, len(train_dataloader), args)

    best_val_loss = 10000
    patience = args.patience
    stats_cache = defaultdict(list)

    score_model = deepcopy(model)
    for epoch_number in range(args.num_train_epochs):
        metrics = GuidedMetrics()
        # Data Aggregation. 
        for step, batch in enumerate(train_dataloader):
            # B_ = {(x_i, yt_i, yg_i, yp_i, \Delta) | (x_i, y_i) \in batch} 
            B_ = aggregate_new_data(batch, model)
            B = B.add(B_)
        
        for (x_i, yt_i, yg_i, yp_i, delta) in B:
            # Le

            batch = batch.squeeze(0)
            assert batch.size(1) >= args.context_length + 1
            inp, target = batch[:, :-1].to(device), batch[:, 1:].to(device)

            # -- Decode with current model (required for computing the 'weights' later).
            max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)
            bpes_curr, distance_curr = ggs_utils.decode_and_distance(
                model, tokenizer, batch, score_model, max_length, device, args
            )
            decoded = defaultdict(list)
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
                bpes_, distance = ggs_utils.decode_and_distance(
                    p_model, tokenizer, batch, score_model, max_length, device, args
                )
                distances.append(distance)
                for i, idxs in enumerate(bpes_):
                    decoded[i].append(
                        tokenizer.decode(idxs)
                    )

            # -- Compute weights
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
    return parser