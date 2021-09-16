import torch
from copy import deepcopy
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler
import seq_level.gpt2.guided.utils as ggs_utils
import seq_level.gpt2.mrt.utils as mrt_utils
import seq_level.gpt2.pg.utils as pg_utils
import seq_level.gpt2.utils as utils
import seq_level.gpt2.train as train_utils
import os
from seq_level.gpt2.guided.metrics import GuidedMetrics


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

    if args.pg_baseline == 'avg':
        baseline = pg_utils.AvgBaseline()

    for epoch_number in range(args.num_train_epochs):
        metrics = GuidedMetrics()
        for step, (batch_id, batch) in enumerate(train_dataloader):
            batch = batch.squeeze(0)
            assert batch.size(1) >= args.context_length + 1
            inp, target = batch[:, :-1].to(device), batch[:, 1:].to(device)
            optimizer.zero_grad()

            # -- MLE case
            if torch.rand(1).item() < args.pg_mle_mix:
                batch = batch.to(device)
                loss, batch_metrics = train_utils.get_mle_loss(
                    model, batch, tokenizer.pad_token_id, tokenizer.eos_token_id
                )
                # -- Update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
            else:
                # -- Decode candidates
                max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)
                bpe_continuations, candidates, distances = mrt_utils.decode_and_distance(
                    model, tokenizer, batch, score_model, max_length, device, args
                )

                # -- Compute loss: forward, normalize, weight by the task loss.
                loss, batch_metrics = pg_utils.pg_loss(
                    model, batch, baseline, candidates, distances, tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                    args.pg_num_samples,
                    args.pg_normalize_distance
                )

                # -- Record statistics
                metrics.step(
                    loss.item(), distances.mean().item(), bpe_continuations, target, args.context_length,
                    tokenizer.pad_token_id, tokenizer.eos_token_id
                )

                # -- Update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            if step % args.print_every == 0 and step >= 10:
                metrics_ = metrics.normalize('train', loss_normalize='batches')
                metrics.reset()
                print("Epoch %d   \t Step %d   \tloss: %.3f\tdist: %.3f\tavg_len: %.1f\tnon_term: %.3E" % (
                    epoch_number,
                    step,
                    metrics_['train/loss'],
                    metrics_['train/distance'],
                    metrics_['train/model_lens'],
                    metrics_['train/non_term'],
                ))

                utils.log_tensorboard(metrics_, args.log_step)
                stats_cache['train/loss'].append(metrics_['train/loss'])
                stats_cache['train/distance'].append(metrics_['train/distance'])

            if args.log_step % args.valid_every == 0:# or step == 0:
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
                    print('Best distance achieved (%.5f)' % val_loss)
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
    parser.add_argument("--pg-mle-mix", type=float, default=0.0)
    parser.add_argument("--pg-num-samples", type=int, default=4)
    parser.add_argument("--pg-normalize-distance", type=int, default=1, choices=[0, 1])
    parser.add_argument("--pg-baseline", type=str, default='avg', choices=['avg'])
    return parser