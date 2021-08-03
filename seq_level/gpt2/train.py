import argparse
import torch
from copy import deepcopy
from torch.utils.data import SequentialSampler, DataLoader, RandomSampler

import torch.nn.functional as F
from tqdm import tqdm
import math
import seq_level.gpt2.utils as utils
import numpy as np
import seq_level.gpt2.guided.utils as ggs_utils
from seq_level.gpt2.metrics import GenerationMetrics

def train(model, loss_func, tokenizer, dataset_tensor_dict, args, device):
    model.train()
    train_sampler = RandomSampler(dataset_tensor_dict['train'])
    train_dataloader = DataLoader(
        dataset_tensor_dict['train'],
        sampler=train_sampler,
        batch_size=args.train_batch_size
    )

    optimizer, scheduler = utils.get_optimizer(model, len(train_dataloader), args)

    score_model = deepcopy(model)
    best_val_loss = 100
    for epoch in range(args.num_train_epochs):
        epoch_loss = 0
        epoch_ntokens = 0

        for step, batch in enumerate(train_dataloader):
            batch = batch.squeeze(0)
            optimizer.zero_grad()
            batch = batch.to(device)
            mle_loss, batch_metrics = loss_func(
                model, batch, tokenizer.pad_token_id, tokenizer.eos_token_id
            )

            loss = mle_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss = epoch_loss + batch_metrics['loss_sum']
            epoch_ntokens = epoch_ntokens + batch_metrics['ntokens']

            if step % args.print_every == 0:
                print("(%d | %d) Training loss: %.2e lr: %.2e" % (
                    epoch, step, epoch_loss / epoch_ntokens, scheduler.get_lr()[0]
                ))
                utils.log_tensorboard({'train/loss': epoch_loss/epoch_ntokens}, args.log_step)

            if args.log_step % args.valid_every == 0 or step == 0:
                val_loss, metrics, decodings = valid_iteration(
                    dataset_tensor_dict['valid'], model, score_model, loss_func, tokenizer, device,
                    context_length=args.eval_context_length,
                    num_decodings=250,
                    args=args
                )
                if val_loss < best_val_loss:
                    print('Best loss achieved (%.5f)' % val_loss)
                    if not args.no_checkpoint:
                        utils.save(model, args.save_base_dir)
                    utils.save_metrics(metrics, args)
                    utils.save_decodings(decodings, args)
                    best_val_loss = val_loss
                model.train()
                utils.log_tensorboard(metrics, args.log_step)
            args.log_step += 1


def evaluate(model, loss_func, score_model, tokenizer, dataset_tensor_dict, args, device):
    model.eval()
    val_loss, metrics, decodings = valid_iteration(
        dataset_tensor_dict[args.eval_split], model, score_model, loss_func, tokenizer, device,
        context_length=args.eval_context_length,
        num_decodings=-1,
        args=args
    )
    utils.save_metrics(metrics, args)
    utils.save_decodings(decodings, args)
    print('Done.')



def valid_iteration(dataset, model, score_model, loss_func, tokenizer, device, context_length, num_decodings, args):
    model.eval()
    # -- Next-token prediction
    valid_sampler = SequentialSampler(dataset)
    valid_dataloader = DataLoader(dataset, sampler=valid_sampler, batch_size=1)
    tqdm_bar = tqdm(valid_dataloader, desc="Validation", total=len(valid_dataloader))
    epoch_loss = 0
    epoch_ntokens = 0
    epoch_hits = 0
    for step, batch in enumerate(tqdm_bar):
        with torch.no_grad():
            batch = batch.squeeze(0)
            batch = batch.to(device)
            mle_loss, batch_metrics = loss_func(
                model, batch, tokenizer.pad_token_id, tokenizer.eos_token_id
            )

            epoch_loss = epoch_loss + batch_metrics['loss_sum']
            epoch_ntokens = epoch_ntokens + batch_metrics['ntokens']
            epoch_hits = epoch_hits + batch_metrics['hits_at_1']
            current_avg = epoch_loss/epoch_ntokens
            tqdm_bar.desc = "Validation loss: {:.2e}, PPL {:.2f}".format(
                current_avg, 2 ** (current_avg/math.log(2))
            )

    # -- Decoding
    bpe_decoding_continuations = []
    bpe_target_continuations = []
    bpe_decoding_including_prefixes = []
    text_decoding_including_prefixes = []
    text_prefixes = []
    bpe_prefixes = []

    valid_dataloader = DataLoader(
        dataset,
        sampler=valid_sampler,
        batch_size=1
    )
    tqdm_bar = tqdm(
        valid_dataloader,
        desc="Validation decoding",
        total=len(valid_dataloader)
    )
    gen_metrics = GenerationMetrics()
    for step, batch in enumerate(tqdm_bar):
        batch = batch.squeeze(0)
        # skip batches with padding in the context for simplicity
        # (plus 1 to ensure a non-empty continuation)
        if (batch[:, :context_length+1] == tokenizer.pad_token_id).sum() > 0:
            continue
        with torch.no_grad():
            bpe_prefix, text_prefix, bpes, texts, outputs = utils.generate_batch(
                model, tokenizer, batch, context_length, device,
                max_length=args.eval_decode_max_length,
                decoder=args.eval_decoder,
                fixed_length=args.fixed_length
            )

            bpes_ = [b[args.context_length:] for b in bpes]  # original has prefix

            prefix_, target_trim, model_trim = ggs_utils.trim(
                bpes_, batch, context_length, tokenizer.eos_token_id
            )

            gen_metrics.step(
                model_trim, target_trim, outputs, score_model, tokenizer.eos_token_id, context_length
            )

            bpe_decoding_continuations.extend(model_trim)
            bpe_decoding_including_prefixes.extend(bpes)
            text_decoding_including_prefixes.extend(texts)

            text_prefixes.extend(text_prefix)
            bpe_prefixes.extend(bpe_prefix)

            bpe_target_continuations.extend(target_trim)

        tqdm_bar.desc = "avg len %.2f non-term %.2f" % (
            np.mean([len(b) for b in bpe_decoding_continuations]),
            np.mean([len(b) == args.eval_decode_max_length for b in bpe_decoding_including_prefixes])
        )
        if num_decodings > 0 and step == num_decodings:
            break

    decodings = {
        'text_decoding_including_prefix': text_decoding_including_prefixes,
        'bpe_decoding_continuation': bpe_decoding_continuations,
        'text_prefix': text_prefixes,
        'bpe_prefix': bpe_prefixes,
        'bpe_target_continuation': bpe_target_continuations
    }

    metrics = {
        'valid/avg_loss': current_avg,
        'valid/ppl': 2 ** (current_avg/math.log(2)),
        'valid/hits': (epoch_hits/float(epoch_ntokens)),
    }
    gen_metrics = gen_metrics.normalize('valid')
    for k, v in gen_metrics.items():
        metrics[k] = v

    print("Validation results:")
    for k, v in metrics.items():
        print("\t%s\t%.3E" % (k, v))

    return current_avg, metrics, decodings


def get_mle_loss(model, batch, pad, eos):
    inp = batch[:, :-1]
    inp_ = inp.clone()
    inp_[inp == pad] = 0

    target = batch[:, 1:]
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
    loss = loss.sum()
    ntokens = target_mask.sum()
    metrics = get_train_metrics(logits, target_, pad)
    metrics['loss_sum'] = loss.item()
    metrics['ntokens'] = ntokens.item()
    loss = loss / ntokens
    return loss, metrics


def get_train_metrics(logits, target, pad):
    true_token_logits = -F.nll_loss(
        logits.view(-1,logits.size(-1)),
        target.reshape(-1),
        reduction='none'
    )
    pad_mask = target.unsqueeze(0).ne(pad)
    true_token_logits = true_token_logits.view(logits.size(0), logits.size(1))
    mask_higher_scored_tokens = (logits > true_token_logits.unsqueeze(2)).float()
    target_rank = mask_higher_scored_tokens.sum(dim=-1)
    hits_at_10 = ((target_rank < 10)*pad_mask).sum().item()
    hits_at_1 = ((target_rank == 0)*pad_mask).sum().item()

    logging_output = {
        'hits_at_1': hits_at_1,
        'hits_at_10': hits_at_10,
    }
    return logging_output


def main():
    parser = argparse.ArgumentParser()
    # -- standard training args
    parser.add_argument("--model-load-dir", type=str, default=None)
    parser.add_argument("--score-model-load-dir", type=str, default=None)
    parser.add_argument("--line-dataset", action='store_true')
    parser.add_argument(
        "--model-name",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        default="gpt2"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/wikitext103_raw_gpt2bpe.pkl"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="../models/tokenizer_cache"
    )
    parser.add_argument(
        "--save-base-dir",
        default='./wikipedia103/',
        type=str
    )
    parser.add_argument(
        '--eval-split',
        type=str,
        choices=['train', 'valid', 'test'],
        default='valid'
    )
    parser.add_argument(
        '--use-batch-max-len',
        action='store_true',
        help='When True, generates up to the ground-truth length of the batch. Else, 500 steps.'
    )
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--valid-every', type=int, default=1000)
    parser.add_argument('--train-batch-size', type=int, default=1)
    parser.add_argument('--eval-batch-size', type=int, default=1)
    parser.add_argument('--eval-context-length', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-num-steps', type=int, default=-1)
    parser.add_argument('--num-train-epochs', type=int, default=100)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--expr-name', default='wikitext103')
    parser.add_argument('--context-length', type=int, default=10)
    parser.add_argument('--token-limit-train', type=int, default=1024)
    parser.add_argument('--token-limit-eval', type=int, default=1024)
    parser.add_argument('--chunk-size-train', type=int, default=1024)
    parser.add_argument('--chunk-size-valid', type=int, default=1024)
    parser.add_argument('--decode-max-length', type=int, default=1000)
    parser.add_argument('--eval-decode-max-length', type=int, default=300)
    parser.add_argument('--val-metric', type=str, default='ppl', choices=['ppl', 'distance'])

    # -- optimizer
    parser.add_argument("--adam-epsilon", default=1e-8, type=float)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup-steps", default=0, type=int)
    parser.add_argument('--lr-schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight-decay', type=float, default=0.01)

    # -- custom
    parser.add_argument('--loss', choices=['mle', 'ggs', 'mrt', 'pg'], default='mle')
    parser.add_argument('--train-decoder', choices=['greedy', 'temp-1.0'], default='greedy')
    parser.add_argument('--eval-decoder', choices=['greedy', 'temp-1.0'], default='greedy')
    parser.add_argument('--fixed-length', type=int, default=-1)
    parser.add_argument('--train-temperature', type=float, default=1.0)

    from seq_level.gpt2.guided.train import add_args
    parser = add_args(parser)
    from seq_level.gpt2.mrt.train import add_args
    parser = add_args(parser)
    from seq_level.gpt2.pg.train import add_args
    parser = add_args(parser)

    args = parser.parse_args()
    device, logger = utils.setup(args)

    if args.loss == 'ggs':
        import seq_level.gpt2.guided.utils as ggs
        load_model = ggs.load_model
        loss_func = get_mle_loss
    elif args.loss == 'mrt':
        import seq_level.gpt2.guided.utils as ggs
        load_model = ggs.load_model
    elif args.loss == 'pg':
        import seq_level.gpt2.guided.utils as ggs
        load_model = ggs.load_model
    else:
        loss_func = get_mle_loss
        load_model = utils.load_model

    model, tokenizer = load_model(args, device)
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    raw_dataset_dict, line_datasets = utils.load_dataset(
        args.dataset_path, tokenizer.pad_token_id, args
    )
    if args.mode == 'train':
        if args.loss == 'ggs':
            from seq_level.gpt2.guided.train import train as guided_train
            guided_train(model, tokenizer, line_datasets, args, device)
        elif args.loss == 'mrt':
            from seq_level.gpt2.mrt.train import train as mrt_train
            mrt_train(model, tokenizer, line_datasets, args, device)
        elif args.loss == 'pg':
            from seq_level.gpt2.pg.train import train as pg_train
            pg_train(model, tokenizer, line_datasets, args, device)
        else:
            train(model, loss_func, tokenizer, line_datasets, args, device)

    if args.mode == 'eval':
        # to load the MLE model as a scoring model
        model_load_dir = args.model_load_dir
        args.model_load_dir = args.score_model_load_dir
        score_model, _ = load_model(args, device)
        args.model_load_dir = model_load_dir
        evaluate(model, loss_func, score_model, tokenizer, line_datasets, args, device)


if __name__ == '__main__':
    main()