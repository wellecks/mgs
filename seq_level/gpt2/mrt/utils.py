import torch

import seq_level.gpt2.utils as gpt2_utils
import seq_level.gpt2.guided.utils as ggs_utils


def decode_and_distance(model, tokenizer, batch, score_model, max_len, device, args):
    model.eval()

    num_samples = args.mrt_num_samples
    if args.mrt_include_target:
        num_samples = num_samples - 1
    if args.mrt_include_greedy:
        num_samples = num_samples - 1

    _, _, bpes, texts, outputs = sample_batch(
        model, tokenizer, batch, args.context_length, device,
        max_length=max_len,
        num_samples=num_samples,
        fixed_length=args.fixed_length,
        temperature=args.train_temperature
    )
    bpe_continuations = [b[args.context_length:] for b in bpes]  # original has prefix


    if args.mrt_include_greedy:
        _, _, bpes_greedy, texts_greedy, outputs_greedy = ggs_utils.generate_batch(
            model, tokenizer, batch, args.context_length, device,
            max_length=max_len,
            decoder='greedy',
            fixed_length=args.fixed_length
        )
        bpe_continuations_greedy = [b[args.context_length:] for b in bpes_greedy]  # original has prefix
        bpe_continuations_ = []
        for i in range(batch.size(0)):
            bpe_continuations_.append(bpe_continuations_greedy[i])
            bpe_continuations_.extend(bpe_continuations[i*num_samples:(i+1)*num_samples])
        bpe_continuations = bpe_continuations_

        l = max(outputs.size(1), outputs_greedy.size(1))
        outputs_ = torch.zeros(
            batch.size(0), num_samples+1, l,
            dtype=torch.long,
            device=outputs.device
        )
        outputs_[:, 0, :outputs_greedy.size(1)] = outputs_greedy
        outputs = outputs.view(batch.size(0), num_samples, -1).contiguous()
        outputs_[:, 1:, :outputs.size(-1)] = outputs
        outputs = outputs_.view(-1, outputs_.size(-1)).contiguous()

        num_samples = num_samples + 1

    model.train()

    batch_ = torch.repeat_interleave(batch, repeats=num_samples, dim=0)
    prefix_, target_trim, model_curr_trim = ggs_utils.trim(
        bpe_continuations, batch_, args.context_length, tokenizer.eos_token_id
    )

    if args.mrt_include_target:
        model_curr_trim_ = []
        target_trim_ = []
        for i in range(batch.size(0)):
            model_curr_trim_.append(target_trim[i*num_samples])
            model_curr_trim_.extend(model_curr_trim[i*num_samples:(i+1)*num_samples])
            target_trim_.extend([target_trim[i*num_samples]]*args.mrt_num_samples)
        model_curr_trim = model_curr_trim_
        target_trim = target_trim_

        l = max(outputs.size(1), batch.size(1))
        outputs_ = torch.zeros(
            batch.size(0), args.mrt_num_samples, l,
            dtype=torch.long,
            device=outputs.device
        )

        outputs_[:, 0, :batch.size(1)] = batch
        outputs = outputs.view(batch.size(0), args.mrt_num_samples-1, -1).contiguous()
        outputs_[:, 1:, :outputs.size(-1)] = outputs
        outputs = outputs_.view(-1, outputs_.size(-1)).contiguous()

    distance = ggs_utils.task_distance(
        target_trim, model_curr_trim, outputs, score_model, args.ggs_metric,
        eos_id=tokenizer.eos_token_id,
        average=False,
    )

    return bpe_continuations, outputs, distance


def sample_batch(model, tokenizer, batch, context_length, device, max_length, num_samples, fixed_length, temperature):
    with torch.no_grad():
        batch, max_len_in_batch = gpt2_utils.wrap_context_batch(batch, context_length)
        batch = torch.repeat_interleave(batch, repeats=num_samples, dim=0)

        batch = batch.to(device)
        bpe_prefixes = batch.tolist()
        text_prefixes = [tokenizer.decode(p) for p in bpe_prefixes]
        bpe_decodings = []
        text_decodings = []

        if batch.size(0) > 0:
            if fixed_length > 0:
                min_length = fixed_length
                max_length = fixed_length
            else:
                min_length = 0
            outputs = model.generate(
                batch,
                max_length=max(1, max_length),
                min_length=min_length,
                do_sample=True,
                temperature=temperature,
                eos_token_ids=tokenizer.eos_token_id
            )
            full_outputs = outputs.clone()
            for b_ind in range(outputs.size(0)):
                bpe_decoding = outputs[b_ind].tolist()
                if tokenizer.eos_token_id in bpe_decoding:
                    bpe_decoding = bpe_decoding[:bpe_decoding.index(tokenizer.eos_token_id)+1]
                text_decoding = tokenizer.decode(bpe_decoding)

                bpe_decodings.append(bpe_decoding)
                text_decodings.append(text_decoding)

    return bpe_prefixes, text_prefixes, bpe_decodings, text_decodings, full_outputs


def mrt_loss(model, batch, candidates, distances, pad_token_id, eos_token_id, num_candidates, normalize_distance, offset_zero):
    # See e.g. https://github.com/pytorch/fairseq/blob/23adb0c110fdd5e9166b3939987c5d26df996ec3/
    #          fairseq/criterions/sequence_risk_criterion.py#L44
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

    # Ensures zero-distance candidates receive a non-zero gradient
    if offset_zero:
        distances += 1

    scores = (log_probs*sequence_mask).sum(2) / sequence_mask.sum(2).clamp(min=1)
    scores = scores.exp()
    probs = torch.softmax(scores, 1)

    loss = (distances*probs).sum(1).mean()

    metrics = dict(
        loss=loss.item()
    )
    return loss, metrics
