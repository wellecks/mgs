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
from functools import partial
from seq_level.gpt2.guided.metrics import GuidedMetrics
from concurrent.futures import ThreadPoolExecutor

from timeit import default_timer as timer
import pandas as pd
import collections
import random
import shelve
import hashlib
import pickle
import logging
import math

def _hash_tensor(obj):
    return hashlib.sha1(bytes(obj.cpu().numpy())).hexdigest()

def _hash_model(model):
    return hashlib.sha1(next(model.parameters()).detach().cpu().numpy()).hexdigest()

MODEL_ID = None

class RingBuffer:
    def __init__(self, max_size=1000, persistence='none', persistent_file_path=None, shuffle=True, iter_device=None, on_device=False):
        self.max_size = max_size
        self.persistence = persistence
        self.queue = []

        if persistence == 'none':
            self.db = {}
        elif persistence == 'shelve':
            self.db = shelve.open(persistent_file_path)
        self.db_counter = {}

        self.shuffle = shuffle
        self.iter_device = None

        self.on_device = on_device
        # self.executor = ThreadPoolExecutor(max_workers=6)

    def __len__(self):
        return len(self.queue)
    
    def append(self, idx, type, batch_id, batch, model, 
                sequences, distances, rng_state=None):

        print(f"Id: {idx}::" + 
               f" Queue Size: {len(self.queue)}," + 
               f" DB size: {len(self.db)}", end='\r')
    
        if len(self.queue) >= self.max_size:
            (_, _, old_batch_key, old_model_key, 
                old_sequences_key, old_distances, _) = self.queue.pop(0)
            logging.debug("Removing item from Queue: " + 
                           f"Batch: {old_batch_key} " + 
                           f"Model: {old_model_key}.")

            self.db_counter[old_model_key] -= 1
            if self.db_counter[old_model_key] == 0:
                del self.db_counter[old_model_key]
                del self.db[old_model_key]

            self.db_counter[old_batch_key] -= 1
            if self.db_counter[old_batch_key] == 0:
                del self.db_counter[old_batch_key]
                del self.db[old_batch_key]

            # self.db_counter[old_sequences_key] -= 1
            # if self.db_counter[old_sequences_key] == 0:
            #     del self.db_counter[old_sequences_key]
            #     del self.db[old_sequences_key]

        # batch_key = f"batch_{_hash_tensor(batch)}"
        batch_key = f"batch_{batch_id}"
        if batch_key not in self.db:
            if not self.on_device:
                batch = deepcopy(batch).to(device=torch.device("cpu"))
            self.db[batch_key] = batch
            self.db_counter[batch_key] = 0
        self.db_counter[batch_key] += 1

        global MODEL_ID
        model_key = f"model_{MODEL_ID}"
        if model_key not in self.db:
            if not self.on_device:
                model = deepcopy(model).to(device=torch.device("cpu"))
            self.db[model_key] = model
            self.db_counter[model_key] = 0
        self.db_counter[model_key] += 1

        # sequences_key_suffix = _hash_tensor(sequences)
        # sequences_key = f"sequence_{batch_key}_{model_key}_{sequences_key_suffix}"
        # if sequences_key not in self.db:
        #     self.db[sequences_key] = sequences.cpu()
        #     self.db_counter[sequences_key] = 0
        # self.db_counter[sequences_key] += 1
        sequences_key = None

        self.queue.append((idx, type, batch_key,
                            model_key, sequences_key,
                            distances.cpu(), rng_state))

    def __iter__(self):
        iterable = self.queue

        if self.shuffle:
            iterable = random.sample(self.queue, len(self.queue))

        for idx, type, batch_key, model_key, sequences_key, distances, rng_state in iterable:
            batch = self.db[batch_key].type(torch.long)
            sequences = None
            model = self.db[model_key]

            if distances.size(0) != batch.size(0):
                logging.error(f"Distance: {distances.size(0)}, Batch: ({batch.size()}), {batch_key} Sequence: {sequences_key}" + \
                        f"Model: {model_key}.")
                continue
            yield (idx, type, batch, model, sequences, distances, rng_state)



total_scoring_time = {
    "cuml": 0,
    "tick": 0,
}

curr_scoring_time = {
    "cuml": 0,
    "tick": 0,
}

mle_grad_computation_time = {
    "cuml": 0,
    "tick": 0,
}

perturb_computation_time = {
    "cuml": 0,
    "tick": 0,
    'num_perturb': 0,
}

perturb_scoring_time = {
    "cuml": 0,
    "tick": 0,
}

weight_computation_time = {
    "cuml": 0,
    "tick": 0,
}

ggs_update_time = {
    "cuml": 0,
    "tick": 0,
}

metrics_update_time = {
    "cuml": 0,
    "tick": 0,
}

total_mgs_time = {
    "cuml": 0,
    "tick": 0,
}

total_train_step_time = {
    "cuml": 0,
    "tick": 0,
}


train_score_network_time = {
    'cuml': 0,
    'tick': 0,
}


aggregation_step_time = {
    'cuml': 0,
    'tick': 0,
}

def aggregate_score_data(step, batch_id, batch, buffer, model, score_model, tokenizer, args, device):
    """ This method does a forward pass over the original model and 
        the perturbed model to compute the yo_i, the decoded output corresponding
        to the input x using the original model, and yp_i, the decoding output corresponding
        to the perturbed model. 
        The perturbations are sampled from $\Deta \sim Q_{MGS}$.

        It returns a set of tuples with each tuple of the form (x_i, y_i, yo_i, yp_i, \Delta).
    """
    batch.squeeze_(0)
    batch = batch.to(device=device)
    if batch.size(1) < args.context_length + 1:
        logging.error(f"Batch at step: {step} has sequences: {batch.size(1)} shorter than the context length: {args.context_length}")
        return buffer

    inp, target = batch[:, :-1], batch[:, 1:]
    max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)
    model = model.to(device=device)
    model.eval()
    _, cur_decodings, cur_distances = ggs_utils.decode_and_distance(
        model, tokenizer, batch, score_model, max_length, device, args, average_distance=False
    )
    buffer.append(step, 'current', batch_id, batch, model, cur_decodings, cur_distances)

    # Get the current MLE gradients
    model.train()
    per_model, rng_state = perturb(model, batch, step, tokenizer, args, device=device)

    _, per_decodings, per_distances = ggs_utils.decode_and_distance(
        per_model, tokenizer, batch, score_model, max_length, device, args, average_distance=False
    )
    buffer.append(step, 'pertubed', batch_id, batch, model, per_decodings, per_distances, rng_state=rng_state)
    return buffer


def perturb(model, batch, step,  tokenizer, args,rng_state=None, device=None):
    per_model = deepcopy(model)
    inp, target = batch[:, :-1], batch[:, 1:]

    if step % 2 == 0:
        model_with_grad, _ = ggs_utils.mle_grad(
            per_model, inp, target, tokenizer.pad_token_id, args.max_grad_norm
        )
        model_with_grad_param_dict = dict(model_with_grad.named_parameters())
    
    with ggs_utils.RNG(rng_state, device) as (rng, rng_state):
        for name, param in per_model.named_parameters():
            perturbation = torch.randn(param.size(), generator=rng, device=param.device)

            if args.noise_scale == 'uniform':
                noise_ = args.ggs_noise * perturbation *  args.learning_rate * (param.data.abs().sum() / param.data.numel())
            else:
                noise_ = args.ggs_noise * perturbation

            # TODO: Should we consider random noise addition for data aggregation and
            # score training. 
            if step % 2 == 0:
                param_with_grad = model_with_grad_param_dict[name]
                gradient = -param_with_grad.grad.data

                epsilon = noise_ + gradient
            else:
                epsilon = noise_

            param.data = param.data + epsilon
    return per_model, rng_state


def train_score_network(buffers, model, phi_network, tokenizer, device, args):
    """ This method takes in the scoring data (B) and learns a parameterized scoring model (S) to 
        mimic the original cost function (C) by minimizing the L2 loss.
        $C$ is defined as
            C(\theta) = 1/|B| \sum_{i=1}^{B} c(y_i, F(x_i, \theta))

        The scoring function optimizes the following equation. 
            min_{W,b} \sum_{(x_i, y_i, yo_i, yp_i, \Delta_i) in B} || S(x_i, \theta, \Delta; W, b) - (C(\theta) - C(\theta + \Delta))||^2
            where S(x_i, \theta, \Delta; W, b) = W^T[R(x;\theta) - R(x;\theta+\Delta)] + b
    """
    min_train_loss = math.inf
    best_phi_network = None
    patience_counter = 0
    print('=' * 150)
    print('Start training the score network.\n')
    phi_network.train()
    phi_network = phi_network.to(device=device)

    phi_optimizer = optim.Adam(phi_network.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(phi_optimizer, step_size=20, gamma=0.5, verbose=True)

    for epoch in range(args.score_network_epochs):
        train_loss = 0.
        num_docs = 0
        train_score_network_start = timer()
        for step, (idx, type, batch, model, sequences, distances, rng_state) in enumerate(buffers):
            
            model = model.to(device=device)

            batch_ = deepcopy(batch).to(device=device)
            pad = tokenizer.pad_token_id
            batch_[batch == pad] = 0

            if type == "pertubed":
                model, _ = perturb(model, batch, idx, tokenizer, args, rng_state=rng_state, device=device)
            
            distances = distances.to(device=device)

            model.eval()

            mask = batch.ne(pad).float().to(device=device)

            model_output = model(batch_, 
                                 attention_mask=mask,
                                 output_hidden_states=True)

            emb = model_output.hidden_states[-1][:, -1, :].detach()

            phi_optimizer.zero_grad()

            outputs = phi_network(emb)

            if distances.size(0) != outputs.size(0):
                logging.error(f"Batch: ({idx} {type} {batch.size()}) != Distance {distances.size()}")
                continue

            loss = F.mse_loss(
                outputs,
                distances.view(-1, 1),
            )
            loss.backward()
            phi_optimizer.step()

            train_loss += loss.item()
            num_docs += batch.size(0)

            cur_loss = train_loss / num_docs
            if step % 5 == 0 and step > 0:
                print('Epoch: %d :: Step: %d, Loss: %.2f' % (epoch, step, cur_loss), end='\r')

            if not args.on_device:
                # Move model back to CPU so that it doesn't hog GPU
                # memory as it will not be removed from the context.
                model.to(device=torch.device("cpu"))
                distances.to(device=torch.device("cpu"))

        if min_train_loss < cur_loss:
            patience_counter += 1
        else:
            patience_counter = 0
            min_train_loss = cur_loss
            best_phi_network = deepcopy(phi_network)

        if patience_counter > args.train_score_patience:
            logging.info(f"Stopping Early at epoch: {epoch} with minimum loss: {min_train_loss}")
            break
            
        scheduler.step()
        train_score_network_end = timer()
        train_score_network_time['cuml'] += train_score_network_end - train_score_network_start
        train_score_network_time['tick'] += 1
        logging.info('Epoch: %d :: Min Loss: %.2f, Loss: %.2f, Epochs Since Last Best: %d ' % (epoch, min_train_loss, cur_loss, patience_counter))
        logging.info(f"Train score network epoch {epoch} done!")
        logging.info(f"Avg Epoch Time: {train_score_network_time['cuml']/train_score_network_time['tick']}")
    print('Done training the score network.\n')
    print('=' * 150)
    phi_network = best_phi_network

def original_mgs_scoring_function(model, tokenizer, batch, score_model, max_length, device, args, prefix):
    decoded = defaultdict(list)
    bpes_curr, outputs, distance_curr = ggs_utils.decode_and_distance(
                                                     model, tokenizer, batch, score_model, 
                                                     max_length, device, args, average_distance=False)

    for i, idxs in enumerate(bpes_curr):
        decoded[f'{prefix}_{i}'].append(tokenizer.decode(idxs))
    return distance_curr.mean().item(), bpes_curr, decoded


def dagger_mgs_scoring_function(phi_network, model, tokenizer, batch, score_model, max_length, device, args, prefix):
    """ This methods takes in the batch and the model and
         returns the estimated scores for batch input according to the model.
    """
    outputs = torch.tensor([])
    decoded = defaultdict(list)
    model.eval() 

    pad = tokenizer.pad_token_id
    mask = batch.ne(pad).float().to(device=model.device)

    batch_ = batch.clone().to(device=model.device)
    batch_[batch == pad] = 0
    output = model(batch_,
                  attention_mask=mask,
                  output_hidden_states=True)
    phi_network = phi_network.to(device=model.device)

    embed = output \
              .hidden_states[-1][:, -1, :] \
              .detach()

    batched_distances = phi_network(embed).detach().cpu() 

    # average across batch to compute c(\theta).
    distances = batched_distances.mean(dim=0).item()

    return distances, outputs, decoded


def MGS(batch, model, score_model, tokenizer, args, device, metrics, optimizer, 
        scoring_function, 
        target_scoring_func=None):
    """ MGS algorithm parameterized to work in original as well as efficient mode.
    """
    distance_comp = []
    mgs_time_start = timer()

    inp, target = batch[:, :-1].to(device=device), batch[:, 1:].to(device=device)

    # -- Decode with current model (required for computing the 'weights' later).
    max_length = ggs_utils.max_length(target, tokenizer.eos_token_id, args)

    decoded = defaultdict(list)

    curr_scoring_start = timer()
    distance_curr, bpes_curr, decoded_samples = scoring_function(
        model, tokenizer, batch, score_model, max_length, device, args, prefix='original'
    )

    if args.efficient and args.log_scoring_function and \
       args.log_step % args.print_every == 1:
        distance_score, _, _ = target_scoring_func(model, tokenizer, batch, 
                                score_model, max_length, device, args, prefix='original')
        distance_comp.append(('original', distance_curr, distance_score))
        logging.info(f"Distances: original: C_\phi => {distance_curr} C_orig => {distance_score}")
    decoded.update(decoded_samples)
    curr_scoring_end = timer()
    curr_scoring_time['cuml'] += curr_scoring_end - curr_scoring_start
    curr_scoring_time['tick'] += 1

    total_scoring_time['cuml'] += curr_scoring_end - curr_scoring_start
    total_scoring_time['tick'] += 1

    # -- Obtain MLE gradients
    mle_grad_computation_start = timer()
    model_ = deepcopy(model)
    model_with_grad, mle_loss = ggs_utils.mle_grad(
        model_, inp, target, tokenizer.pad_token_id, args.max_grad_norm
    )
    mle_grad_computation_end = timer()
    mle_grad_computation_time['cuml'] += mle_grad_computation_end - mle_grad_computation_start
    mle_grad_computation_time['tick'] += 1

    perturb_computation_start = timer()
    # -- Perturb
    perturbed_models, log_rhos, noise_magnitudes = ggs_utils.perturb(
        model, model_with_grad, args.ggs_num_samples, args.ggs_noise,
        noise_scale=args.noise_scale,
        zero_dist_only=args.zero_dist_only,
        mle_dist_only=args.mle_dist_only,
        include_mle_gradient=args.include_mle_gradient,
    )
    perturb_computation_end = timer()
    perturb_computation_time['cuml'] += perturb_computation_end - perturb_computation_start
    perturb_computation_time['tick'] += 1

    perturb_scoring_start = timer()
    # -- Decode with perturbed models and compute task metric
    distances = []
    for i, p_model in enumerate(perturbed_models):

        distance, _, decoded_samples  = scoring_function(p_model, tokenizer, batch, score_model, 
                                                            max_length, device, args, prefix=f'preturb_{i}')
        if args.efficient and args.log_scoring_function and \
            args.log_step % args.print_every == 1:
            distance_score, _, _ = target_scoring_func(p_model, tokenizer, batch, 
                                score_model, max_length, device, args, prefix='preturb_{i}')
            distance_comp.append(('preturb_{i}', distance, distance_score))
            logging.info(f"Distances: preturb_{i}: C_\phi => {distance} C_orig => {distance_score}")


        distances.append(distance)
        decoded.update(decoded_samples)
    perturb_scoring_end = timer()
    perturb_scoring_time['cuml'] += perturb_scoring_end - perturb_scoring_start
    total_scoring_time['cuml'] += perturb_scoring_end - perturb_scoring_start
    perturb_scoring_time['tick'] += 1

    # -- Compute weights
    # Kushal: please revise phi_network(distance_curr - distances), where the score_function's output is embedding.

    weight_computation_start = timer()
    log_weights = ggs_utils.compute_weight(distance_curr, distances, log_rhos, args.ggs_beta)

    # -- Compute weighted average of the directions
    update_directions = ggs_utils.parameter_weighted_average(
        model, perturbed_models, log_weights
    )
    weight_computation_end = timer()
    weight_computation_time['cuml'] += weight_computation_end - weight_computation_start
    weight_computation_time['tick'] += 1

    ggs_update_start = timer()
    # -- Perform update
    global MODEL_ID
    MODEL_ID = ggs_utils.update(model, update_directions, optimizer, args.max_grad_norm)
    ggs_update_end = timer()
    ggs_update_time['cuml'] += ggs_update_end - ggs_update_start
    ggs_update_time['tick'] += 1

    metrics_update_start = timer()
    # -- Record statistics
    metrics.step(
        mle_loss.item(), distance_curr, bpes_curr, target, args.context_length,
        tokenizer.pad_token_id, tokenizer.eos_token_id,
        model_with_grad, update_directions, log_rhos, log_weights,
        noise_magnitudes, distances
    )
    metrics_update_end = timer()
    metrics_update_time['cuml'] += metrics_update_end - metrics_update_start
    metrics_update_time['tick'] += 1

    mgs_time_end = timer()
    total_mgs_time['cuml'] += mgs_time_end - mgs_time_start
    total_mgs_time['tick'] += 1
    return decoded

def shall_aggregate_data(step, total_num_batches, args):
    # For first 25% of batches, aggregate data every batch.
    if step < total_num_batches//4: 
        return True
    
    # For best 25% of the batches, aggregate data every alternate batch.
    if step < total_num_batches//2 and step % 2 == 0:
        return True
    
    # For last 50% of the batches, sample every fourth batch.
    if step % 4 == 0:
        return True
    
    return False
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=1024):
        super(MLP, self).__init__()

        self.fc = nn.Sequential(
                        nn.Linear(input_size, hidden_size), 
                        nn.ReLU(), 
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1))

    def forward(self, x):
        output = self.fc(x)
        return output


def train(model, tokenizer, dataset_tensor_dict, args, device):
    global MODEL_ID
    MODEL_ID = ggs_utils.get_model_id(model)

    model.train()
    train_sampler = RandomSampler(dataset_tensor_dict['train'])
    train_dataloader = DataLoader(
        dataset_tensor_dict['train'],
        sampler=train_sampler,
        batch_size=1
    )

    total_num_batches = len(train_dataloader)
    optimizer, scheduler = utils.get_optimizer(model, total_num_batches, args)
    best_val_loss = 10000
    patience = args.patience
    stats_cache = defaultdict(list)
    average_times = {}
    score_model = deepcopy(model)
    scoring_function = original_mgs_scoring_function

    config = GPT2Config()
    phi_network = MLP(input_size=config.hidden_size).to(device=device)

    if args.efficient:
        # Initialize buffer and pretrain score network.
        print('=' * 150)
        buffer = RingBuffer(max_size=args.max_buffer_size, 
                    persistence='none',
                    persistent_file_path=os.path.join(
                                            args.save_base_dir,
                                            "persistence_datastore"), 
                    on_device=args.on_device)


        # If using saved score network, use it, else accumulate training data, 
        # and train network on the accumulated data.
        if args.use_saved_score_network:
            score_model_checkpoint = torch.load(args.score_network_file)
            phi_network.load_state_dict(score_model_checkpoint['model_save_dict'])
            epochs_trained = score_model_checkpoint['epochs']
            logging.info(f"Loading Phi Network trained for {epochs_trained} epochs from {args.score_network_file}.")
        else:
            # If using saved aggregated data, use it, else, initialize an empty buffer.
            if args.use_saved_aggregated_data:
                with open(args.aggregated_data_path, 'rb') as aggregated_datafile:
                    buffer = pickle.load(aggregated_datafile)
                logging.info(f"Loading Aggregated data from {args.aggregated_data_path}. Size: {len(buffer)}")
            else:
                logging.info("Started Initial Data Aggregation.")
                for step, (batch_id, batch) in enumerate(train_dataloader):
                    if step >= args.aggregated_data_size:
                        break

                    aggregate_step_start = timer()
                    aggregate_score_data(step, 
                                            batch_id,
                                            batch, 
                                            buffer,
                                            model, 
                                            score_model, 
                                            tokenizer, 
                                            args, 
                                            device)

                    aggregate_step_end = timer()
                    aggregation_step_time['cuml'] += aggregate_step_end - aggregate_step_start
                    aggregation_step_time['tick'] += 1
                    if step % args.print_every == 0:
                        logging.info(f"Aggregated Batches:  {step}/{total_num_batches}." +
                        f"Avg time: {aggregation_step_time['cuml']/aggregation_step_time['tick']}")

                logging.info(f"Aggregated: {step * 2} items in {aggregation_step_time['cuml']} seconds.")

                if args.save_aggregated_data:
                    buffer_filepath = os.path.join(args.save_base_dir, 'buffer.pkl')
                    logging.info(f"Saving Aggregated Data at {buffer_filepath}")
                    with open(buffer_filepath, 'wb') as buffer_file:
                        pickle.dump(buffer, buffer_file)

            logging.info("Training Scoring Network on Aggregated Data.")

            train_score_network(buffer, 
                                model, 
                                phi_network, 
                                tokenizer, 
                                device,
                                args)

            if args.save_score_network:
                score_network_filepath = os.path.join(args.save_base_dir, 'score_network.pkl')
                torch.save({
                    'model_save_dict': phi_network.state_dict(),
                    'epochs': args.score_network_epochs,
                    'dataset_size': len(buffer),
                }, score_network_filepath)

        scoring_function = partial(dagger_mgs_scoring_function, phi_network)
        print('=' * 150)


    for epoch_number in range(args.num_train_epochs):
        metrics = GuidedMetrics()

        for step, (batch_id, batch) in enumerate(train_dataloader):
            if args.efficient:
                if shall_aggregate_data(step, total_num_batches, args):
                    aggregate_score_data(step,
                                            batch_id,
                                            batch, 
                                            buffer,
                                            model, 
                                            score_model, 
                                            tokenizer, 
                                            args, 
                                            device)


                if (step + 1) % args.retrain_score_network_every == 0:
                    train_score_network(buffer, 
                                        model, 
                                        phi_network, 
                                        tokenizer,
                                        device,
                                        args)

            train_step_time_start = timer()

            if len(batch.shape) < 2:
                    logging.error(f"Batch has a single item and is of shape: {batch.shape}")
                    continue

            if len(batch.shape) > 2:
                batch = batch.squeeze(0)

            if batch.size(-1) < args.context_length + 1:
                logging.error(f"Batch at step: {step} has sequences: {batch.size(1)} shorter than the context length: {args.context_length}")
                continue

            batch = batch.to(device=device)

            decoded = MGS(batch=batch,
                          model=model,
                          score_model=score_model,
                          tokenizer=tokenizer,
                          args=args,
                          device=device,
                          metrics=metrics,
                          optimizer=optimizer,
                          scoring_function=scoring_function,
                          target_scoring_func=original_mgs_scoring_function
                         )
            train_step_time_end = timer()

            total_train_step_time['cuml'] += train_step_time_end - train_step_time_start
            total_train_step_time['tick'] += 1

            if step % args.print_every == 0:
                metrics_ = metrics.normalize('train')
                metrics.reset()
                logging.info("Epoch %d   \t Step %d   \tmle: %.3f\tdist: %.3f\tnon_term: %.3E\tmle_weight: %.3E" % (
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

                average_times['total_scoring_time'] = total_scoring_time['cuml']/total_scoring_time['tick']
                average_times['curr_scoring_time'] = curr_scoring_time['cuml']/curr_scoring_time['tick']
                average_times['mle_grad_computation_time'] = mle_grad_computation_time['cuml']/mle_grad_computation_time['tick']
                average_times['perturb_computation_time'] = perturb_computation_time['cuml']/perturb_computation_time['tick']
                average_times['perturb_scoring_time'] = perturb_scoring_time['cuml']/perturb_scoring_time['tick']
                average_times['weight_computation_time'] = weight_computation_time['cuml']/weight_computation_time['tick']
                average_times['ggs_update_time'] = ggs_update_time['cuml']/ggs_update_time['tick']
                average_times['metrics_update_time'] = metrics_update_time['cuml']/metrics_update_time['tick']
                average_times['total_mgs_time'] = total_mgs_time['cuml']/total_mgs_time['tick']
                average_times['total_train_step_time'] = total_train_step_time['cuml']/total_train_step_time['tick']

                if args.plot_times:
                    df = pd.DataFrame.from_dict(average_times, 
                                                orient='index', 
                                                columns=['avg. time'])
                    print(df)

            if args.log_step % args.valid_every == 0:
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
                    logging.info('Best distance achieved (%.5f)' % val_metrics['valid/distance-%s' % args.ggs_metric])
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

    parser.add_argument(
        "--aggregated-data-size", type=int, default=2000,
    )
    parser.add_argument(
        "--aggregated-data-path", type=str,
    )
    parser.add_argument(
        "--save-aggregated-data", action='store_true',
    )
    parser.add_argument(
        "--use-saved-aggregated-data", action='store_true',
    )

    parser.add_argument('--include-mle-gradient', action='store_true')

    parser.add_argument(
        "--max-buffer-size", type=int, default=4000,
    )


    parser.add_argument(
        "--score-network-epochs", type=int, default=100,
    )
    parser.add_argument(
        "--retrain-score-network-every", type=int, default=500
    )
    parser.add_argument(
        "--use-saved-score-network", action='store_true',
    )
    parser.add_argument(
        "--score-network-file", type=str,
    )
    parser.add_argument(
        "--save-score-network", action='store_true',
    )

    parser.add_argument('--efficient', action='store_true')
    
    parser.add_argument('--plot-times', action='store_true')

    parser.add_argument('--log-scoring-function', action='store_true')

    parser.add_argument('--on-device', action='store_true')

    parser.add_argument(
        "--train-score-patience", type=int, default=10,
    )
    return parser
