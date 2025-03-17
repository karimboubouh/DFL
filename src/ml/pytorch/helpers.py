import copy
import time

from src.ml.pytorch.models import *
from .aggregators import average, median, aksel, krum


def initialize_models(args, same=False):
    # INITIALIZE PEERS MODELS
    models = []
    modelClass = None
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            modelClass = CNNMnist
        elif args.dataset == 'fmnist':
            modelClass = CNNFashionMnist
        elif args.dataset == 'cifar':
            # modelClass = CNNCifar
            modelClass = CIFAR10CNN
            # modelClass = OptimizedCNNCifar
    elif args.model == 'mlp':
        # Multi-layer perceptron
        if args.dataset == 'mnist':
            modelClass = FFNMnist
        elif args.dataset == 'cifar':
            log('error', f"Model <MLP> is not compatible with <CIFAR> dataset.")
            exit(0)
        else:
            modelClass = MLP
    elif args.model == 'linear':
        modelClass = LogisticRegression
    else:
        exit('Error: unrecognized model')

    if same:
        # Initialize all models with same weights
        if args.model == 'cnn':
            model = modelClass(args=args)
        else:
            len_in = 28 * 28
            model = modelClass(dim_in=len_in, dim_out=args.num_classes)
        for i in range(args.num_users):
            models.append(copy.deepcopy(model))
        return models

    else:
        # Independent initialization
        for i in range(args.num_users):
            if args.model == 'cnn':
                model = modelClass(args=args)
            else:
                len_in = 28 * 28
                model = modelClass(dim_in=len_in, dim_out=args.num_classes)
            models.append(model)

    for model in models:
        model.to(args.device)

    return models


def model_fit(peer):
    history = []
    optimizer = peer.params.opt_func(peer.model.parameters(), peer.params.lr)  # , 0.99
    for epoch in range(peer.params.epochs):
        t = time.time()
        for batch in peer.train:
            # Train Phase
            loss = peer.model.train_step(batch, peer.device)
            loss.backward()
            optimizer.step()
            # test_hello(peer)
            optimizer.zero_grad()
        # Validation Phase
        result = peer.model.evaluate(peer.val, peer.device)
        peer.model.epoch_end(epoch, result, time.time() - t)
        history.append(result)

    return history


def train_for_x_epoch(peer, batches=1, evaluate=False):
    for i in range(batches):
        # train for x batches randomly chosen when Dataloader is set with shuffle=True
        batch = next(iter(peer.train))
        # execute one training step
        optimizer = peer.params.opt_func(peer.model.parameters(), peer.params.lr)
        loss = peer.model.train_step(batch, peer.device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # get gradients
        # TODO review store gradients in "peer.grads"
        # grads = []
        # for param in peer.model.parameters():
        #     grads.append(param.grad.view(-1))
        # peer.grads = torch.cat(copy.deepcopy(grads))
    if evaluate:
        return peer.model.evaluate(peer.val, peer.device)

    return None


def train_grads(peer):
    batch = next(iter(peer.train))
    loss = peer.model.train_step(batch, peer.device)
    loss.backward()


def evaluate_model(model, dataholder, one_batch=False, device="cpu"):
    return model.evaluate(dataholder, one_batch=one_batch, device=device)


def model_inference(peer, one_batch=False):
    t = time.time()
    r = peer.model.evaluate(peer.inference, peer.device, one_batch)
    o = "I" if one_batch else "*"
    acc = round(r['val_acc'] * 100, 2)
    loss = round(r['val_loss'], 2)
    t = round(time.time() - t, 1)
    log('result', f"Node {peer.id} [{t}s]{o} Inference loss: {loss}, acc: {acc}%")


def get_params(model, named=False, numpy=False):
    if named:
        return model.get_named_params(numpy=numpy)
    else:
        return model.get_params(numpy=numpy)


def set_params(model, params, named=False, numpy=False):
    if named:
        log("error", "Setting params using named params is not supported")
        exit()
    else:
        model.set_params(params, numpy=numpy)


def GAR(peer, grads, weighted=True):
    # Weighted Gradients Aggregation rule
    grads = torch.stack(grads)
    if peer.params.gar == "average":
        return average(grads)
    elif peer.params.gar == "median":
        return median(grads)
    elif peer.params.gar == "aksel":
        return aksel(grads)
    elif peer.params.gar == "krum":
        return krum(grads)
    else:
        raise NotImplementedError()


import torch


def compress_model(model, gamma, pid=None):
    """
    Compress the model by sparsifying its weights.
    Retain top K elements according to gamma.
    """
    compressed_model = {}
    for name, param in model.state_dict().items():
        if param is not None:
            param_flat = param.view(-1)
            k = int(gamma * param_flat.numel())
            if k == 0:
                continue
            # Find the top-k absolute values and their indices
            topk = torch.topk(param_flat.abs(), k=k)
            values, indices = param_flat[topk.indices], topk.indices
            compressed_model[name] = {
                "values": values,
                "indices": indices,
                "shape": param.shape,
            }
    original_size = sum(param.numel() for param in model.state_dict().values())
    compressed_size = sum(len(param["values"]) for param in compressed_model.values())
    savings = 100 * (1 - compressed_size / original_size)
    if pid is not None and pid == 0:
        log('event', f"Compression Savings: Reduced model size by (gamma={gamma})  {savings:.4f}% "
                     f"({compressed_size} out of {original_size})")

    return compressed_model, {'original': original_size, 'compressed': compressed_size}


def uncompress_model(compressed_model):
    """
    Uncompress the model back to its original size.
    """
    model = {}
    for name, compressed_param in compressed_model.items():
        full_param = torch.zeros(compressed_param["shape"], dtype=compressed_param["values"].dtype)
        full_param_flat = full_param.view(-1)
        full_param_flat[compressed_param["indices"]] = compressed_param["values"]
        model[name] = full_param_flat.view(compressed_param["shape"])

    return model
