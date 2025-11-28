import copy
import time

from src.ml.pytorch.models import *
from .aggregators import average, sparsified_average, median, aksel, krum
from ...conf import EVAL_ROUND


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


def train_for_x_epoch(peer, epochs=1, batches=None, evaluate=False):
    for epoch in range(epochs):
        if batches is None:
            for i, batch in enumerate(peer.train):
                optimizer = peer.params.opt_func(peer.model.parameters(), peer.params.lr, weight_decay=1e-4)
                loss = peer.model.train_step(batch, peer.device)
                loss.backward()
                optimizer.step()
                print(f"Train loss for epoch/batch [{epoch}][{i}]: {loss.item():.6f}", end="\r")
                optimizer.zero_grad()
        else:
            t = time.time()
            for i in range(batches):
                # train for x batches randomly chosen when Dataloader is set with shuffle=True
                batch = next(iter(peer.train))
                # execute one training step
                optimizer = peer.params.opt_func(peer.model.parameters(), peer.params.lr, weight_decay=1e-4)
                loss = peer.model.train_step(batch, peer.device)
                if peer.params.verbose > 2:
                    print(f"{peer} >> Train loss for epoch/batch [{epoch}][{i}]: {loss.item():.6f}", end="\r")
                # print(f"Train loss for epoch/batch [epoch={epoch}][batch={i}][lr={peer.params.lr:.5f}]: {loss.item():.6f}", end="\r")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # get gradients
                # TODO review store gradients in "peer.grads"
                # grads = []
                # for param in peer.model.parameters():
                #     grads.append(param.grad.view(-1))
                # peer.grads = torch.cat(copy.deepcopy(grads))
            # print(f"{peer} >> Done after {round(time.time() - t, 1):.4f} seconds.")
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
        return sparsified_average(grads)
        # return average(grads)
    elif peer.params.gar == "median":
        return median(grads)
    elif peer.params.gar == "aksel":
        return aksel(grads)
    elif peer.params.gar == "krum":
        return krum(grads)
    else:
        raise NotImplementedError()


def compress_model(model, gamma, pid=None, t=0):
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
    if pid is not None and pid == 0 and t % EVAL_ROUND == 0:
        log('success', f"Compression model reduced by (gamma={gamma})  {savings:.4f}% "
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


def get_model_info(model: nn.Module):
    """
    Calculates the number of parameters and the estimated memory size of a PyTorch model.
    """

    # 1. Calculate Parameter Counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # 2. Calculate Memory Size (in Bytes)
    # We sum the size of Parameters (weights) AND Buffers (e.g., BatchNorm running_mean)

    param_size = 0
    for p in model.parameters():
        param_size += p.nelement() * p.element_size()

    buffer_size = 0
    for b in model.buffers():
        buffer_size += b.nelement() * b.element_size()

    total_size_bytes = param_size + buffer_size
    total_size_kb = total_size_bytes / 1024
    total_size_mb = total_size_bytes / (1024 ** 2)

    print("-" * 40)
    print(f"Model: {model.__class__.__name__}")
    print("-" * 40)
    print(f"Total Parameters:        {total_params:,}")
    print(f"Trainable Parameters:    {trainable_params:,}")
    print(f"Non-Trainable Params:    {non_trainable_params:,}")
    print("-" * 40)
    print(f"                         {total_size_kb:.2f} KB")
    print(f"Estimated Size (RAM):    {total_size_mb:.2f} MB")
    print("-" * 40)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": total_size_mb
    }
