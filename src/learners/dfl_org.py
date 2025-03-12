import copy
import time

import matlab.engine
import numpy as np
from tqdm import tqdm

from src import protocol
from src.conf import EVAL_ROUND, WAIT_TIMEOUT, WAIT_INTERVAL, ML_ENGINE, PORT
from src.ml import GAR, model_inference, compress_model, uncompress_model
from src.p2p import Graph, Node
from src.plots import w_heatmaps
from src.utils import log, wait_until, transmission_energy, save, load

name = "Decentralized Federated Learning (DFL)"


# ---------- Algorithm functions ----------------------------------------------

def collaborate(graph: Graph, args):
    eng = matlab.engine.start_matlab()
    optimized_Ws = {}
    prev_freq = "MAX"
    prev_gamma = 1
    for i in range(args.rounds):
        W = graph.update_connectivity_matrix(rho=np.random.uniform(0.4, 1))
        print(W)
        w_p, freq, gamma = optimize_w(eng, W, info=True, heatmap=False)
        if w_p is None:
            optimized_Ws[i] = (W, W, prev_freq, prev_gamma)
        else:
            prev_freq = freq
            prev_gamma = gamma
            optimized_Ws[i] = (W, w_p, prev_freq, prev_gamma)
        print(f"\n============================================================================== R::{i}\n")
    save("optimized_W50x50", optimized_Ws)
    exit("Done.")









    # eng = matlab.engine.start_matlab()
    eng = args.eng
    weird_Ws = []
    prev_freq = "MAX"
    prev_gamma = 1

    log("info", f"Initializing DFL...")
    # init peers parameters
    for peer in graph.peers:
        peer.execute(train_init)
    graph.join()

    log("info", f"Training for T = {graph.args.rounds} rounds")
    T = tqdm(range(graph.args.rounds), position=0)
    Wx = load("optimized_WR.pkl")
    for t in T:
        # get new connectivity matrix
        # W = graph.update_connectivity_matrix(rho=np.random.uniform(0.4, 1))
        # update peers set of neighbors
        W = Wx[t][0]
        for i, peer in enumerate(graph.peers):
            peer.params.neighbors = W[i]
        # optimize the network
        log('event', f"Network has been updated")
        if graph.args.optimize_network:
            log('', f"Optimazing new W...")
            w_p, freq, gamma = Wx[t][1], Wx[t][2], Wx[t][3]
            # w_p, freq, gamma = optimize_w(eng, W, info=True, heatmap=False)
            # if w_p is None:
            #     weird_Ws.append((W, W, prev_freq, prev_gamma))
            # else:
            #     prev_freq = freq
            #     prev_gamma = gamma
            #     weird_Ws.append((W, w_p, freq, gamma))

        else:
            w_p, freq, gamma = None, None, None
        # Execute the training step
        for peer in graph.peers:
            peer.execute(train_step, t, W, w_p, freq, gamma, graph.args.rounds)
        graph.join(t)

    log("info", f"Evaluating the output of the collaborative training.")
    for peer in graph.peers:
        peer.execute(train_stop)
    graph.join()
    log('info', f"Graph G disconnected.")

    # get collaboration logs
    times = [peer.params.train_time for peer in graph.peers]
    total_energy = sum([peer.params.energy for peer in graph.peers])
    total_energy_opt = sum([peer.params.energy_opt for peer in graph.peers])
    log('result', f"Total transmission Energy: {total_energy:.4f}")
    log('result', f"Total transmission Energy after optimization: {total_energy_opt:.4f}")
    time_logs = np.cumsum(np.max(times, axis=0))
    train_logs = {peer.id: peer.params.logs for peer in graph.peers}
    energy_logs = {peer.id: [peer.params.energy, peer.params.energy_opt] for peer in graph.peers}
    if len(weird_Ws) > 0:
        save("weird_Ws", weird_Ws)
    return train_logs, energy_logs, time_logs


# ---------- Algorithm functions ----------------------------------------------

def train_init(peer):
    r = peer.evaluate(peer.inference, one_batch=True)
    peer.params.neighbors = peer.neighbors
    peer.params.logs = [r]
    peer.params.train_time = [0]
    peer.params.exchanges = 0
    peer.params.mu = 0.1
    peer.params.freq = "MAX"
    peer.params.gamma = 1
    peer.params.energy = 0
    peer.params.energy_opt = 0
    return


def train_step(peer, t, w, w_p, freq, gamma, rounds):
    T = t if isinstance(t, tqdm) or isinstance(t, range) else [t]
    for t in T:
        start_time = time.time()
        if w_p is not None:
            peer.params.neighbors = w_p[peer.port - PORT]
            peer.params.freq = freq[peer.port - PORT]
        else:
            peer.params.neighbors = w[peer.port - PORT]
        peer.train_one_epoch()
        model = peer.get_model_params()
        full_neighbors = sum([i > 0 for i in w[peer.port - PORT]])
        neighbors_list = peer.params.neighbors.tolist().copy()
        neighbors_list.pop(peer.id)
        active_neighbors = [peer.neighbors[i] for i in range(len(peer.neighbors)) if neighbors_list[i] > 0]
        if peer.params.optimize_network:
            full_model_size = sum(param.numel() for param in peer.model.state_dict().values())
            if peer.params.optimize_model:
                model, cinfo = compress_model(peer.model, gamma, peer.id)
                msize = cinfo['compressed']
            else:
                msize = full_model_size
            energy = transmission_energy(full_model_size, full_neighbors)
            peer.params.energy += energy
            energy_opt = transmission_energy(msize, len(active_neighbors))
            peer.params.energy_opt += energy_opt
            if peer.id == 0:
                log('success', f"Transmission Energy for {peer} is [{len(active_neighbors)}] "
                               f"{energy_opt:.6f} J instead of [{full_neighbors}] {energy:.6f} J.")
        msg = protocol.train_step(t, model)
        peer.broadcast(msg, active_neighbors)
        # wait for enough updates labeled with round number t
        wait_until(enough_received, WAIT_TIMEOUT, WAIT_INTERVAL, peer, t, len(active_neighbors))
        if t not in peer.V:
            peer.V[t] = []
            log('log', f"{peer} received no messages in round {t}.")
        else:
            log('log', f"{peer} got {len(peer.V[t])}/{len(active_neighbors)} messages in round {t}.")
        # collaborativeUpdate
        v = collaborativeUpdate(peer, t)
        # update and evaluate the model
        end_time = time.time() - start_time
        update_model(peer, v, evaluate=(t % EVAL_ROUND == 0))
        # start accepting gradients from next round
        peer.current_round = t + 1
        if t % EVAL_ROUND == 0:
            if peer.id == 0:
                log('event', f"{peer} Execution time: {end_time:.4f} seconds.")
            peer.params.train_time.append(end_time)
        del peer.V[t]
    return


def train_stop(peer):
    model_inference(peer, one_batch=True)
    peer.stop()
    return


def collaborativeUpdate(peer, t):
    vi: list = peer.get_model_params()
    if peer.params.optimize_model:
        accepted = []
        model = copy.deepcopy(peer.model)
        for x in peer.V[t]:
            model.load_state_dict(uncompress_model(x[1]))
            accepted.append(model.get_params())
    else:
        accepted = [x[1] for x in peer.V[t]]
    if len(accepted) == 0:
        return vi
    if ML_ENGINE == "PyTorch":
        return peer.params.mu * vi + (1 - peer.params.mu) * GAR(peer, accepted)
    else:
        avg = GAR(peer, accepted)
        return [peer.params.mu * vi_k + (1 - peer.params.mu) * avg[k] for k, vi_k in enumerate(vi)]


def update_model(peer: Node, v, evaluate=False):
    peer.set_model_params(v)
    # TODO Review update function
    # peer.take_step()
    if evaluate:
        t_eval = peer.evaluate(peer.inference, one_batch=True)
        if peer.id == 0:
            print(f"{peer} t_eval = {t_eval}.")
        peer.params.logs.append(t_eval)


def optimize_w(eng, W, info=True, heatmap=False, threshold=0.001):
    matlab_matrix = matlab.double(W.tolist())
    future = eng.OPT50(matlab_matrix, nargout=3, background=True)
    start_time = time.time()
    while not future.done():
        if info:
            print(f"Optimization time: {int(time.time() - start_time)} seconds", end="\r", flush=True)
        time.sleep(1)
    W_p, F, G = future.result()
    if W_p == -999:
        log('error', f"Optimization timeout :: Rollback to initial W, prev freq & gamma")
        print(W)
        row_sums = np.round(W.sum(axis=1), 3)
        col_sums = np.round(W.sum(axis=0), 3)
        log('error', f"W ==> Sum of rows: {row_sums} | Sum of cols: {col_sums} | ")
        return None, None, None
    W_p = np.array(W_p)
    F = np.array(F)
    G = float(G)
    if info:
        log('warning', f"Initial W has {np.sum(W < threshold)} zero elements | "
                       f"W_p has {np.sum(W_p < threshold)} zero elements")
        row_sums = np.round(W_p.sum(axis=1), 3)
        col_sums = np.round(W_p.sum(axis=0), 3)
        log('log', f"W_p ==> Sum of rows: {row_sums} | Sum of cols: {col_sums} | ")
        print(W_p)
    if heatmap:
        w_heatmaps(W, W_p, threshold=0.01)
    return W_p, F, G


# ---------- Helper functions -------------------------------------------------

def enough_received(peer: Node, t, size):
    if t in peer.V and len(peer.V[t]) >= size:
        return True
    return False
