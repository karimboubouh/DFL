from src.learners import dfl
from src.ml import get_dataset
from src.ml import initialize_models
from src.network import network_graph
from src.plots import plot_train_history
from src.utils import exp_details, load_conf, fixed_seed, save

if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    args = load_conf()
    args.mp = 0
    args.optimize_network = True
    args.optimize_model = True
    args.dataset = "cifar"
    args.model = "cnn"
    args.epochs = 1
    args.iid = 0
    args.unequal = 1
    args.num_users = 10
    args.rounds = 251
    args.verbose = 1
    # =================================
    fixed_seed(True)
    exp_details(args)
    train_ds, test_ds, user_groups = get_dataset(args)
    models = initialize_models(args, same=False)
    graph = network_graph(models, train_ds, test_ds, user_groups, args)
    # Phase II: Collaborative training
    train_logs, energy_logs, time_logs = graph.collaborative_training(learner=dfl, args=args)
    title = (f"Params :: Peers: {args.num_users} | Epochs: {args.epochs} | Rounds: {args.rounds} | "
             f"IID: {'Yes' if args.iid else 'No'} | Unequal: {'Yes' if args.unequal else 'No'} | "
             f"G: W_opt")
    info = {'xlabel': "Rounds", 'title': title}
    data = {'train': train_logs, 'energy': energy_logs, 'time': time_logs[0], 'time_delta': time_logs[1]}
    save(f"SR_{args.dataset}_{args.rounds}_opt_{args.optimize_network}_{args.optimize_model}", data)
    plot_train_history(train_logs, time_logs[0], metric='accuracy', measure="mean-std", info=info)
    print("END.")
