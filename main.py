from src.learners import dfl
from src.ml import get_dataset
from src.ml import initialize_models
from src.network import network_graph
from src.plots import plot_train_history
from src.utils import exp_details, load_conf, fixed_seed, save

if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    args = load_conf()
    # =================================
    args.mp = 0
    args.optimize_network = True
    args.optimize_model = True
    args.dataset = "mnist"
    args.model = "mlp"
    args.epochs = 1
    args.iid = 0
    args.unequal = 1
    args.num_users = 10
    args.rounds = 501
    args.verbose = 10
    # =================================
    fixed_seed(True)
    # print experiment details
    exp_details(args)
    # load dataset and initialize user groups
    train_ds, test_ds, user_groups = get_dataset(args)
    # build users models
    models = initialize_models(args, same=False)
    # build the network graph
    graph = network_graph(models, train_ds, test_ds, user_groups, args)
    # Phase II: Collaborative training
    train_logs, energy_logs, time_logs = graph.collaborative_training(learner=dfl, args=args)
    # save(f"p3_log_{args.num_users}_{args.epochs}", train_logs)
    title = (f"Params :: Peers: {args.num_users} | Epochs: {args.epochs} | Rounds: {args.rounds} | "
             f"IID: {'Yes' if args.iid else 'No'} | Unequal: {'Yes' if args.unequal else 'No'} | "
             f"Test Scope: {args.test_scope}")
    info = {'xlabel': "Rounds", 'title': title}
    data = {'train': train_logs, 'energy': energy_logs, 'time': time_logs}
    save(f"{args.dataset}_{args.num_users}_{args.rounds}_opt_{args.optimize_network}_{args.optimize_model}", data)
    plot_train_history(train_logs, metric='accuracy', measure="mean-std", info=info)
    print("END.")
