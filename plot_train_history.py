import argparse
from utilities import load_history, plot_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get hyper-parameters')
    parser.add_argument("--history_path", required=True, type=str, help='It should be an existed path')

    args = parser.parse_args()
    history_path = args.history_path
    # history_path = './model/prune_model/iter1/history.json'
    history = load_history(history_path)
    plot_history(history)
