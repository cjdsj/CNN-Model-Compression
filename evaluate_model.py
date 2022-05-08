import numpy as np
import torch
import argparse
import random
import torch.nn as nn
from utilities import get_dataset, evaluate
from model import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get hyper-parameters')
    parser.add_argument("--model_path", required=True, type=str, help='It should be an existed path')
    parser.add_argument("--batch_size", default="128", type=int, help='It should be an interger')
    parser.add_argument("--class_num", default="10", type=int, help='It should be an interger')
    parser.add_argument("--num_worker", default="0", type=int, help='It should be an interger')
    parser.add_argument("--gpu", action="store_false", help='Default value is true')

    args = parser.parse_args()
    MODEL_PATH = args.model_path
    BATCH_SIZE = args.batch_size
    CLASS_NUM = args.class_num
    NUM_WORKER = args.num_worker
    GPU = args.gpu

    BLOCKS = [3, 4, 6, 3]
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and GPU) else "cpu")

    seed = 826
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    ''' 1. Create dataloader '''
    dataloaders, data_sizes = get_dataset(BATCH_SIZE, NUM_WORKER)

    ''' 2. Build model '''
    model = load_model(MODEL_PATH, DEVICE, BLOCKS, CLASS_NUM)

    ''' 3. Evaluate '''
    criterion = nn.CrossEntropyLoss()
    acc = evaluate(model, dataloaders, data_sizes, BATCH_SIZE, criterion, DEVICE)
    print(f'Validation accuracy: {acc}')
