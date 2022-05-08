import numpy as np
import os
import json
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utilities import get_dataset, train, evaluate, save_mask, save_model_history
from model import load_model
from prune import compute_apoz, create_mask, prune_all_layers, compute_sparsity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get hyper-parameters')
    parser.add_argument("--root_folder", default='./model/pruned_model', type=str, help='It should be an existed path')
    parser.add_argument("--base_model_path", default='./model/base_model/model.h5', type=str,
                        help='It should be an existed path')
    parser.add_argument("--prune_iter", default="16", type=int, help='It should be an interger')
    parser.add_argument("--batch_size", default="128", type=int, help='It should be an interger')
    parser.add_argument("--epoch_num", default="200", type=int, help='It should be an interger')
    parser.add_argument("--class_num", default="10", type=int, help='It should be an interger')
    parser.add_argument("--num_worker", default="0", type=int, help='It should be an interger')
    parser.add_argument("--learning_rate", default="0.01", type=float, help='It should be an interger')
    parser.add_argument("--save_best_model", action="store_false", help='Default value is true')
    parser.add_argument("--gpu", action="store_false", help='Default value is true')

    args = parser.parse_args()
    PRUNE_ITER = args.prune_iter
    ROOT_FOLDER = args.root_folder
    BASE_MODEL_PATH = args.base_model_path
    BATCH_SIZE = args.batch_size
    EPOCH_NUM = args.epoch_num
    CLASS_NUM = args.class_num
    NUM_WORKER = args.num_worker
    LEARNING_RATE = args.learning_rate
    GPU = args.gpu
    SAVE_BEST_MODEL = args.save_best_model

    BLOCKS = [3, 4, 6, 3]
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and GPU) else "cpu")

    seed = 826
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    ''' 1. Create dataloader '''
    print('*****Creating dataloader...*****')
    dataloaders, data_sizes = get_dataset(BATCH_SIZE, NUM_WORKER)

    ''' 2. Build model '''
    print('*****Loading model...*****')
    model = load_model(BASE_MODEL_PATH, DEVICE, BLOCKS, CLASS_NUM)

    criterion = nn.CrossEntropyLoss()
    result_dict = {'base_acc': [], 'sparsity': [], 'pruned_acc': [], 'final_acc': []}
    old_mask = None
    for step in range(1, PRUNE_ITER + 1):
        print(f'\nIteration {step}:')
        save_folder = os.path.join(ROOT_FOLDER, f'iter{step}')

        ''' 3. Evaluate base model'''
        print('*****Evaluating base model...*****')
        base_acc = evaluate(model, dataloaders, data_sizes, BATCH_SIZE, criterion, DEVICE)
        result_dict['base_acc'].append(base_acc)

        ''' 4. Prune '''
        print('*****Pruning...*****')
        layer_dict = compute_apoz(model, [3, 4, 6, 3], dataloaders, data_sizes, DEVICE)
        mask = create_mask(layer_dict, old_mask)
        prune_all_layers(model, [3, 4, 6, 3], mask)
        save_mask(mask, save_folder)
        old_mask = mask
        sparsity = compute_sparsity(model)
        result_dict['sparsity'].append(sparsity)

        print('*****Evaluating pruned model...*****')
        pruned_acc = evaluate(model, dataloaders, data_sizes, BATCH_SIZE, criterion, DEVICE)
        result_dict['pruned_acc'].append(pruned_acc)

        ''' 5. Finetune '''
        print('*****Fine-tuning...*****')
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=55, gamma=0.1)
        model, history = train(model, EPOCH_NUM, dataloaders, data_sizes, BATCH_SIZE, optimizer, criterion, scheduler,
                               DEVICE, SAVE_BEST_MODEL)
        save_model_history(save_folder, model, history)

        print('*****Evaluating final model...*****')
        final_acc = evaluate(model, dataloaders, data_sizes, BATCH_SIZE, criterion, DEVICE)
        result_dict['final_acc'].append(final_acc)

    with open(os.path.join(ROOT_FOLDER, 'result.json'), 'w') as f:
        json.dump(result_dict, f)
