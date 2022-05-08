import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import load_model_for_base
from utilities import get_dataset, train, save_model_history, plot_history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get hyper-parameters')
    parser.add_argument("--save_folder", default='./model/base_model', type=str, help='It should be an existed path')
    parser.add_argument("--model_path", default='./model/pretrained_resnet50.h5', type=str,
                        help='It should be an existed path')
    parser.add_argument("--batch_size", default="128", type=int, help='It should be an interger')
    parser.add_argument("--epoch_num", default="200", type=int, help='It should be an interger')
    parser.add_argument("--class_num", default="10", type=int, help='It should be an interger')
    parser.add_argument("--num_worker", default="0", type=int, help='It should be an interger')
    parser.add_argument("--learning_rate", default="0.01", type=float, help='It should be an interger')
    parser.add_argument("--gpu", action="store_false", help='Default value is true')

    args = parser.parse_args()
    SAVE_FOLDER = args.save_folder
    MODEL_PATH = args.model_path
    BATCH_SIZE = args.batch_size
    EPOCH_NUM = args.epoch_num
    CLASS_NUM = args.class_num
    NUM_WORKER = args.num_worker
    LEARNING_RATE = args.learning_rate
    GPU = args.gpu

    BLOCKS = [3, 4, 6, 3]
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and GPU) else "cpu")

    seed = 826
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    ''' 1. Build model '''
    model = load_model_for_base(MODEL_PATH, DEVICE, BLOCKS, CLASS_NUM)

    ''' 2. Create dataloader '''
    dataloaders, data_sizes = get_dataset(BATCH_SIZE, NUM_WORKER)

    ''' 3. Train'''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    model, history = train(model, EPOCH_NUM, dataloaders, data_sizes, BATCH_SIZE, optimizer, criterion, scheduler,
                           DEVICE)
    save_model_history(SAVE_FOLDER, model, history)

    ''' 4. Plot result '''
    plot_history(history)
