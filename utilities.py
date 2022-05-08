import copy
import os
import json
import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


def get_dataset(batch_size, num_worker):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.CIFAR10(root='./data', train=True if x == 'train' else False,
                                          download=True, transform=data_transforms[x]) for x in {'train', 'val'}}

    data_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                                 num_workers=num_worker) for x in {'train', 'val'}}

    return dataloaders, data_sizes


def train(model, epoch_num, dataloaders, data_sizes, batch_size, optimizer, criterion, scheduler, device,
          save_best_model=True):
    # process = tqdm.notebook.tqdm(range(epoch_num))
    process = tqdm.tqdm(range(epoch_num))
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    epoch_loss = 1.0
    epoch_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in process:
        process.set_description(f'Epoch {epoch + 1} / {epoch_num}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = dataloaders['train']
            else:
                model.eval()  # Set model to evaluate mode
                data_loader = dataloaders['val']

            data_size = data_sizes[phase]
            total_steps = len(data_loader)
            running_loss = 0.0
            running_corrects = 0
            steps = 0

            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs[0], 1)
                    loss = criterion(outputs[0], labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                loss_item = loss.item() * inputs.size(0)
                running_loss += loss_item
                acc_item = torch.sum(preds == labels.data)
                running_corrects += acc_item
                steps += 1
                process.set_description(
                    f'{phase} Step:{steps}/{total_steps} Running_Loss:{loss_item:.4f} Running_Acc:{acc_item / float(batch_size):.4f} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects.double() / data_size
            process.set_description(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:  # Store best model
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            else:
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
    if save_best_model:
        model.load_state_dict(best_model_wts)
    return model, history


def evaluate(model, dataloaders, data_sizes, batch_size, criterion, device):
    data_loader = dataloaders['val']
    data_size = data_sizes['val']
    process = tqdm.tqdm(data_loader)
    # process = tqdm.notebook.tqdm(data_loader)
    model.eval()  # Set model to evaluate mode

    total_steps = len(data_loader)
    running_loss = 0.0
    running_corrects = 0
    steps = 0

    for inputs, labels in process:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs[0], 1)
            loss = criterion(outputs[0], labels)

        # statistics
        loss_item = loss.item() * inputs.size(0)
        running_loss += loss_item
        acc_item = torch.sum(preds == labels.data)
        running_corrects += acc_item
        steps += 1
        process.set_description(f' val Step:{steps}/{total_steps} Running_Loss:{loss_item:.4f} Running_Acc:{acc_item / float(batch_size):.4f}')

    epoch_loss = running_loss / data_size
    epoch_acc = running_corrects.double() / data_size
    process.set_description(f'val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_acc.detach().cpu().item()


def save_mask(layer_dict, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    mask_path = os.path.join(save_folder, 'mask.npy')
    np.save(mask_path, layer_dict)

def load_mask(mask_path):
    return np.load(mask_path, allow_pickle=True).item()


def save_model_history(save_folder, model, history=None):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    history_path = save_folder + '/history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)

    model_path = save_folder + '/model.h5'
    torch.save(model.state_dict(), model_path)


def load_history(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def plot_history(history):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Train Loss')
    plt.xlabel('Epoch Num')
    plt.ylabel('Loss')
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'])
    plt.title('Train ACC')
    plt.xlabel('Epoch Num')
    plt.ylabel('Accuracy')
    plt.subplot(2, 2, 3)
    plt.plot(history['val_loss'])
    plt.title('Val Loss')
    plt.xlabel('Epoch Num')
    plt.ylabel('Loss')
    plt.subplot(2, 2, 4)
    plt.plot(history['val_acc'])
    plt.title('Val ACC')
    plt.xlabel('Epoch Num')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    plt.show()
