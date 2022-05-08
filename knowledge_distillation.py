import numpy as np
import argparse
import os
import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from utilities import get_dataset, load_mask, save_model_history
from model import ResNet, Bottleneck, load_model, conv1x1
from trim import trim_model, trim_params


class Discriminator(nn.Module):
    """
    Vanilla GAN Discriminator
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.25),
            nn.Linear(256, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.layers(x)


def get_studnet_kernel(model):
    num_kernel = {}
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            for param_name, param in module.named_parameters():
                if "conv3" in module_name:
                    num_kernel[module_name] = param.shape[0] // 32
                elif "downsample" in module_name:
                    num_kernel[module_name] = param.shape[0] // 32
                else:
                    num_zero = 0
                    for i in range(param.shape[0]):
                        num_zero += int(torch.all(param[i, :, :, :] == 0).cpu().detach().numpy())
                    num_kernel[module_name] = param.shape[0] - num_zero

    return num_kernel


def get_student_network(num_kernel, model, BLOCKS, CLASS_NUM):
    student_model = ResNet(Bottleneck, BLOCKS)
    num_ftrs = student_model.fc.in_features
    student_model.fc = nn.Linear(num_ftrs, CLASS_NUM)
    prev = 64
    layers = [student_model.layer1, student_model.layer2, student_model.layer3, student_model.layer4]
    for l, layer in enumerate(layers):
        for i in range(BLOCKS[l]):
            conv1_size = max(1, num_kernel[f"layer{l + 1}.{i}.conv1"])
            conv2_size = max(1, num_kernel[f"layer{l + 1}.{i}.conv2"])
            conv3_size = max(1, num_kernel[f"layer{l + 1}.{i}.conv3"])

            layer[i].conv1 = nn.Conv2d(prev, conv1_size, kernel_size=layer[i].conv1.kernel_size,
                                       stride=layer[i].conv1.stride,
                                       padding=layer[i].conv1.padding,
                                       bias=layer[i].conv1.bias)

            layer[i].bn1 = nn.BatchNorm2d(conv1_size, eps=layer[i].bn1.eps,
                                          momentum=layer[i].bn1.momentum,
                                          affine=layer[i].bn1.affine,
                                          track_running_stats=layer[i].bn1.track_running_stats)

            layer[i].conv2 = nn.Conv2d(conv1_size, conv2_size, kernel_size=layer[i].conv2.kernel_size,
                                       stride=layer[i].conv2.stride,
                                       padding=layer[i].conv2.padding,
                                       bias=layer[i].conv2.bias)

            layer[i].bn2 = nn.BatchNorm2d(conv2_size, eps=layer[i].bn2.eps,
                                          momentum=layer[i].bn2.momentum,
                                          affine=layer[i].bn2.affine,
                                          track_running_stats=layer[i].bn2.track_running_stats)

            layer[i].conv3 = nn.Conv2d(conv2_size, conv3_size, kernel_size=layer[i].conv3.kernel_size,
                                       stride=layer[i].conv3.stride,
                                       padding=layer[i].conv3.padding,
                                       bias=layer[i].conv3.bias)

            layer[i].bn3 = nn.BatchNorm2d(conv3_size, eps=layer[i].bn3.eps,
                                          momentum=layer[i].bn3.momentum,
                                          affine=layer[i].bn3.affine,
                                          track_running_stats=layer[i].bn3.track_running_stats)

            if i == 0:  # block 0 of each layer
                downsample_size = num_kernel[f"layer{l + 1}.{i}.downsample.0"]
                layer[i].downsample[0] = nn.Conv2d(prev, downsample_size,
                                                   kernel_size=layer[i].downsample[0].kernel_size,
                                                   stride=layer[i].downsample[0].stride,
                                                   bias=layer[i].downsample[0].bias)
                layer[i].downsample[1] = nn.BatchNorm2d(downsample_size, eps=layer[i].downsample[1].eps,
                                                        momentum=layer[i].downsample[1].momentum,
                                                        affine=layer[i].downsample[1].affine,
                                                        track_running_stats=layer[i].downsample[1].track_running_stats)

            prev = conv3_size

    student_model.fc = nn.Linear(in_features=prev,
                                 out_features=student_model.fc.out_features,
                                 bias=True)

    return student_model


def kd_loss(student_output, teacher_output, student_inter, teacher_inter, regressors, discriminator, name, lambdas):
    T = 20
    lambda1, lambda2, lambda3 = lambdas
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Backbone / Soft Label Loss
    KL_criterion = nn.KLDivLoss()
    soft_label_loss = KL_criterion(F.log_softmax(student_output / T, dim=1),
                                   F.softmax(teacher_output / T, dim=1)) * (lambda1 * T * T)

    # Intermediate Layer Loss
    intermediate_criterion = nn.MSELoss()
    intermediate_loss = []
    for i in range(len(student_inter)):
        if name == "constant":
            s_feat = F.conv2d(student_inter[i], regressors[i]).to(DEVICE)
        elif name == "conv1x1":
            s_feat = regressors[i](student_inter[i]).to(DEVICE)
        else:
            s_feat = student_inter[i]
        t_feat = teacher_inter[i]

        loss = intermediate_criterion(s_feat, t_feat)
        intermediate_loss.append(loss)
    intermediate_loss = sum(intermediate_loss) * lambda2

    # Adversarial Loss
    adv_criterion = nn.BCELoss()
    student_label = 0.0
    b_size = student_output.shape[0]
    label = torch.full((b_size,), student_label, device=DEVICE)
    output = discriminator(student_output.detach()).view(-1)
    d_loss_student = adv_criterion(output, label) * lambda3

    return soft_label_loss + intermediate_loss + d_loss_student


def create_regressors(regressors, DEVICE, name, student_inter, teacher_inter):
    """
    for constant filters:
    teacher: O_h * N_h * N_h
    student: O_g * N_g * N_g
    N_g - k  + 1 = N_h
    k = N_g - N_h + 1

    conv: nn.Conv2d(O_g, O_h, kernel size = k, padding=0)

    """
    for i in range(len(student_inter)):
        channel_t, kernel_size_t = teacher_inter[i].shape[1], teacher_inter[i].shape[2]
        channel_s, kernel_size_s = student_inter[i].shape[1], student_inter[i].shape[2]

        if name == "constant":
            k = kernel_size_s - kernel_size_t + 1
            filters = 1 / k ** 2 * torch.ones(channel_t, channel_s, k, k).to(DEVICE)
            regressors.append(filters)
        elif name == "conv1x1":
            inter_channel = (channel_s + channel_t) // 2
            regressors.append(nn.Sequential(
                conv1x1(channel_s, inter_channel),
                nn.ReLU(),
                conv1x1(inter_channel, inter_channel),
                nn.ReLU(),
                conv1x1(inter_channel, channel_t),
            ).to(DEVICE))
    return regressors


def train_kd(teacher, student, discriminator, epoch_num, dataloaders, data_sizes, batch_size, optimizer, d_optimizer, scheduler, device, regressor_name, lambdas):
    process = tqdm.tqdm(range(epoch_num))
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Freeze teacher model's parameters
    for param in teacher.parameters():
        param.requires_grad = False

    epoch_loss = 1.0
    epoch_acc = 0.0
    regressors = []

    for epoch in process:
        process.set_description(f'Epoch {epoch + 1} / {epoch_num}')

        for phase in ['train', 'val']:
            if phase == 'train':
                student.train()  # Set model to training mode
                data_loader = dataloaders['train']
            else:
                student.eval()  # Set model to evaluate mode
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
                d_optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    teacher_outputs, teacher_intermediate = teacher(inputs)
                    student_outputs, student_intermediate = student(inputs)

                if len(regressors) == 0:
                    regressors = create_regressors(regressors, DEVICE, regressor_name, student_intermediate,
                                                   teacher_intermediate)

                # Adversarial training
                adv_criterion = nn.BCELoss()
                teacher_label = 1.0
                b_size = inputs.shape[0]
                output = discriminator(teacher_outputs).view(-1)
                label = torch.full((b_size,), teacher_label, device=DEVICE)
                d_loss_teacher = adv_criterion(output, label) * lambdas[-1]
                if phase == "train":
                    d_loss_teacher.backward()

                # General Training
                _, preds = torch.max(student_outputs, 1)
                loss = kd_loss(student_outputs, teacher_outputs, student_intermediate,
                                               teacher_intermediate, regressors, discriminator, regressor_name, lambdas)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    d_optimizer.step()

                # statistics
                loss_item = loss.item() * inputs.size(0)
                running_loss += loss_item
                acc_item = torch.sum(preds == labels.data)
                running_corrects += acc_item
                steps += 1
                process.set_description(
                    f'{phase} Step:{steps}/{total_steps} Running_Loss:{loss_item:.4f}\
                    Running_Acc:{acc_item / float(batch_size):.4f} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects.double() / data_size
            process.set_description(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            else:
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
    return student, history


if __name__ == '__main__':
    # root_folder = './model/pruned_model/iter1'
    # save_folder = './model/kd_model'
    parser = argparse.ArgumentParser(description='Get hyper-parameters')
    parser.add_argument("--root_folder", required=True, type=str, help='It should be an existed path')
    parser.add_argument("--save_folder", default='./model/kd_model', type=str, help='It should be an existed path')
    parser.add_argument("--batch_size", default="128", type=int, help='It should be an interger')
    parser.add_argument("--epoch_num", default="200", type=int, help='It should be an interger')
    parser.add_argument("--class_num", default="10", type=int, help='It should be an interger')
    parser.add_argument("--num_worker", default="0", type=int, help='It should be an interger')
    parser.add_argument("--student_learning_rate", default="0.01", type=float, help='It should be an interger')
    parser.add_argument("--discriminator_learning_rate", default="0.0001", type=float, help='It should be an interger')
    parser.add_argument("--lambda1", default="0.7", type=float, help='It should be an float number')
    parser.add_argument("--lambda2", default="0.3", type=float, help='It should be an float number')
    parser.add_argument("--lambda3", default="0.2", type=float, help='It should be an float number')
    parser.add_argument("--regressor_name", default='conv1x1', type=str, choices=['conv1x1', 'constant'],
                        help='It must be either conv1x1 or constant')
    parser.add_argument("--gpu", action="store_false", help='Default value is true')

    args = parser.parse_args()
    ROOT_FOLDER = args.root_folder
    SAVE_FOLDER = args.save_folder
    BATCH_SIZE = args.batch_size
    EPOCH_NUM = args.epoch_num
    CLASS_NUM = args.class_num
    NUM_WORKER = args.num_worker
    STUDENT_LEARNING_RATE = args.student_learning_rate
    DISCRIMINATOR_LEARNING_RATE = args.discriminator_learning_rate
    GPU = args.gpu
    lambdas = (args.lambda1, args.lambda2, args.lambda3)
    regressor_name = args.regressor_name  # "conv1x1" or "constant"

    BLOCKS = [3, 4, 6, 3]
    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and GPU) else "cpu")

    seed = 826
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    dataloaders, data_sizes = get_dataset(BATCH_SIZE, NUM_WORKER)
    criterion = nn.CrossEntropyLoss()

    teacher = load_model(os.path.join(ROOT_FOLDER, 'model.h5'), DEVICE, BLOCKS, CLASS_NUM)
    state_dict = teacher.state_dict()
    mask = load_mask(os.path.join(ROOT_FOLDER, 'mask.npy'))
    teacher = trim_model(teacher, mask)
    state_dict = trim_params(state_dict, mask)
    teacher.load_state_dict(state_dict)
    teacher = teacher.to(DEVICE)

    """Student Model General Training"""
    num_kernel = get_studnet_kernel(teacher)
    student = get_student_network(num_kernel, teacher, BLOCKS, CLASS_NUM)
    student = student.to(DEVICE)

    optimizer = optim.SGD(student.parameters(), lr=STUDENT_LEARNING_RATE, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=55, gamma=0.1)
    discriminator = Discriminator()
    discriminator.to(DEVICE)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE)

    student_model, history = train_kd(teacher, student, discriminator, EPOCH_NUM, dataloaders, data_sizes, BATCH_SIZE, optimizer, d_optimizer, scheduler, DEVICE, regressor_name, lambdas)
    save_model_history(SAVE_FOLDER, student_model, history)
