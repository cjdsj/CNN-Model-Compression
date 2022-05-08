import torch
import torch.nn as nn


def trim_model(model, mask):
    layer_ls = [model.layer1, model.layer2, model.layer3, model.layer4]
    layer_name_ls = ['layer1', 'layer2', 'layer3', 'layer4']
    conv_name_ls = ['conv1', 'conv2', 'conv3']
    block_num_ls = [3, 4, 6, 3]
    in_channels = 64

    for i in range(len(layer_ls)):
        layer = layer_ls[i]  # Module
        for block_num in range(block_num_ls[i]):
            block_name = layer_name_ls[i] + '_block' + str(block_num)  # Mask name
            block = layer[block_num]  # Module

            conv_module_ls = [block.conv1, block.conv2, block.conv3]

            for j in range(len(conv_name_ls)):
                conv_name = block_name + '_' + conv_name_ls[j]  # Mask name

                conv_module = conv_module_ls[j]  # Convolution module
                if j != 2:  # Not the last unprunable layer
                    out_channels = mask[conv_name].sum()  # Number of unpruned channels
                else:
                    out_channels = conv_module.out_channels

                kernel_size = conv_module.kernel_size
                stride = conv_module.stride
                padding = conv_module.padding
                conv_module = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

                bn_module = nn.BatchNorm2d(out_channels)  # Batch Normlization module

                if j == 0:
                    block.conv1 = conv_module
                    block.bn1 = bn_module
                elif j == 1:
                    block.conv2 = conv_module
                    block.bn2 = bn_module
                elif j == 2:
                    block.conv3 = conv_module
                    block.bn3 = bn_module

                in_channels = out_channels
            layer[block_num] = block
        if i == 0:
            model.layer1 = layer
        elif i == 1:
            model.layer2 = layer
        elif i == 2:
            model.layer3 = layer
        elif i == 3:
            model.layer4 = layer

    return model


def trim_params(state_dict, mask):
    layer_name_ls = ['layer1', 'layer2', 'layer3', 'layer4']
    conv_name_ls = ['conv1', 'conv2', 'conv3']
    bn_name_ls = ['bn1', 'bn2']
    block_num_ls = [3, 4, 6, 3]

    for i, layer_name in enumerate(layer_name_ls):
        for block_num in range(block_num_ls[i]):
            block_mask_name = layer_name + '_block' + str(block_num)  # Mask name
            block_params_name = layer_name + '.' + str(block_num)  # Parameters name

            conv1_mask_name = block_mask_name + '_' + conv_name_ls[0]  # Mask name
            conv2_mask_name = block_mask_name + '_' + conv_name_ls[1]

            conv1_params_name = block_params_name + '.' + conv_name_ls[0]  # Conv module name
            conv2_params_name = block_params_name + '.' + conv_name_ls[1]
            conv3_params_name = block_params_name + '.' + conv_name_ls[2]
            bn1_params_name = block_params_name + '.' + bn_name_ls[0]  # BN module name
            bn2_params_name = block_params_name + '.' + bn_name_ls[1]  # BN module name

            # Trim Conv layers
            state_dict[conv1_params_name + '.weight'] = state_dict[conv1_params_name + '.weight'][
                mask[conv1_mask_name] == 1]

            state_dict[conv2_params_name + '.weight'] = state_dict[conv2_params_name + '.weight'][
                mask[conv2_mask_name] == 1]
            state_dict[conv2_params_name + '.weight'] = torch.transpose(state_dict[conv2_params_name + '.weight'], 1, 0)
            state_dict[conv2_params_name + '.weight'] = state_dict[conv2_params_name + '.weight'][
                mask[conv1_mask_name] == 1]
            state_dict[conv2_params_name + '.weight'] = torch.transpose(state_dict[conv2_params_name + '.weight'], 1, 0)

            state_dict[conv3_params_name + '.weight'] = torch.transpose(state_dict[conv3_params_name + '.weight'], 1, 0)
            state_dict[conv3_params_name + '.weight'] = state_dict[conv3_params_name + '.weight'][
                mask[conv2_mask_name] == 1]
            state_dict[conv3_params_name + '.weight'] = torch.transpose(state_dict[conv3_params_name + '.weight'], 1, 0)

            # Trim BN layers:
            conv_mask_name_ls = [conv1_mask_name, conv2_mask_name]
            bn_params_name_ls = [bn1_params_name, bn2_params_name]
            for j in range(2):
                conv_mask_name = conv_mask_name_ls[j]
                bn_params_name = bn_params_name_ls[j]
                for attribute in ['weight', 'bias', 'running_mean', 'running_var']:
                    state_dict[bn_params_name + '.' + attribute] = state_dict[bn_params_name + '.' + attribute][
                        mask[conv_mask_name] == 1]
    return state_dict


if __name__ == '__main__':
    from utilities import get_dataset, load_mask, evaluate
    from model import load_model


    model_path = './model/prune_model/iter1/model.h5'
    mask_path = './model/prune_model/iter1/mask.npy'
    BATCH_SIZE = 128
    NUM_WORKER = 0
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, data_sizes = get_dataset(BATCH_SIZE, NUM_WORKER)
    model = load_model(model_path, DEVICE)
    state_dict = model.state_dict()
    mask = load_mask(mask_path)

    criterion = nn.CrossEntropyLoss()
    model_acc = evaluate(model, dataloaders, data_sizes, BATCH_SIZE, criterion, DEVICE)
    print(f'The accuracy is {model_acc} before triming')

    model = trim_model(model, mask)
    state_dict = trim_params(state_dict, mask)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model_acc = evaluate(model, dataloaders, data_sizes, BATCH_SIZE, criterion, DEVICE)
    print(f'The accuracy is {model_acc} after triming')