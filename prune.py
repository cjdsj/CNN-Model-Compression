import numpy as np
import tqdm
import torch
import torch.nn.utils.prune as prune


class ApozPrune(prune.BasePruningMethod):
    PRUNING_TYPE = "structured"

    def __init__(self, mask, dim=0):
        self.mask = mask == 1
        self.dim = dim

    def compute_mask(self, t, default_mask):
        def make_mask(t, dim, mask_1d):
            mask = torch.zeros_like(t)
            slc = [slice(None)] * len(t.shape)
            slc[dim] = mask_1d
            mask[slc] = 1
            return mask

        mask = make_mask(t, self.dim, self.mask)
        mask *= default_mask.to(dtype=mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, mask, dim=0):
        return super(ApozPrune, cls).apply(module, name, mask=mask, dim=dim)


def apoz(output):
    #  Compute poz per batch
    output = output.cpu().detach().numpy()
    res = np.mean(np.mean(np.sum(np.where(output == 0, 1, 0), axis=0), axis=1), axis=1)
    return res


def compute_apoz(model, block_num_ls, dataloaders, data_sizes, device):
    layer_ls = [model.layer1, model.layer2, model.layer3, model.layer4]
    layer_name_ls = ['layer1', 'layer2', 'layer3', 'layer4']
    conv_name_ls = ['conv1', 'conv2']
    layer_dict = dict()
    data_loader = dataloaders['val']
    data_size = data_sizes['val']

    for inputs, _ in tqdm.tqdm(data_loader):
        # for inputs, _ in tqdm.notebook.tqdm(data_loader):
        inputs = inputs.to(device)
        model(inputs)  # Forward propogation
        for i in range(len(layer_name_ls)):
            layer = layer_ls[i]
            for block_num in range(block_num_ls[i]):
                block = layer[block_num]
                block_name = layer_name_ls[i] + '_block' + str(block_num)
                conv_ls = [block.out1, block.out2]
                for j in range(len(conv_name_ls)):
                    conv_name = block_name + '_' + conv_name_ls[j]
                    if conv_name not in layer_dict.keys():
                        layer_dict[conv_name] = apoz(conv_ls[j])
                    else:
                        layer_dict[conv_name] += apoz(conv_ls[j])  # Compute poz for whole val dataset
    for k, v in layer_dict.items():
        layer_dict[k] = v / data_size  # Compute apoz
    return layer_dict


def create_mask(layer_dict, old_mask=None):
    for k, v in layer_dict.items():
        if old_mask:
            unpruned_v = v[old_mask[k] != 0]  # Ignore pruned filters (old_mask == 0 means has been pruned)
        else:
            unpruned_v = v
        v_mean = unpruned_v.mean()
        v_std = unpruned_v.std()
        layer_dict[k] = np.where(v >= v_mean + v_std, 0, 1)
    return layer_dict


def apply_apoz_prune(mask, conv_module=None, bn_module=None, dim=0):
    if conv_module:
        ApozPrune.apply(conv_module, 'weight', mask, dim)
    if bn_module:
        ApozPrune.apply(bn_module, 'weight', mask, dim)
        ApozPrune.apply(bn_module, 'bias', mask, dim)


def prune_all_layers(model, block_num_ls, layer_dict):
    layer_ls = [model.layer1, model.layer2, model.layer3, model.layer4]
    layer_name_ls = ['layer1', 'layer2', 'layer3', 'layer4']
    conv_name_ls = ['conv1', 'conv2']

    for i in range(len(layer_ls)):
        layer = layer_ls[i]
        for block_num in range(block_num_ls[i]):
            block = layer[block_num]
            block_name = layer_name_ls[i] + '_block' + str(block_num)
            conv_module_ls = [block.conv1, block.conv2]
            bn_module_ls = [block.bn1, block.bn2]
            for j in range(len(conv_name_ls)):
                conv_name = block_name + '_' + conv_name_ls[j]
                mask = layer_dict[conv_name]
                conv_module = conv_module_ls[j]
                bn_module = bn_module_ls[j]
                apply_apoz_prune(mask, conv_module, bn_module)
                prune.remove(conv_module, 'weight')
                prune.remove(bn_module, 'weight')
                prune.remove(bn_module, 'bias')


def compute_sparsity(model):  # Only compute the parameters of conv layers
    num_zeros = 0
    num_elements = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            for param_name, param in module.named_parameters():
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements
    return sparsity
