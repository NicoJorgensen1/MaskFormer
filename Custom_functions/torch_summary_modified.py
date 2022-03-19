import torch
import torch.nn as nn
import os
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def model_summary(model, input_size, logs=None, batch_size=-1, device="cuda"):

    # Define function to log information about the dataset
    def printAndLog(input_to_write, logs, print_str=True, write_input_to_log=True, prefix="\n", postfix=""):
        if logs != None:                                                                        # If no logfile has been passed, then the inputs will only be printed, not logged
            mode = "a" if os.path.isfile(logs) else "w"                                         # Whether or not we are appending to the logfile or creating a new logfile
            logs = open(file=logs, mode=mode)                                                   # Open the logfile, i.e. making it writable

            # If the input needs to be logged
            if write_input_to_log==True and isinstance(input_to_write, str):
                logs.writelines("{:s}{:s}{:s}".format(prefix, input_to_write, postfix))
            if write_input_to_log==True and isinstance(input_to_write, list):
                logs.writelines("{:s}".format(prefix))
                for string in input_to_write: logs.writelines("{}\n".format(string))
                logs.writelines("{:s}".format(postfix))
            
            # Close the logfile again
            logs.close()
        
        # If the input needs to be printed
        if print_str==True and isinstance(input_to_write, str): print(input_to_write)           # If we need to print the input, we'll print it
        if print_str==True and isinstance(input_to_write, list):                                # If the input is a list, that should be printed ...
            for string in input_to_write: print(string)                                         # ... we'll simply print each time in the list
        

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                ### This line I have edited my self...  => https://stackoverflow.com/questions/68988195/attributeerror-collections-ordereddict-object-has-no-attribute-size
                if isinstance(output, OrderedDict): output = output['out']
                ### This below is the original code
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    try: model(*x)
    except: pass

    # remove these hooks
    for h in hooks:
        try: h.remove()
        except: pass

    printAndLog(input_to_write="Model summary:".upper(), logs=logs, prefix="", postfix="")
    single_dash_line = "-"
    single_dash_line = single_dash_line.join(np.repeat(single_dash_line, 40))
    double_dash_line = "="
    double_dash_line = double_dash_line.join(np.repeat(double_dash_line, 40))
    printAndLog(input_to_write=double_dash_line, logs=logs, prefix="\n", postfix="\n")
    line_new = "{:<28}  {:<27} {:<20}".format("Layer (type)", "Output Shape", "Parameters")
    printAndLog(input_to_write=line_new, logs=logs, prefix="\n", postfix="\n")
    printAndLog(input_to_write=double_dash_line, logs=logs, prefix="\n", postfix="")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:<28}  {:<27} {:<20}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        printAndLog(input_to_write=line_new, logs=logs, prefix="\n", postfix="")

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
    
    printAndLog(input_to_write=double_dash_line, logs=logs)
    printAndLog(input_to_write="Total params: {0:,}".format(total_params), logs=logs)
    printAndLog(input_to_write="Trainable params: {0:,}".format(trainable_params), logs=logs)
    printAndLog(input_to_write="Non-trainable params: {0:,}".format(total_params - trainable_params), logs=logs)
    printAndLog(input_to_write=single_dash_line, logs=logs, prefix="\n", postfix="\n")
    printAndLog(input_to_write="Input size (MB): %0.2f" % total_input_size, logs=logs, prefix="")
    printAndLog(input_to_write="Forward/backward pass size (MB): %0.2f" % total_output_size, logs=logs)
    printAndLog(input_to_write="Params size (MB): %0.2f" % total_params_size, logs=logs)
    printAndLog(input_to_write="Estimated Total Size (MB): %0.2f" % total_size, logs=logs)
    printAndLog(input_to_write=double_dash_line, logs=logs, prefix="\n", postfix="\n")
    # return summary
