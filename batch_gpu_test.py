import torch

print('cuda is available:', torch.cuda.is_available())

if (torch.cuda.is_available()):
    print('cuda device name:', torch.cuda.get_device_name())
