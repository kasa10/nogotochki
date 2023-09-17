import os

import torch

from torch.autograd import Variable

from .model import Net
from .processing import preprocess_batch, tensor_load_rgbimage, tensor_save_bgrimage


style_model = Net(ngf=128)
model_dict = torch.load('models/style-model.pth')
model_dict_clone = model_dict.copy()
for key, value in model_dict_clone.items():
    if key.endswith(('running_mean', 'running_var')):
        del model_dict[key]
style_model.load_state_dict(model_dict, False)

styles = [torch.load(f"styles/{path}") for path in os.listdir("styles")]


def stylize(image, style):

    content_image = tensor_load_rgbimage(image, size=600, keep_asp=True).unsqueeze(0)

    style_v = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget(style_v)
    output = style_model(content_image)

    return tensor_save_bgrimage(output.data[0])
