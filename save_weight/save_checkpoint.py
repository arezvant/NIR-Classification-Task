import torch
import shutil
from configuration.config import OUTPUT_PATH


def save_checkpoint(state, is_best, model_name=None, filename=OUTPUT_PATH + 'checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, OUTPUT_PATH + model_name + '.pth.tar')