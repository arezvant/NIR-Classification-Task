import torch
import shutil
from configuration.config import OUTPUT_PATH
from configuration import config
from dataset.custom_data_loader import RetiSpecDataset
from models.eca_mobilenetv2 import ECA_MobileNetV2
from models.mobilenetv2 import MobileNetV2
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from tqdm import tqdm, notebook
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



device = torch.device("cpu")

def load_checkpoint(filepath, arch=None):
        checkpoint = torch.load(filepath)
        if arch == 'att-chan':
            model = ECA_MobileNetV2(num_classes=2, channels = 4, width_mult=1.0).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
            return model
        elif arch == 'rgb-chan':
            model = MobileNetV2(num_classes = 2, channels = 3, width_mult = 1.0).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
            return model

        elif arch == 'nir-chan':
            model = MobileNetV2(num_classes = 2, channels = 3, width_mult = 1.0).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
            return model

        else:
            model = MobileNetV2(num_classes = 2, channels = 4, width_mult = 1.0).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
            return model

def test(test_dir=None, bs=16, arch=None):
  if arch == 'att-chan':
    model_name = arch
    rgb_n_test_dataset = RetiSpecDataset(test_dir)
    test_loader = DataLoader(rgb_n_test_dataset, batch_size=bs, shuffle=True)
    model = load_checkpoint(OUTPUT_PATH + model_name + '.pth.tar', arch)

    with torch.no_grad():
      CM = 0
      test_accuracy = 0
      loop = tqdm(test_loader)
      for idx, (data, label) in enumerate(loop):
          data = data.to(device)
          label = label.to(device)
          test_output = model(data).to(device)
          acc = (test_output.argmax(1) == label.squeeze()).float().mean()
          test_accuracy += acc / len(test_loader)
          CM += confusion_matrix(label.squeeze(), test_output.argmax(1))
      
      tn = CM[1][1]
      tp = CM[0][0]
      fp = CM[1][0]
      fn = CM[0][1]
      sensitivity = (tp / (tp + fn)) * 100
      precision = (tp / (tp + fp)) * 100
      specificity = (tn / (tn + fp)) * 100
      f1_score = ((2 * sensitivity * precision) / 
      (sensitivity + precision))
      print(f"\nOveral Test Accuracy: {test_accuracy*100:.2f}\n")
      print(f"Sensitivity: {sensitivity}\n")
      print(f"Specificity: {specificity}\n")
      print(f"Precision: {precision}\n")
      print(f"F1-score: {f1_score}\n")
      print(f"Confusion Matirx:\n{CM}\n")
      mlflow.end_run()
      with mlflow.start_run() as run:
          mlflow.log_param("Architecture", arch)
          mlflow.log_metric("Overal Test Accuracy", test_accuracy*100)
          mlflow.log_metric("Sensitivity", sensitivity)
          mlflow.log_metric("Specificity", specificity)
          mlflow.log_metric("Precision", precision)
          mlflow.log_metric("F1-score", f1_score)
          mlflow.pytorch.log_model(model, "models")


  elif arch == 'rgb-chan':
    model_name = arch
    rgb_test_dataset = RetiSpecDataset(test_dir, flag='rgb')
    test_loader = DataLoader(rgb_test_dataset, batch_size=bs, shuffle=True)
    model = load_checkpoint(OUTPUT_PATH + model_name + '.pth.tar', arch)

    with torch.no_grad():
      CM = 0
      test_accuracy = 0
      loop = tqdm(test_loader)
      for idx, (data, label) in enumerate(loop):
          data = data.to(device)
          label = label.to(device)
          test_output = model(data).to(device)
          acc = (test_output.argmax(1) == label.squeeze()).float().mean()
          test_accuracy += acc / len(test_loader)
          CM += confusion_matrix(label.squeeze(), test_output.argmax(1))

      tn = CM[1][1]
      tp = CM[0][0]
      fp = CM[1][0]
      fn = CM[0][1]
      sensitivity = (tp / (tp + fn)) * 100
      precision = (tp / (tp + fp)) * 100
      specificity = (tn / (tn + fp)) * 100
      f1_score = ((2 * sensitivity * precision) / 
      (sensitivity + precision))
      print(f"\nOveral Test Accuracy: {test_accuracy*100:.2f}\n")
      print(f"Sensitivity: {sensitivity}\n")
      print(f"Specificity: {specificity}\n")
      print(f"Precision: {precision}\n")
      print(f"F1-score: {f1_score}\n")
      print(f"Confusion Matirx:\n{CM}\n")
      mlflow.end_run()
      with mlflow.start_run() as run:
          mlflow.log_param("Architecture", arch)
          mlflow.log_metric("Overal Test Accuracy", test_accuracy*100)
          mlflow.log_metric("Sensitivity", sensitivity)
          mlflow.log_metric("Specificity", specificity)
          mlflow.log_metric("Precision", precision)
          mlflow.log_metric("F1-score", f1_score)
          mlflow.pytorch.log_model(model, "models")

  elif arch == 'nir-chan':
      model_name = arch
      nir_test_dataset = RetiSpecDataset(test_dir, flag='nir')
      test_loader = DataLoader(nir_test_dataset, batch_size=bs, shuffle=True)
      model = load_checkpoint(OUTPUT_PATH + model_name + '.pth.tar', arch)

      with torch.no_grad():
          CM = 0
          test_accuracy = 0
          loop = tqdm(test_loader)
          for idx, (data, label) in enumerate(loop):
              data = data.to(device)
              label = label.to(device)
              test_output = model(data).to(device)
              acc = (test_output.argmax(1) == label.squeeze()).float().mean()
              test_accuracy += acc / len(test_loader)
              CM += confusion_matrix(label.squeeze(), test_output.argmax(1))

          tn = CM[1][1]
          tp = CM[0][0]
          fp = CM[1][0]
          fn = CM[0][1]
          sensitivity = (tp / (tp + fn)) * 100
          precision = (tp / (tp + fp)) * 100
          specificity = (tn / (tn + fp)) * 100
          f1_score = ((2 * sensitivity * precision) / 
          (sensitivity + precision))
          print(f"\nOveral Test Accuracy: {test_accuracy*100:.2f}\n")
          print(f"Sensitivity: {sensitivity}\n")
          print(f"Specificity: {specificity}\n")
          print(f"Precision: {precision}\n")
          print(f"F1-score: {f1_score}\n")
          print(f"Confusion Matirx:\n{CM}\n")
          mlflow.end_run()
          with mlflow.start_run() as run:
              mlflow.log_param("Architecture", arch)
              mlflow.log_metric("Overal Test Accuracy", test_accuracy*100)
              mlflow.log_metric("Sensitivity", sensitivity)
              mlflow.log_metric("Specificity", specificity)
              mlflow.log_metric("Precision", precision)
              mlflow.log_metric("F1-score", f1_score)
              mlflow.pytorch.log_model(model, "models")

  else:
      model_name = arch
      mixed_test_dataset = RetiSpecDataset(test_dir)
      test_loader = DataLoader(mixed_test_dataset, batch_size=bs, shuffle=True)
      model = load_checkpoint(OUTPUT_PATH + model_name + '.pth.tar', arch)

      with torch.no_grad():
          CM = 0
          test_accuracy = 0
          loop = tqdm(test_loader)
          for idx, (data, label) in enumerate(loop):
              data = data.to(device)
              label = label.to(device)
              test_output = model(data).to(device)
              acc = (test_output.argmax(1) == label.squeeze()).float().mean()
              test_accuracy += acc / len(test_loader)
              CM += confusion_matrix(label.squeeze(), test_output.argmax(1))

          tn = CM[1][1]
          tp = CM[0][0]
          fp = CM[1][0]
          fn = CM[0][1]
          sensitivity = (tp / (tp + fn)) * 100
          precision = (tp / (tp + fp)) * 100
          specificity = (tn / (tn + fp)) * 100
          f1_score = ((2 * sensitivity * precision) / 
          (sensitivity + precision))
          print(f"\nOveral Test Accuracy: {test_accuracy*100:.2f}\n")
          print(f"Sensitivity: {sensitivity}\n")
          print(f"Specificity: {specificity}\n")
          print(f"Precision: {precision}\n")
          print(f"F1-score: {f1_score}\n")
          print(f"Confusion Matirx:\n{CM}\n")
          mlflow.end_run()
          with mlflow.start_run() as run:
              mlflow.log_param("Architecture", arch)
              mlflow.log_metric("Overal Test Accuracy", test_accuracy*100)
              mlflow.log_metric("Sensitivity", sensitivity)
              mlflow.log_metric("Specificity", specificity)
              mlflow.log_metric("Precision", precision)
              mlflow.log_metric("F1-score", f1_score)
              mlflow.pytorch.log_model(model, "models")