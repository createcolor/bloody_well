import torch
import numpy as np
from sklearn import metrics

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_net(device, net, testloader, thr=0.5):
    y_true, y_pred = [], []

    net.eval()
    with torch.no_grad():
        for data in testloader:
            imgs, labels, name = data['image'].to(device).float(), data['agg_type'].to(device), \
                data['name'][0]

            outputs = net(imgs)
            outputs = outputs.cpu().detach().numpy()[..., 0]
            outputs = sigmoid(outputs)

            cur_pred = np.where(outputs >= thr, 1, 0)
            y_pred += list(cur_pred)
            
            labels = labels.cpu().detach().numpy().astype(int)
            y_true += list(labels)
    
    return y_true, y_pred

def test(device, net, testloader, threshold=0.5, print_metrics=True):
    
    y_true, y_pred = \
        run_net(device, net, testloader, threshold)
    
    if print_metrics:
        print(metrics.confusion_matrix(y_true, y_pred, labels=[1, 0]))
        print(metrics.classification_report(y_true, y_pred, digits=3))

    return y_true, y_pred
