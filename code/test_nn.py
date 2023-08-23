from torch.utils.data import DataLoader
from torchvision.models.efficientnet import efficientnet_v2_l
from torchvision.models.densenet import densenet201
from torchvision import transforms
import torch
from pathlib import Path
import argparse
import numpy as np
from dataset import DatasetGenerator, ToTensor
import json
from test import test
from sklearn import metrics 
import csv


def parse_args():
    parser = argparse.ArgumentParser(
        'Evaluate estimators on the synthesized test dataset')
    parser.add_argument('-ctest', '--test_config', default=Path('test_config.json'), 
        help='configuration file')

    args = parser.parse_args()

    with open(args.test_config, "r") as config_file:
        config = json.load(config_file)
    
    return config


def get_net_model(net_arch_name):
    if "EfficientNetV2L" in net_arch_name:
        model = efficientnet_v2_l(num_classes=1)
    elif "DenseNet201" in net_arch_name:
        model = densenet201(num_classes=1)
    
    return model, [ToTensor()]


def test_net(device, net, testloader, thr, print_metrics):
    net.load_state_dict(torch.load(net_path, map_location=device))
    net.to(device)
    
    print("\nProcessing test dataset:")
    y_true, y_pred, reagents = test(device, net, testloader, threshold=thr, print_metrics=print_metrics)
    y_true, y_pred, reagents = np.array(y_true), np.array(y_pred), np.array(reagents)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = metrics.accuracy_score(y_true, y_pred)

    test_info = {}
    test_info["threshold"] = thr
    test_info["accuracy"] = acc
    test_info["tn"] = tn
    test_info["fp"] = fp
    test_info["fn"] = fn
    test_info["tp"] = tp
    test_info["precision"] = metrics.precision_score(y_true, y_pred, labels=[1, 0])
    test_info["recall"] = metrics.recall_score(y_true, y_pred, labels=[1, 0])

    reagents_info = {}
    for i in set(reagents):
        idx = np.where(reagents == i)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true[idx], y_pred[idx], labels=[0, 1]).ravel()
        reagents_info.update({i : {"accuracy" : float(metrics.accuracy_score(y_true[idx], y_pred[idx]))}})
        reagents_info[i]["tn"] = int(tn)
        reagents_info[i]["fp"] = int(fp)
        reagents_info[i]["fn"] = int(fn)
        reagents_info[i]["tp"] = int(tp)

    return test_info, reagents_info


def get_loader(path_to_dataset, markup_path, transforms_list, batch_size=1):
    with open(markup_path, "r") as f:
        markup = json.load(f)

    dataset = DatasetGenerator(path_to_dataset, markup, \
        transform=transforms.Compose(transforms_list))
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=8)
    return dataloader


def validation_check(*args):
    for arg in args:
        assert arg.exists(), f"{arg} does not exist"

if __name__ == "__main__":
    config = parse_args()
    net_name = config["net_name"]
    path2dataset = Path(config["dataset"])
    net_path = Path(config["net_path"])
    test_markup = Path(config["markup"])
    threshold = config["threshold"]
    
    validation_check(path2dataset, net_path, test_markup)
    device = torch.device(config["device_type"] if torch.cuda.is_available() else "cpu")  
    net, transforms_list = get_net_model(config["architecture"])
    testloader = get_loader(path2dataset, test_markup, transforms_list)

    test_info, reagents_info = test_net(device, net, testloader, threshold, config["print_comments"])

    if config["print_comments"]:
        print(json.dumps(reagents_info, indent="\t"))

    with open(config["results"], 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([net_name, test_info["threshold"], test_info["accuracy"],\
                         test_info["recall"], test_info["precision"], test_info["tp"], \
                         test_info["tn"], test_info["fp"], test_info["fn"]])
