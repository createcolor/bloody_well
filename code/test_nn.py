from torch.utils.data import DataLoader
from torchvision.models.efficientnet import efficientnet_v2_l
from torchvision.models.densenet import densenet201
from torchvision import transforms
import torch
from pathlib import Path
import argparse

from dataset import DatasetGenerator, ToTensor
import json
from test import test
from dashboard import save_test_info
from sklearn import metrics 
import csv


def parse_args():
    parser = argparse.ArgumentParser(
        'Evaluate estimators on the synthesized test dataset')
    parser.add_argument('-ctest', '--test_config', default=Path('test_config.json'), 
        help='configuration file')
    parser.add_argument('-r', '--dataset_dir', default=Path(
        "../datasets/alvs_dataset/"), type=Path, \
        help='Path to images')
    parser.add_argument('-tm', '--test_markup', default=Path(
        "markup/test_dataset.json"), type=Path, help='Path to test dataset markup')
    parser.add_argument('-np', '--net_path', type=Path, \
        default=Path("models/dataset_aug7_002_240ep_0"), help='Path to net')
    parser.add_argument('-n', '--net_name', type=str, \
        default="mobilenet_v3_large", help='Net name')
    parser.add_argument('-thr', '--threshold', default=0.5, type=float, 
                        help='The best threshold')

    args = parser.parse_args()

    with open(args.test_config, "r") as config_file:
        config = json.load(config_file)
    
    return args, config


def get_net_model(net_arch_name):
    if "efficientnet" in net_arch_name:
        model = efficientnet_v2_l(num_classes=1)
    elif "densenet" in net_arch_name:
        model = densenet201(num_classes=1)
    
    return model, [ToTensor()]


def test_net(device, net, testloader, thr, print_metrics):
    net.load_state_dict(torch.load(net_path, map_location=device))
    net.to(device)
    
    print("\nProcessing test dataset:")
    y_true, y_pred = test(device, net, testloader, threshold=thr, print_metrics=print_metrics)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    acc = metrics.accuracy_score(y_true, y_pred)

    test_info = save_test_info(thr=thr, acc=acc)
    test_info["tn"] = tn
    test_info["fp"] = fp
    test_info["fn"] = fn
    test_info["tp"] = tp
    test_info["precision"] = metrics.precision_score(y_true, y_pred, labels=[1, 0])
    test_info["recall"] = metrics.recall_score(y_true, y_pred, labels=[1, 0])

    return test_info


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
    args, config = parse_args()
    net_name = args.net_name
    path2dataset = args.dataset_dir
    net_path = args.net_path
    
    validation_check(path2dataset, net_path, args.test_markup)
    device = torch.device(config["device_type"] if torch.cuda.is_available() else "cpu")  
    net, transforms_list = get_net_model(args.net_name)
    testloader = get_loader(path2dataset, args.test_markup, transforms_list)

    test_info = test_net(device, net, testloader, args.threshold, config["print_comments"])

    if config["print_comments"]:
        for k in test_info.keys():
            print(f"{k}: {test_info[k]}")

    with open('results.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([args.net_name, test_info["threshold"], test_info["accuracy"],\
                         test_info["recall"], test_info["precision"], test_info["tp"], \
                         test_info["tn"], test_info["fp"], test_info["fn"]])
