# Bloody Well
Image-based second opinion for blood typing

Bloody Well is the latest dataset for the agglutination reaction classification problem. It includes over 92 plates containing 550 blood samples mixed with 13 different reagents. In total, there are 3139 non-empty wells on all plates. Information from the medical records of blood donors was used as ground truth, since their groups were confirmed by multiple checks. This dataset was used in [ref article].

# Dataset description
* 92 plates, each containing 42 wells (6 rows and 7 columns).
* Each row corresponds to one blood sample and each column to one reagent.
* Reagents can be used to determine blood group according to the AB0, Rhesus, Kell systems.
* Difficult cases

# Availability

Full dataset is not available for public download. You can download only the test part of the dataset, which includes 539 wells and was balanced by reagents with full dataset. It is available here:
* [download ref]
The markup file (test_dataset.json) is in this repository.
# Usage
For testing, use the following command:
```
python3 code/test_nn.py -ctest <path_to_test_config>
```
test_config.json has the following options:
* markup: markup file path.
* dataset: path to dataset.
* net_path: path to neural network you want to test.
* architecture: architecture of used neural network (only EfficientNetV2L or DenseNet201).
* threshold: neural network output number threshold, starting from which the neural network responds that agglutination is observed.
* results: .csv file, where you want to save results.
* net_name: net name used only for results file.
* device_type: pytorch device on which the neural network will run
* print_comments: print some messages during testing
