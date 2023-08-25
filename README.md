# Bloody Well
Image-based second opinion for blood typing

Bloody Well is the latest dataset for the agglutination reaction classification problem. It includes over 92 plates containing 550 blood samples mixed with 13 different reagents. Information from the medical records of blood donors was used as ground truth, since their groups were confirmed by multiple checks. 

# Dataset description
* 92 plates, each containing 42 wells (6 rows and 7 columns).
* Each row corresponds to one blood sample and each column to one reagent.
* All plates are cut into 3139 well images (529 in test) with a resolution of 512x512.
* Reagents can be used to determine blood group according to the AB0, Rhesus, Kell systems.
* Difficult cases

Reagents info:

<table>
<thead>
  <tr>
    <th rowspan="2">Reagents</th>
    <th colspan="2">AB0</th>
    <th>Rh</th>
    <th colspan="5">Rh: minor</th>
    <th colspan="2">Kell</th>
    <th colspan="4">Control</th>
  </tr>
  <tr>
    <th>A</th>
    <th>B</th>
    <th>D</th>
    <th>E</th>
    <th>e</th>
    <th>C</th>
    <th>c</th>
    <th>Cw</th>
    <th>K</th>
    <th>k</th>
    <th>NaCl 0.9%</th>
    <th>O(I)</th>
    <th>A(II)</th>
    <th>B(III)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Number of wells</td>
    <td>47</td>
    <td>47</td>
    <td>48</td>
    <td>34</td>
    <td>40</td>
    <td>40</td>
    <td>40</td>
    <td>34</td>
    <td>46</td>
    <td>29</td>
    <td>4</td>
    <td>38</td>
    <td>38</td>
    <td>38</td>
  </tr>
  <tr>
    <td>DenseNet201, %</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>90.0</td>
    <td>100</td>
    <td>95.0</td>
    <td>94.1</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>97.4</td>
    <td>100</td>
    <td>100</td>
  </tr>
  <tr>
    <td>EfficientNetV2L, %</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>85.0</td>
    <td>100</td>
    <td>95.0</td>
    <td>94.1</td>
    <td>97.8</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>97.4</td>
  </tr>
</tbody>
</table>

# Availability

Full dataset is not available for public download. You can download only the test part of the dataset, which includes 529 wells and was balanced by reagents with full dataset. It is available here:
* [Bloody Well test part](https://color.iitp.ru/index.php/s/NMYsd58NbTYcPEH) (529 images (228MB) + EfficientNetV2L (450 MB) and DenseNet201 (70 MB) trained on full dataset)

The markup file (test_markup.json) is in this repository. It is a .json file with a dictionary inside, where each image name corresponds to information about it:
* gt_result: ground truth. The absence of an agglutination reaction is indicated as 0, the presence of a reaction as 1.
* reagent: type of reagent used in this well.
* row_idx: number of row in plate (corresponds to one patient)
* col_idx: number of column in plate (corresponds to one reagent)
# Testing

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
