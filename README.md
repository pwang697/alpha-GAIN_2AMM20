# $\alpha$-GAIN: a variance of the Generative Adversarial Imputation Networks (GAIN)

$\alpha$-GAIN is a variance of the Generative Adversarial Imputation Networks (GAIN) algorithm ([Codebase for "Generative Adversarial Imputation Networks (GAIN)"](https://github.com/jsyoon0823/GAIN)), which is created in the course Research Topics in Data Mining ([TU/e](https://www.tue.nl/en/) course code: 2AMM20). Compared with the original hint generator in GAIN, two new mechanisms are involved in $\alpha$-GAIN: fake indication and dynamic hint.

## Installation

- To install dependent packages, activate the experimental environment and type: <br>

```
pip install -r requirements.txt
```

- Known dependencies: Python (3.6.13), torch (1.4.0), numpy (1.21.5), sacred (0.8.2)

## Usage

Firstly, `cd` into the directory, then

- Start an experiment by default: <br>

```
python train.py
```

- [Sacred](https://github.com/IDSIA/sacred) is used to manage configuration and record logging. Results will be automatically saved in a folder named `Experiments`. The configuration can be changed with their CLI, i.e. <br>

```
python train.py with dataset="Popular" repetition=100 mr=0.7 hr=0.8 is_fi=True is_dh=True
```

- System parameters: <br>
  `dataset` choose a dataset (default: `"Letter"`)<br>
  `repetition` repeat the experiment a couple of times (default: `10`)<br>
  `iteration` the number of iterations per repetition (default: `5000`)<br>
  `batch_size` mini-batch size (default: `128`)<br>
  `mr` missing rate (default: `0.5`)<br>
  `hr` hint rate (default: `0.8`)<br>
  `alpha` loss hyperparameters (default: `10`)<br>
  `train_rate` train rate (default: '0.8')<br>
  `is_fi` whether use fake indication (default: `False`)<br>
  `is_dh` whether use dynamic hint rate (default: `False`)

## More Info

The default setting (`is_fi=False is_dh=False`) performs as the GAIN framework, a fully equipped $\alpha$-GAIN should be set with `is_fi=True is_dh=True`. The "Datasets" directory contains three datasets: single-level dataset [Letter](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition), single-level dataset [Spam](https://archive.ics.uci.edu/ml/datasets/Spambase), and multi-level dataset [Popularity](https://github.com/MultiLevelAnalysis/Datasets-third-edition-Multilevel-book/tree/479cec4390efb7f84bb94df0e3c381e173782669/chapter%202/popularity/SPSS).

## Acknowledgement

We would like to thank [Rianne Schouten](https://github.com/RianneSchouten) for her guidance and advice during this research and extensive reviewing of earlier versions of this assignment.
