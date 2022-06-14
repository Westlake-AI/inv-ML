
# Invertible-Manifold-Learning-for-Dimension-Reduction (inv-ML)

## Introduction

The code is a deep neural network implementation of *inv-ML*, training and testing with toy datasets (Swiss roll, S Curve, Sphere) and real-world datasets (MNIST, USPS, KMNIST, Fashion-MNIST, COIL-20, etc.).

## Requirements

* pytorch == 1.3.1
* numpy == 1.17.2
* matplotlib == 3.1.1
* opencv-python == 4.4.0.42
* scikit-learn == 0.21.3
* scipy == 1.3.1
* imageio == 2.6.0
* ipython == 7.8.0

## Description

* main.py  -- Train and test the model defined in the test_config.py.
* test_config.py -- Parameters to define *i-ML-Enc* models.
* dataset
  * dataset.py -- Load data of selected toy datasets.
  * dataloader.py -- Load data of selected real-world datasets.
* models  
  * InvML.py -- Define the MLP-based *inv-ML-Enc* model.
* loss  
  * InvML_loss.py -- Calculate losses of *inv-ML-Enc*: ℒ<sub>LIS</sub>, ℒ<sub>push_away</sub>, ℒ<sub>Extra</sub>, ℒ<sub>Orth</sub>, ℒ<sub>Padding</sub>.
* trainer  
  * invML_trainer.py -- Training loop for the network.
* invMLEnc_toy -- Toy version of *i-ML-Enc* for toy datasets.
* good_params -- Contain config file for all datasets, their results are given in Baidu Netdisk.
* scikit-learn_data -- Contain all datasets for testing *i-ML-Enc*. MNIST, FMNIST, KMNIST, and CIFAR-10 will be downloaded automatically; please download COIL-20, unzip it and put it in this folder.

## Running the code

1. Install the required dependency packages.
2. To get results on six real-world datasets (MNIST, USPS, KMNIST, FMNIST, COIL-20, CIFAR-10), run

  ```python
bash run_test.sh
  ```

3. To get results on two toy datasets (Swiss roll, S Curve), run

  ```python
cd invMLEnc_toy
bash run_test.sh
  ```
Visualization results will be saved in the folder defined in run_test.sh, default are from "Test" to "Test8". You can try different datasets with config files in "./good_params".

## Results

- Visualization of embeddings

  We provide the config file and visualization results of eight datasets in **[[Data_baidu](https://pan.baidu.com/s/1VRatHzJHM3lcIgU1QZONzw)(code:5zfx)]**. Take MNIST as an example: there are "+ExtraHead", "+Orth_loss", "+Padding", and "baseline" in the folder "./MNIST", which contains results and test_config.py.

- Interpolation & reconstruction results

  We also provide the interpolation results of MNIST, USPS, KMNIST, FMNIST in **[[Data_baidu](https://pan.baidu.com/s/1VRatHzJHM3lcIgU1QZONzw)(code:5zfx)]**. You can find them under "./interpolation" and "./reconstruction".

## Acknowledgement

- This repo borrows the architecture design and part of the code from [MLDL](https://github.com/westlake-cairi/Markov-Lipschitz-Deep-Learning).

## Citation

If you are interested in our repository and our paper, please cite the following paper:
```
@inproceedings{Li2021InvertibleML,
  title={Invertible Manifold Learning for Dimension Reduction},
  author={Siyuan Li and Haitao Lin and Zelin Zang and Lirong Wu and Jun Xia and S. Li},
  booktitle={ECML/PKDD},
  year={2021}
}
```
Citation of the implementation of the MLDL paper.

```
@article{Li-MLDL-2020,
  title   = {Markov-Lipschitz Deep Learning},
  author  = {Stan Z Li and Zelin Zang and Lirong Wu},
  journal = {arXiv preprint arXiv:2006.08256},
  year    = {2020}
}
```
