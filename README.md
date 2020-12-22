# Quantum GAN with Hybrid Generator (QGAN-HG)
PennyLane and Pytorch implementation of QGAN-HG: Quantum generative models for small molecule drug discovery, based on MolGAN (https://arxiv.org/abs/1805.11973)  
This library refers to the following source code.
* [yongqyu/MolGAN-pytorch](https://github.com/yongqyu/MolGAN-pytorch)
* [nicola-decao/MolGAN](https://github.com/nicola-decao/MolGAN)
* [yunjey/StarGAN](https://github.com/yunjey/StarGAN)

## Dependencies

* **python>=3.5**
* **pytorch>=0.4.1**: https://pytorch.org
* **rdkit**: https://www.rdkit.org
* **pennylane**

## Structure
* [data](https://github.com/jundeli/quantum-gan/data): should contain your datasets. If you run `download_dataset.sh` the script will download the dataset used for the paper (then you should run `data/sparse_molecular_dataset.py` to conver the dataset in a graph format used by MolGAN models).
* [models](https://github.com/jundeli/quantum-gan/models.py): Class for Models.

## Usage
```
python main.py
```

## Citation
```
[1] De Cao, N., and Kipf, T. (2018).MolGAN: An implicit generative
model for small molecular graphs. ICML 2018 workshop on Theoretical
Foundations and Applications of Deep Generative Models.
```

BibTeX format:
```
@article{de2018molgan,
  title={{MolGAN: An implicit generative model for small
  molecular graphs}},
  author={De Cao, Nicola and Kipf, Thomas},
  journal={ICML 2018 workshop on Theoretical Foundations
  and Applications of Deep Generative Models},
  year={2018}
}

```
