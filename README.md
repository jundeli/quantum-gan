# Quantum GAN with Hybrid Generator
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
* **tensorflow==1.15**
* **frechetdist**

## Structure
* [data](https://github.com/jundeli/quantum-gan/data): should contain your datasets. If you run `download_dataset.sh` the script will download the dataset used for the paper (then you should run `data/sparse_molecular_dataset.py` to conver the dataset in a graph format used by MolGAN models).
* [models](https://github.com/jundeli/quantum-gan/models.py): Class for Models.

## Usage
```
python main.py
```

## Citation
```
[1] J. Li, R. Topaloglu, and S. Ghosh. (2021). Quantum Generative Models for 
Small Molecule Drug Discovery. arXiv preprint arXiv:submit/355091, 2021.
```


BibTeX format:
```
@misc{jundeli2020,
  author = {Li, Junde and Ghosh, Swaroop},
  title = {Quantum-GAN},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jundeli/quantum-gan}}
}

```
