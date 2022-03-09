# RotoGrad for MTL


## Installation

```bash
git clone https://github.com/AvivNavon/rotograd-mtl.git 
cd rotograd-mtl
pip install -e .
```

## NYUv2

Modification of the code in [CAGrad](https://github.com/Cranial-XIX/CAGrad) and [MTAN](https://github.com/lorenmt/mtan).

```bash
python trainer.py --data-path path/to/data
```

## Dataset

The dataset is available at [this link](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0). Put the downloaded files in `./dataset` so that the folder structure is `.dataset/train` and `./dataset/val`.   