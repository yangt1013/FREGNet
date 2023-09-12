# FREGNet: Ship recognition based on feature respresentation enhancement and GCN combiner in complex environment

## Requirement
python 3.9

Pytorch >=1.10

torchvision >=0.8

## Training

1. Download datatsets for FREGNet (e.g. MAR-ships, CIB-ships, Game-of-ships etc) and organize the structure as follows:
```bash
dataset

└── train/test

    ├── class_001
    
    |      ├── 1.jpg    
    |      ├── 2.jp
    |      └── ...    
    ├── class_002
    
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...
```
2、Train from scratch with `train.py`.


## Citation
Please cite our paper if you use FREGNet code in your work.
```bash
@InProceedings{du2023fine,
  title={Fine-Grained Ship Recognition for Complex Background Based on Global to Local and Progressive Learning},
  author={Yang Tian; Hao Meng; Fei Yuan}
}
```

## MAR-ships dataset link:
```bash
ARGOS-Venice boat classification

website：https://pan.baidu.com/s/1FJ6j3MUQLqZYP2jpc2p7qA?pwd=fgko  
word：fgko
```


## Game-of-ships dataset link:
```bash
website：https://pan.baidu.com/s/12SvLfHiWxHhEF1sEVDIvuQ?pwd=k2f8 
word：k2f8
```

## Sea-ships dataset link:
```bash
website：http://www.lmars.whu.edu.cn/prof_web/shaozhenfeng/datasets/SeaShips%287000%29.zip
```

## SMD dataset link:
```bash
website：https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset
```
