# Butterfly Hybrid Detection based on AnomalyFactory
( :construction: UNDER CONSTRUCTION)
![](icons/teaser.png)
[Challenge](https://www.codabench.org/competitions/3764/) | [AnomalyFactory](https://arxiv.org/abs/2408.09533)
## Overview
We propose a solution for Butterfly Hybrid Detection Challenge based on the AnomalyFactory that is an unsupervised anomaly generation framework. The Butterfly Hybrid Detection Challenge provides a training set consisting of 1,991 non-hybrid and 91 hybrid images and a butterfly_anomaly_train.csv file that reveals the subspecies ID of all images. Based on the challenge training set, we train two generative models, AF4hybrid and AF4nonhybrid, to generate 8,874 hybrid and 1,991 non-hybrid images respectively. With the challenge training set and generated images, we use DINOV2 to extract the representative features and further train a sgd classifier sgd_clf.pkl and a linear classifier clf1024.pth. The final result combines the predictions of these two classifiers.
## Structure of this Repository
This repository provides training and testing resource both for the butterfly hybrid detection(HybridDetection) and generation(AnomalyFactory).  
Note: The HybridDetection is independent from the AnomalyFactory that is provided for reproducting the image generative models. 
```
AF4ButterflyHybridDetection
├── HybridDetection
│   ├── requirements.txt
│   ├── submission
│   │   ├── ingestion.py
│   │   ├── metadata
│   │   ├── model.py
│   │   └── requirements.txt
│   └── train
│       ├── classifier.py
│       ├── data_utils.py
│       ├── dataset.py
│       ├── evaluation.py
│       ├── classifier.py
│       ├── model_utils.py
│       └── training.py
│
└── AnomalyFactory/...
```
## Preparetion
1. Download data
   https://drive.google.com/drive/folders/1QMTpK6q29D42IJZcKKkEu999a7w6eTTo?usp=drive_link
```   
Datasets/
│
├── Images4TrainAFhybrid/
│   ├── Images/
│   │   ├── hybrid/
│   │   └── non-hybrid/
│   └── Lists/
│       └── train_AFhybrid/
│
├── Images4TrainAFnonhybrid/
│   ├── Images/
│   │   ├── 0
│   │   ├── ...
│   │   └── 13
│   └── Lists/
│       ├── train_AFnonhybrid/
│       ├── ...
│       └── test_nonhybrid/
│
└── Images4TrainClassifier/
    ├── Images/
    │   ├── hybrid/
    │   └── non-hybrid/
    └── Lists/
        └── butterfly_anomaly_AF.csv
```

2. Download trained models
   https://drive.google.com/drive/folders/1Iv30FCUTJWL10abU4dknADGqNTODVZhM?usp=sharing
```
Models/
│
├── AFgenerators/
│   ├── AF4nonhybrid/
│   │   ├── latest_net_G.pth
│   │   └── latest_net_D.pth
│   └── AF4hybrid/
│       ├── latest_net_G.pth
│       └── latest_net_D.pth
│
└── classifiers/
    ├── clf1024.pth
    └── sgd_clf.pkl
```
3. Download pre-trained models (optional)
   [Edge extractor: PidiNet](https://github.com/hellozhuo/pidinet), [Feature extractor: DINOv2](https://github.com/facebookresearch/dinov2)
## Butterfly Hybrid Detection
### Testing 
Test images with a sgd classifier sgd_clf.pkl and a linear classifier clf1024.pth.
```bash
   cd HybridDetection\submission\dino2B_contestAF_1024linear2cls
   python ingestion.py
```
### Training
The classifiers are trained with the images in Images4TrainClassifier that consists of original contest images and images generated by AnomalyFactory.
```bash
   cd HybridDetection\train
   python training.py
```
## Butterfly Hybrid Generation
### Testing
![](icons/synthetic-hybrid.png)
The synthetic hybrid and non-hybrid images are generated by two generative models, AF4hybrid and AF4nonhybrid.
```bash
   cd AnomalyFactory
   ./scripts/test_AF4hybrid.sh
   ./scripts/test_AF4nonhybrid.sh
```
### Training
The generative models, AF4hybrid and AF4nonhybrid, are trained with images and edge maps from Images4TrainAFhybrid and Images4TrainAFnonhybrid sets respectively. 
The edge maps are extracted by the pre-trained [PidiNet](https://github.com/hellozhuo/pidinet).
```bash
   cd AnomalyFactory
   ./scripts/train_AF4hybrid.sh
   ./scripts/train_AF4nonhybrid.sh
```
## Citation
If you find this useful for your research, please use the following.

```
@article{DBLP:journals/corr/abs-2408-09533,
  author       = {Ying Zhao},
  title        = {AnomalyFactory: Regard Anomaly Generation as Unsupervised Anomaly
                  Localization},
  journal      = {CoRR},
  volume       = {abs/2408.09533},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2408.09533},
  doi          = {10.48550/ARXIV.2408.09533},
  eprinttype    = {arXiv},
  eprint       = {2408.09533},
  timestamp    = {Mon, 30 Sep 2024 13:54:02 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2408-09533.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
