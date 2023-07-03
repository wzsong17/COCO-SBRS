# COCO-SBRS: A counterfactual collaborative session-based recommender system.

COCO-SBRS was designed to address the problem that the confounder in session-aware/personalized session-based RSs can cause the models to learn spurious correlations in the data.

## environment setup:
python=3.8.16

pytorch=1.13.1

numpy=1.24.3

pandas=1.5.3

tqdm=4.65.0

---
I set up my env using the following commands:
```bash
# cuda=11.7 installed
conda create -n rs python=3.8
conda activate rs
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install numpy pandas tqdm
```
## data preprocessing
You can find an example file for delicious dataset in 'data/raw_data/delicious/' folder.
1. run 'data/raw_data/scripts/1preprocess.py' to process the data.
A 'session_data.pkl' file will be generated.
2. run 'data/raw_data/scripts/2exp_setup.py' to set up the experiments.
It includes the codes to split the training/test set, and generated data files with suitable format for COCO.

## train and evaluate coco
```bash
conda activate rs

# b1 is the boost factor, i.e., \epsilon in eq.17
# b2 is the weight of counterfactual prediction, just used for testing 
# c is the beta (the weight of self-supervised loss) in eq.13 in our paper
# sample_size is the number of counterfactuals, i.e., |\pai (s)|
python main.py --b1 0.1 --b2 1 --c 1 --sample_size 10 --data delicious 
```
The results will be saved in the 'result/' folder.

# Citation
If you find our paper is useful for your research, please consider to cite this paper:
```
[1] Wenzhuo Song, Shoujin Wang, Yan Wang, Kunpeng Liu, Xueyan Liu, and Minghao Yin. 2023. A Counterfactual Collaborative Session-based Recommender System. In Proceedings of the ACM Web Conference 2023 (WWW '23). Association for Computing Machinery, New York, NY, USA, 971–982. https://doi.org/10.1145/3543507.3583321
```
or
```
@inproceedings{10.1145/3543507.3583321,
author = {Song, Wenzhuo and Wang, Shoujin and Wang, Yan and Liu, Kunpeng and Liu, Xueyan and Yin, Minghao},
title = {A Counterfactual Collaborative Session-Based Recommender System},
year = {2023},
isbn = {9781450394161},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3543507.3583321},
doi = {10.1145/3543507.3583321},
abstract = {Most session-based recommender systems (SBRSs) focus on extracting information from the observed items in the current session of a user to predict a next item, ignoring the causes outside the session (called outer-session causes, OSCs) that influence the user’s selection of items. However, these causes widely exist in the real world, and few studies have investigated their role in SBRSs. In this work, we analyze the causalities and correlations of the OSCs in SBRSs from the perspective of causal inference. We find that the OSCs are essentially the confounders in SBRSs, which leads to spurious correlations in the data used to train SBRS models. To address this problem, we propose a novel SBRS framework named COCO-SBRS (COunterfactual COllaborative Session-Based Recommender Systems) to learn the causality between OSCs and user-item interactions in SBRSs. COCO-SBRS first adopts a self-supervised approach to pre-train a recommendation model by designing pseudo-labels of causes for each user’s selection of the item in data to guide the training process. Next, COCO-SBRS adopts counterfactual inference to recommend items based on the outputs of the pre-trained recommendation model considering the causalities to alleviate the data sparsity problem. As a result, COCO-SBRS can learn the causalities in data, preventing the model from learning spurious correlations. The experimental results of our extensive experiments conducted on three real-world datasets demonstrate the superiority of our proposed framework over ten representative SBRSs.},
booktitle = {Proceedings of the ACM Web Conference 2023},
pages = {971–982},
numpages = {12},
keywords = {session-based recommendation, self-supervised learning, counterfactuals},
location = {Austin, TX, USA},
series = {WWW '23}
}
```