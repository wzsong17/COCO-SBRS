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