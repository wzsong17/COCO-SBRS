# coco
python main.py --b 1 --c 1 --sample_size 10 --data delicious --d b0c0s0 --device cuda:3
# gru4rec
python baselines/iirnn/train_rnn.py --data delicious --c comments
# iirnn
python baselines/iirnn/train_ii_rnn.py --data delicious
# hgru4rec
python baselines/hgru/train_hgru.py --data delicious
# srgnn
python baselines/srgnn/main.py --data delicious
# insert
python baselines/insert/insert.py --data delicious
# csrm
python baselines/csrm/main.py --dataset delicious
# stamp
pip install dill

python baselines/stamp/cmain.py -d delicious -e #epoch

# sknn
python baselines/stan/main.py --data delicious --modeltype sknn --k 500
# stan
python baselines/stan/main.py --data delicious --modeltype stan --k 500