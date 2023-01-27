import math
import numpy as np
import torch
import os
from causal import SessionDataset, collate_fn, Recommender
import pickle
from pathlib import Path
import time
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

# from pyinstrument import Profiler


def main():

    # # todo
    # profiler = Profiler()
    # profiler.start()

    #! 处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", default=' ', help='comments')
    parser.add_argument("--data", default='delicious')
    parser.add_argument("--split", default=0, type=int)

    parser.add_argument("--load", default=False)
    parser.add_argument("--save_path", default='model.pkl')

    parser.add_argument("--sampling", default='recent')
    parser.add_argument("--sample_size", default=10, type=int)
    parser.add_argument("--max_len_recent", default=10, type=int)
    parser.add_argument("--c", default=1, type=float)
    parser.add_argument("--b1", default=0.1, type=float)
    parser.add_argument("--b2", default=1, type=float)

    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--n_epoch", default=20, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--l2", default=1e-5, type=float)
    parser.add_argument("--eval_every", default=5,
                        help='evalutate every 5 epochs', type=int)
    parser.add_argument("--no_cuda", default=False)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--seed", default=2022)

    parser.add_argument("--weighting", default='div')
    parser.add_argument("--user_key", default='user')
    parser.add_argument("--item_key", default='item')
    parser.add_argument("--time_key", default='timestamp')
    parser.add_argument("--session_key", default='sessionId')
    args = parser.parse_args()

    #! 初始化其他参数
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')
    PRJ_PATH = Path(__file__).resolve().parent
    DATA_PATH = PRJ_PATH / 'data'

    #! dataset处理
    # stime = time.time()
    with open(DATA_PATH/f'fm_{args.data}_{args.split}.pkl', 'rb') as f:
        trainset, testset = pickle.load(f)

    # todo test code
    load_dataset = False
    if not load_dataset:
        dataset = SessionDataset(trainset, testset, user_key=args.user_key, item_key=args.item_key, session_key=args.session_key,
                                 time_key=args.time_key, max_len_recent=args.max_len_recent, device=device,
                                 sample_size=args.sample_size, sampling=args.sampling)
        with open(f'dataset_{args.data}.pkl', 'wb') as f:
            pickle.dump(dataset, f)
            print('write dataset to disk')
    else:
        print('load data')
        with open(f'dataset_{args.data}.pkl', 'rb') as f:
            dataset = pickle.load(f)
    # profiler.stop()
    # profiler.print()
    # exit()
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda p: collate_fn(p, device=device))

    model = Recommender(args.dim, dataset.user_number,
                        dataset.item_number, device=device)
    model = model.to(device)

    # todo load model from file
    if args.load and os.path.exists(args.save_path):
        model.load_state_dict(torch.load(
            args.save_path, map_location=device))
        return
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2)

    n_batch = (dataset.session_number // args.batch_size) + 1

    best_recall20 = 0
    best_test = ''
    log_file = open(PRJ_PATH/'results' /
                    f'my-{args.data}-ss{args.sample_size}-c{args.c:.1f}-b{args.b1:.1f}-{args.d}.txt', 'w')
    for k, v in args.__dict__.items():
        print(f'{str(k)}:  {str(v)}', file=log_file, flush=True)
        print(f'{str(k)}:  {str(v)}', flush=True)
    patience_idx = 0

    #! train
    for it in range(args.n_epoch):
        print(f'Epoch: {(it + 1)} starts...')
        print(f'Epoch: {(it + 1)}', file=log_file, flush=True)
        loss_list = []
        loss_plist = []
        loss_clist = []
        idx_batch = 1
        for users, session, recent, all_item in train_loader:
            model.train()
            optimizer.zero_grad()

            # filtering samples have empty recent item set
            users, session, recent, all_item = mask_no_recent(
                users, session, recent, all_item)
            items_to_score, map_label = session.unique(return_inverse=True)
            # nonzero = items_to_score != 0
            # items_to_score = items_to_score[nonzero]
            scores, hu, hc, attn = model(
                users, session[:, :-1], recent, items_to_score)

            loss_predict = model.loss_predict1(
                scores, session, map_label)
            loss = loss_predict
            loss_plist.append(loss_predict.item())

            if args.c > 0:
                # loss_contrst = model.loss_contrastive(
                #     all_item, session[:, :-1], hu, hc)
                # loss = loss + model.c * loss_contrst
                # loss_clist.append(loss_contrst.item())

                loss_causal = model.loss_causal(
                    attn, session[:, :-1], all_item, items_to_score
                )
                loss = loss + args.c * loss_causal
                loss_clist.append(loss_causal.item())

            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            print(f'\rbatch {idx_batch}/{n_batch} loss: {loss.item():.5f}',
                  end='')  # loss1: {loss_predict.item():.5f} loss2: {loss_contrst.item():.5f}

            idx_batch += 1
            assert torch.isnan(model.item_embedding.weight).sum() == 0
            assert torch.isnan(model.user_embedding.weight).sum() == 0
        epoch_loss = np.array(loss_list).mean()
        print(
            f'\nEpoch: {(it + 1)} total loss: {epoch_loss}, loss1 {np.array(loss_plist).mean()}, loss2 {np.array(loss_clist).mean()}')
        print('====================================================================')

        #! evaluation
        l = []  # 每个测试session，百分之多少的target item在item to predict中
        if True:  # it > 0 and (it+1)//args.eval_every == 0:
            print('Start evaluation...')
            print(f'total: {len(dataset.session_ids_test)} sessions to test.')
            tester_val = Tester(k_list=[5, 10, 20])
            tester_tst = Tester(k_list=[5, 10, 20])
            for idx, test_session in tqdm(enumerate(dataset.session_ids_test)):
                session_item, users, recent, similarity, items_to_boost = dataset.next_test_session(
                    test_session, device)

                session_context = session_item[:-1].unsqueeze(0)
                with torch.no_grad():
                    scores = model.test_session(users, session_context,
                                                recent, similarity)
                    scores = scores * args.b2
                    # scores = scores + float(args.b1) * \
                    #     items_to_boost[np.newaxis, :]
                    boost_score = float(args.b1) * \
                        items_to_boost[np.newaxis, :]
                    # if float(args.b2) > 0:  # boost context item neighbors
                    #     items_to_boost2 = dataset.next_test_session2(
                    #         test_session)  # context boost item
                    #     boost_score = boost_score + \
                    #         float(args.b2) * items_to_boost2
                    #     boost_score[boost_score > max(float(args.b1), float(args.b2))] = max(
                    #         float(args.b1), float(args.b2))
                    scores = scores + boost_score

                    sorted_score_index = scores.argsort(1)[:, ::-1]
                    predicted_seq = sorted_score_index + 1  # todo 只适用于预测所有item
                    # predicted_seq = np.arange(1, (len(dataset.items)+1))[
                    #     sorted_score_index]
                    # predicted_seq = items_to_predict.detach().cpu().numpy()[
                    #     sorted_score_index]
                    target_seq = session_item[1:].detach().cpu().numpy()
                # l.append((np.isin(np.unique(target_seq), np.unique(
                #     predicted_seq)).sum()/len(np.unique(target_seq))))
                # l.extend(items_to_boost[target_seq-1])
                # assert (np.isin(target_seq, np.unique(
                #     predicted_seq)).sum()/len(target_seq)) == 1
                # rank = np.nonzero(predicted_seq == target_seq[:, None])[1]
                # print(rank) # todo ground label rank
                if idx % 2 == 0:
                    tester_val.evaluate_sequence(
                        predicted_seq, target_seq, len(target_seq))
                else:
                    tester_tst.evaluate_sequence(
                        predicted_seq, target_seq, len(target_seq))

            score_message, recall5, recall20 = tester_val.get_stats()
            # np.savetxt('a.txt', np.array(l)[:, None])
            # print(np.mean(l))
            print(score_message)
            print(score_message, file=log_file, flush=True)
            if recall20 > best_recall20:
                patience_idx = 0
                print('better recall@20!')
                print(f'better recall@20!', file=log_file, flush=True)
                best_recall20 = recall20
                best_test, _, _ = tester_tst.get_stats()
                print('Result on test set:')
                print(best_test, file=log_file, flush=True)
                print(best_test)
            else:
                patience_idx += 1
                print(f'patience {patience_idx}|{args.patience}')
        if patience_idx >= args.patience:
            print(f'early stop!')
            break

    print('train finished. \nbest performence:')
    print(f'train finished.', file=log_file, flush=True)
    print(best_test)
    print(best_test, file=log_file, flush=True)
    torch.save(model.state_dict(), args.save_path)
    print('result saved in: ', PRJ_PATH/'results' /
          f'my-{args.data}-ss{args.sample_size}-c{args.c:.1f}-b{args.b1:.1f}-{args.d}.txt')
    log_file.close()


def mask_no_recent(user_ids, sess_item, rcnt_item, all_item):
    '''
    过滤掉recnet_item为空的sample
    '''
    mask = rcnt_item.sum(1) != 0
    if mask.sum() == len(mask):
        return user_ids, sess_item, rcnt_item, all_item
    # if mask.sum() == 0:
    #     print('no sample has recent items')
    user_ids_ = user_ids[mask]
    sess_item_ = sess_item[mask]
    sess_len = (sess_item_ != 0).sum(1).max()
    sess_item_ = sess_item_[:, -sess_len:]
    rcnt_item_ = rcnt_item[mask]
    all_item_ = [d for d, m in zip(all_item, mask) if m]

    return user_ids_, sess_item_, rcnt_item_, all_item_


class Tester:
    def __init__(self, session_length=4, k_list=[5, 10, 20]):
        self.k_list = k_list
        self.session_length = session_length
        self.n_decimals = 4
        self.initialize()

    def initialize(self):
        self.i_count = np.zeros(self.session_length)  # [0]*self.session_length
        # [[0]*len(self.k) for i in range(self.session_length)]
        self.recall = np.zeros((self.session_length, len(self.k_list)))
        # [[0]*len(self.k) for i in range(self.session_length)]
        self.mrr = np.zeros((self.session_length, len(self.k_list)))
        self.ndcg = np.zeros((self.session_length, len(self.k_list)))

    def get_rank(self, target, predictions):
        for i in range(len(predictions)):
            if target == predictions[i]:
                return i+1

        raise Exception("could not find target in sequence")

    def evaluate_sequence(self, predicted_sequence, target_sequence, seq_len):
        for i in range(min(self.session_length, seq_len)):
            target_item = target_sequence[i]
            k_predictions = predicted_sequence[i]

            for j in range(len(self.k_list)):
                k = self.k_list[j]
                if target_item in k_predictions[:k]:
                    self.recall[i][j] += 1
                    rank = self.get_rank(target_item, k_predictions[:k])
                    self.mrr[i][j] += 1.0/rank
                    self.ndcg[i][j] += 1 / math.log(rank + 1, 2)
            self.i_count[i] += 1

    def evaluate_batch(self, predictions, targets, sequence_lengths):
        for batch_index in range(len(predictions)):
            predicted_sequence = predictions[batch_index]
            target_sequence = targets[batch_index]
            self.evaluate_sequence(
                predicted_sequence, target_sequence, sequence_lengths[batch_index])

    def get_stats(self):
        score_message = "Position\tR@5   \tMRR@5 \tNDCG@5\tR@10   \tMRR@10\tNDCG@10\tR@20  \tMRR@20\tNDCG@20\n"
        current_recall = np.zeros(len(self.k_list))
        current_mrr = np.zeros(len(self.k_list))
        current_ndcg = np.zeros(len(self.k_list))
        current_count = 0
        recall_k = np.zeros(len(self.k_list))
        for i in range(self.session_length):
            score_message += "\ni<="+str(i+2)+"    \t"
            current_count += self.i_count[i]
            for j in range(len(self.k_list)):
                current_recall[j] += self.recall[i][j]
                current_mrr[j] += self.mrr[i][j]
                current_ndcg[j] += self.ndcg[i][j]

                r = current_recall[j]/current_count
                m = current_mrr[j]/current_count
                n = current_ndcg[j]/current_count

                score_message += str(round(r, self.n_decimals))+'\t'
                score_message += str(round(m, self.n_decimals))+'\t'
                score_message += str(round(n, self.n_decimals))+'\t'

                recall_k[j] = r

        recall5 = recall_k[0]
        recall20 = recall_k[2]

        return score_message, recall5, recall20


if __name__ == '__main__':
    main()

# class trainer:
#     '''
#     Parameters
#     -----------
#     k : int
#         Number of neighboring session to calculate the item scores from. (Default value: 100)
#     sample_size : int
#         Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
#     sampling : string
#         String to define the sampling method for sessions (recent, random). (default: recent)
#     similarity : string
#         String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
#     weighting : string
#         Decay function to determine the importance/weight of individual actions in the current session (linear, same, div, log, quadratic). (default: div)
#     weighting_score : string
#         Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic). (default: div_score)
#     weighting_time : boolean
#         Experimental function to give less weight to items from older sessions (default: False)
#     dwelling_time : boolean
#         Experimental function to use the dwelling time for item view actions as a weight in the similarity calculation. (default: False)
#     last_n_days : int
#         Use only data from the last N days. (default: None)
#     last_n_clicks : int
#         Use only the last N clicks of the current session when recommending. (default: None)
#     extend : bool
#         Add evaluated sessions to the maps.
#     normalize : bool
#         Normalize the scores in the end.
#     session_key : string
#         Header of the session ID column in the input file. (default: 'SessionId')
#     item_key : string
#         Header of the item ID column in the input file. (default: 'ItemId')
#     time_key : string
#         Header of the timestamp column in the input file. (default: 'Time')
#     user_key : string
#         Header of the user ID column in the input file. (default: 'UserId')

#     extend_session_length: int
#         extend the current user's session

#     extending_mode: string
#         how extend the user session (default: lastViewed)
#         lastViewed: extend the current user's session with the his/her last viewed items #TODO: now it saves just X last items, and they might be exactly the same: can try as well: save 5 distinctive items
#         score_based: higher score: if the items appeared in more previous sessions AND more recently #TODO

#     boost_own_sessions: double
#         to increase the impact of (give weight more weight to) the sessions which belong to the user. (default: None)
#         the value will be added to 1.0. For example for boost_own_sessions=0.2, weight will be 1.2

#     past_neighbors: bool
#         Include the neighbours of the past user's similar sessions (to the current session) as neighbours (default: False)

#     reminders: bool
#         Include reminding items in the (main) recommendation list. (default: False)

#     remind_strategy: string
#         Ranking strategy of the reminding list (recency, session_similarity). (default: recency)

#     remind_sessions_num: int
#         Number of the last user's sessions that the possible items for reminding are taken from (default: 6)

#     reminders_num: int
#         length of the reminding list (default: 3)

#     remind_mode: string
#         The postion of the remining items in recommendation list (top, end). (default: end)

#     '''

#     def __init__(self, k, sample_size=1000, sampling='random',  weighting='div', last_n_days=None, last_n_clicks=None,
#                  normalize=True, device='cpu',
#                  session_key='SessionId', item_key='ItemId', time_key='Time', user_key='UserId',
#                  dim=64, batch_size=128, n_epoch=50, lr=1e-2, l2=1e-5,
#                  c=0.1, margin=0.2):

#         self.k = k
#         # ! 注意与原实现不同，当self.sample_size=0的时候，验证阶段只使用当前用户当前context，加模型来预测next item，不再使用knn思想
#         self.sample_size = sample_size  # 验证时，选择的最大相关context个数,
#         self.sampling = sampling  # 验证时，采样相关context的策略，包含recent random
#         # 验证阶段计算recent items相似度，包括div, same, log, quadratic, linear, 实现见self对应function
#         self.weighting = weighting
#         self.session_key = session_key  # 以下四个key是dataframe的列名
#         self.item_key = item_key
#         self.time_key = time_key
#         self.user_key = user_key
#         self.normalize = normalize  # 归一化结果
#         self.last_n_days = last_n_days
#         self.last_n_clicks = last_n_clicks  # 当前session只考虑最近多少个click

#         # updated while recommending
#         self.session = -1  # 验证时，session id
#         self.user = -1  # 验证session的user
#         self.session_items = []  # 验证session的item列表
#         self.recent_items = []  # 验证session的用户的recent item list
#         # 验证session的相关用户列表，根据当前session的context中的每个item，寻找其他包含该item的session
#         self.relevant_sessions = set()
#         self.recent_neighbors = set()

#         self.load = True
#         # self.load = False
#         self.save_path = Path(__file__).parents[2] / 'model.pkl'
#         # model parameterss
#         self.dim = dim  # dim
#         self.device = device
#         print(f'use device: {device}.')
#         self.lr = lr
#         self.l2 = l2
#         self.n_epoch = n_epoch
#         self.batch_size = batch_size
#         self.c = c  # weight of contrastive loss
#         self.margin = margin

#     def prepare_for_predict(self, dataset):
#         self.session_item_map = dataset.session_item_map
#         self.item_session_map = dataset.item_session_map
#         self.item_user_map = dataset.item_user_map
#         self.user_item_map = dataset.user_item_map
#         self.session_user_map = dataset.session_user_map
#         self.user_session_map = dataset.user_session_map
#         self.session_time = dataset.session_time
#         self.session_recent_map = dataset.session_recent_map
#         self.recent_session_map = dataset.recent_session_map
#         self.user_recent_map = dataset.user_recent_map
#         self.max_len_recent = dataset.max_len_recent
#         self.user2id = dataset.user2id
#         self.item2id = dataset.item2id
#         self.id2item = dataset.id2item

#     def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, timestamp=0):
#         """
#         Gives predicton scores for a selected set of items on how likely they be the next item in the session.

#         Parameters
#         --------
#         session_id : int or string
#             The session IDs of the event.
#         input_item_id : int or string
#             The item ID of the event. Must be in the set of item IDs of the training set.
#         predict_for_item_ids : 1D array
#             IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

#         Returns
#         --------
#         out : pandas.Series
#             Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

#         """
#         device = 'cpu'  # 验证
#         predict_for_item_ids = [self.id2item[i] for i in range(
#             1, len(self.item2id)+1)]  # 将待预测的item按照item embedding顺序重排列

#         if self.session != session_id:  # new session

#             # add evaluated sessions to the maps.
#             # ! 问题：上一个session的item并不完整，因为predict_next并未扫描到每个session最后一个item
#             item_set = set(self.session_items)
#             if len(item_set) > 0:
#                 self.session_item_map[self.session] = item_set
#                 for item in item_set:
#                     map_is = self.item_session_map.get(item)
#                     if map_is is None:
#                         map_is = set()
#                         self.item_session_map.update({item: map_is})
#                     map_is.add(self.session)

#                 ts = time.time()
#                 self.session_time.update({self.session: ts})
#                 self.session_user_map.update({self.session: self.user})

#                 # 先更新recent map
#                 self.session_recent_map.update(
#                     {self.session: self.recent_items.copy()})  # session id到包含的item列表的映射
#                 for item in set(self.recent_items):  # 构造recent记录中item到session id的映射
#                     if item != 0:
#                         if item in self.recent_session_map:
#                             self.recent_session_map[item].add(self.session)
#                         else:
#                             self.recent_session_map[item] = {self.session}

#                 # 再更新recent
#                 self.recent_items = self.user_recent_map.get(self.user,
#                                                              [])  # 将上一个处理的session内容更新到recent_items，对应上一个处理的session和user
#                 self.recent_items.extend(self.session_items)
#                 if self.max_len_recent > 0:
#                     self.recent_items = self.recent_items[-self.max_len_recent:]
#                 self.user_recent_map.update(
#                     {self.user: self.recent_items.copy()})  # user id到recent item的映射

#             # 新验证session, 初始化验证session的信息
#             self.last_ts = -1
#             self.session = session_id
#             self.session_items = list()  # 验证session的所有item的list
#             self.user = self.user2id[input_user_id]
#             self.recent_items = self.user_recent_map[self.user]
#             if self.sample_size > 0:
#                 self.recent_neighbors = self.recent_neighbor_sessions(
#                     self.recent_items)
#             self.relevant_sessions = set()

#         item = self.item2id[input_item_id]
#         # 将扫描到的item加入到当前session的item列表，input_item_id是context最近的一个item
#         self.session_items.append(item)

#         # 只考虑最后n个click
#         self.session_items = self.session_items if self.last_n_clicks is None else self.session_items[
#             -self.last_n_clicks:]
#         sess_item = torch.LongTensor(
#             self.session_items).unsqueeze(0).to(device)

#         if self.sample_size > 0:
#             # 根据当前session的context，查找可能相关的其它用户的context：对于context中的每个item，查找包含此item的session
#             neighbors = self.find_neighbors(item, session_id, self.user)
#             # 同时考虑根据context查找的和根据recent items查找的neighbors
#             neighbors = neighbors | self.recent_neighbors

#             # 计算recent items相似度，取相似度最高的k个
#             _, similarities = self.calc_similarity(
#                 self.recent_items, neighbors)
#             sorted_idx = np.argsort(similarities)
#             similarities = similarities[sorted_idx[-self.k:]]  # 取最相似的k个
#             # similarities = np.append(similarities, similarities.sum())  # todo，暂定当前用户的预测结果权重占一半
#             similarities = similarities / similarities.sum()

#             rcnt_items = np.array([self.session_recent_map[sess_id] for sess_id in neighbors])[
#                 sorted_idx[-self.k:]]  # 过滤掉超出top k的
#             # rcnt_items = np.append(rcnt_items, self.recent_items).reshape(-1, rcnt_items.shape[1])  # 补充当前用户
#             rcnt_items = torch.LongTensor(rcnt_items).to(device)

#             # 构造输入，user id和recent items使用其他用户的，当前session context使用当前用户的
#             user_ids = np.array([self.session_user_map[sess_id] for sess_id in neighbors])[
#                 sorted_idx[-self.k:]]  # 过滤掉超出top k的
#             # user_ids = np.append(user_ids, self.user)  # 补充当前用户 预测结果包括当前用户的recent items+当前session context
#             user_ids = torch.LongTensor(user_ids).to(device)

#         else:  # 若sample_size==0, 只使用当前context和recent items，加上模型来预测
#             user_ids = torch.LongTensor([self.user]).to(device)
#             rcnt_items = torch.LongTensor([self.recent_items]).to(device)
#             similarities = np.ones(1)
#             neighbors = set()

#         if len(neighbors) > 0:
#             items_to_predict = set()
#             for s in neighbors:
#                 items_to_predict = items_to_predict | set(
#                     self.session_item_map[s])
#             items_to_predict = torch.LongTensor(
#                 list(items_to_predict)).to(device)
#         else:
#             items_to_predict = torch.LongTensor(
#                 list(map(self.item2id.get, predict_for_item_ids))).to(device)

#         # 模型预测查找到的相似session context下，不同recent item下在同一context下用户的选择
#         self.model.to(device)  # faster than cal with gpu
#         self.model.eval()
#         with torch.no_grad():
#             scores, _, _ = self.model(
#                 user_ids, sess_item, rcnt_items, items_to_predict)
#         # (num_neighbor+1)*num_item2predict, pytorch tensor to numpy list
#         scores = scores.detach().cpu().numpy()

#         # 根据相似度聚合，若sample_size==0, 则similarities = [1]
#         scores = (scores * similarities[:, np.newaxis]).sum(0)

#         # scores = self.score_items(neighbors, items, timestamp) # 返回一个item id:score的dict

#         # Create things in the format ..
#         predictions = np.zeros(len(self.item2id))
#         for item, score in zip(items_to_predict, scores):
#             predictions[(item - 1)] = score

#         if self.normalize:  # scale to 0-1
#             predictions = predictions - predictions.max()
#             predictions = np.exp(predictions)
#         # mask = np.in1d(predict_for_item_ids, list(scores.keys()))

#         # predict_for_items = predict_for_item_ids[mask]
#         # values = [scores[x] for x in predict_for_items]
#         # predictions[mask] = values
#         series = pd.Series(data=predictions, index=predict_for_item_ids)

#         return series

#     def mask_no_recent(self, user_ids, sess_item, rcnt_item, all_item):
#         '''
#         过滤掉recnet_item为空的sample
#         '''
#         mask = rcnt_item.sum(1) != 0
#         if mask.sum() == len(mask):
#             return user_ids, sess_item, rcnt_item, all_item
#         # if mask.sum() == 0:
#         #     print('no sample has recent items')
#         user_ids_ = user_ids[mask]
#         sess_item_ = sess_item[mask]
#         rcnt_item_ = rcnt_item[mask]
#         all_item_ = [d for d, m in zip(all_item, mask) if m]

#         return user_ids_, sess_item_, rcnt_item_, all_item_

#     def item_pop(self, sessions):
#         '''
#         Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)

#         Parameters
#         --------
#         sessions: set

#         Returns
#         --------
#         out : dict
#         '''
#         result = dict()
#         max_pop = 0
#         for session, weight in sessions:
#             items = self.items_for_session(session)
#             for item in items:

#                 count = result.get(item)
#                 if count is None:
#                     result.update({item: 1})
#                 else:
#                     result.update({item: count + 1})

#                 if (result.get(item) > max_pop):
#                     max_pop = result.get(item)

#         for key in result:
#             result.update({key: (result[key] / max_pop)})

#         return result

#     def jaccard(self, first, second):
#         '''
#         Calculates the jaccard index for two sessions

#         Parameters
#         --------
#         first: Id of a session
#         second: Id of a session

#         Returns
#         --------
#         out : float value
#         '''
#         sc = time.clock()
#         intersection = len(first & second)
#         union = len(first | second)
#         res = intersection / union

#         self.sim_time += (time.clock() - sc)

#         return res

#     def cosine(self, first, second):
#         '''
#         Calculates the cosine similarity for two sessions

#         Parameters
#         --------
#         first: Id of a session
#         second: Id of a session

#         Returns
#         --------
#         out : float value
#         '''
#         li = len(first & second)
#         la = len(first)
#         lb = len(second)
#         result = li / sqrt(la) * sqrt(lb)

#         return result

#     def tanimoto(self, first, second):
#         '''
#         Calculates the cosine tanimoto similarity for two sessions

#         Parameters
#         --------
#         first: Id of a session
#         second: Id of a session

#         Returns
#         --------
#         out : float value
#         '''
#         li = len(first & second)
#         la = len(first)
#         lb = len(second)
#         result = li / (la + lb - li)

#         return result

#     def binary(self, first, second):
#         '''
#         Calculates the ? for 2 sessions

#         Parameters
#         --------
#         first: Id of a session
#         second: Id of a session

#         Returns
#         --------
#         out : float value
#         '''
#         a = len(first & second)
#         b = len(first)
#         c = len(second)

#         result = (2 * a) / ((2 * a) + b + c)

#         return result

#     def vec(self, first, second, map):
#         '''
#         Calculates the ? for 2 sessions

#         Parameters
#         --------
#         first: Id of a session
#         second: Id of a session

#         Returns
#         --------
#         out : float value
#         '''
#         a = first & second
#         sum = 0
#         for i in a:
#             sum += map[i]

#         result = sum / len(map)

#         return result

#     def items_for_session(self, session):
#         '''
#         Returns all items in the session

#         Parameters
#         --------
#         session: Id of a session

#         Returns
#         --------
#         out : set
#         '''
#         return self.session_item_map.get(session)

#     def vec_for_session(self, session):
#         '''
#         Returns all items in the session

#         Parameters
#         --------
#         session: Id of a session

#         Returns
#         --------
#         out : set
#         '''
#         return self.session_vec_map.get(session)

#     def sessions_for_item(self, item_id):
#         '''
#         Returns all session for an item

#         Parameters
#         --------
#         item: Id of the item session

#         Returns
#         --------
#         out : set
#         '''
#         return self.item_session_map.get(item_id) if item_id in self.item_session_map else set()

#     def most_recent_sessions(self, sessions, number):
#         '''
#         Find the most recent sessions in the given set

#         Parameters
#         --------
#         sessions: set of session ids

#         Returns
#         --------
#         out : set
#         '''
#         sample = set()

#         tuples = list()
#         for session in sessions:
#             time = self.session_time.get(session)  # ! ?这里不应该是
#             if time is None:
#                 print(' EMPTY TIMESTAMP!! ', session)
#             tuples.append((session, time))

#         tuples = sorted(tuples, key=itemgetter(1), reverse=True)
#         # print 'sorted list ', sortedList
#         cnt = 0
#         for element in tuples:
#             cnt = cnt + 1
#             if cnt > number:
#                 break
#             sample.add(element[0])
#         # print 'returning sample of size ', len(sample)
#         return sample

#     def possible_neighbor_sessions(self, input_item_id, session_id, user_id):
#         '''
#         Find a set of session to later on find neighbors in.
#         A self.sample_size of 0 uses all sessions in which any item of the current session appears.
#         self.sampling can be performed with the options "recent" or "random".
#         "recent" selects the self.sample_size most recent sessions while "random" just choses randomly.

#         Parameters
#         --------
#         sessions: set of session ids

#         Returns
#         --------
#         out : set
#         '''

#         # add relevant sessions for the current item
#         self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(
#             input_item_id)

#         if self.sample_size == 0:  # use all session as possible neighbors

#             # print('!!!!! runnig KNN without a sample size (check config)')
#             possible_neighbors = self.relevant_sessions

#         else:  # sample some sessions
#             if len(self.relevant_sessions) > self.sample_size:

#                 if self.sampling == 'recent':
#                     sample = self.most_recent_sessions(
#                         self.relevant_sessions, self.sample_size)
#                 elif self.sampling == 'random':
#                     sample = random.sample(
#                         self.relevant_sessions, self.sample_size)
#                 else:
#                     sample = self.relevant_sessions[:self.sample_size]

#                 possible_neighbors = sample
#             else:
#                 possible_neighbors = self.relevant_sessions

#         return possible_neighbors

#     def recent_neighbor_sessions(self, recent_items):
#         '''
#         根据recent items查找neighbor sessions
#         给定当前session的recent列表
#         对其中的每个item，查询recent列表中包含该item的session的id，依赖self.session_recent_map
#         '''
#         possible_neighbors = set()

#         for item in set(recent_items):
#             possible_neighbors = possible_neighbors | self.recent_session_map.get(
#                 item, set())

#         if self.sample_size == 0:  # use all session as possible neighbors

#             # print('!!!!! runnig KNN without a sample size (check config)')
#             result = possible_neighbors

#         else:  # sample some sessions
#             if len(possible_neighbors) > self.sample_size:

#                 if self.sampling == 'recent':
#                     sample = self.most_recent_sessions(
#                         possible_neighbors, self.sample_size)
#                 elif self.sampling == 'random':
#                     sample = random.sample(
#                         possible_neighbors, self.sample_size)
#                 else:
#                     sample = possible_neighbors[:self.sample_size]

#                 result = sample
#             else:
#                 result = possible_neighbors

#         return result

#     def calc_similarity(self, recent_items, sessions):
#         '''
#         Calculates the configured similarity for the items in recent_items and each session in sessions.

#         Parameters
#         --------
#         recent_items: set of item ids
#         sessions: list of session ids

#         Returns
#         --------
#         out : list of tuple (session_id,similarity)
#         '''

#         pos_map = {}  # 计算当前session的item权重，根据位置
#         length = len(recent_items)

#         count = 1
#         for item in recent_items:
#             if self.weighting is not None:
#                 pos_map[item] = getattr(self, self.weighting)(count, length)
#                 count += 1
#             else:
#                 pos_map[item] = 1

#         # if self.dwelling_time:
#         #     dt = dwelling_times.copy()
#         #     dt.append(0)
#         #     dt = pd.Series(dt, index=session_items)
#         #     dt = dt / dt.max()
#         #     # dt[session_items[-1]] = dt.mean() if len(session_items) > 1 else 1
#         #     dt[session_items[-1]] = 1

#         #     # print(dt)
#         #     for i in range(len(dt)):
#         #         pos_map[session_items[i]] *= dt.iloc[i]
#         # print(pos_map)

#         # if self.idf_weighting_session: # 未使用
#         #     max = -1
#         #     for item in session_items:
#         #         pos_map[item] = self.idf[item] if item in self.idf else 0
#         #                 if pos_map[item] > max:
#         #                     max = pos_map[item]
#         #             for item in session_items:
#         #                 pos_map[item] = pos_map[item] / max

#         # print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
#         items = set(recent_items)
#         neighbors = []
#         similarities = []
#         cnt = 0
#         for session in sessions:  # 对于每个可能的相似session
#             cnt = cnt + 1
#             # get items of the session, look up the cache first
#             # n_items = self.items_for_session(session) # 相似session的item set
#             n_items = set(self.session_recent_map[session])
#             # sts = self.session_time[session]

#             # dot product
#             # 计算session相似度，内积，但考虑每个共同item根据其在当前session中的位置
#             similarity = self.vec(items, n_items, pos_map)
#             if similarity > 0:

#                 # if self.weighting_time: # 未使用
#                 #     diff = timestamp - sts
#                 #     days = round(diff / 60 / 60 / 24)
#                 #     decay = pow(7 / 8, days)
#                 #     similarity *= decay

#                 # # print("days:",days," => ",decay)

#                 # if self.boost_own_sessions is not None:  # user_based
#                 #     similarity = self.apply_boost(session, user_id, similarity) # 如果相似session是用户自己的，similarity再增加

#                 neighbors.append(session)
#                 similarities.append(similarity)
#             else:
#                 neighbors.append(session)
#                 similarities.append(1e-10)

#         return neighbors, np.array(similarities)

#     # -----------------
#     # Find a set of neighbors, returns a list of tuples (sessionid: similarity)
#     # -----------------
#     def find_neighbors(self, input_item_id, session_id, user_id):
#         '''
#         Finds the k nearest neighbors for the given session_id and the current item input_item_id.

#         Parameters
#         --------
#         session_items: set of item ids
#         input_item_id: int
#         session_id: int

#         Returns
#         --------
#         out : list of tuple (session_id, similarity)
#         '''
#         possible_neighbors = self.possible_neighbor_sessions(
#             input_item_id, session_id, user_id)  # user_based
#         # possible_neighbors = self.calc_similarity(possible_neighbors, dwelling_times, timestamp, user_id)  # user_based

#         # possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
#         # possible_neighbors = possible_neighbors[:self.k]

#         return possible_neighbors

#     def score_items(self, neighbors, current_session, timestamp):
#         '''
#         Compute a set of scores for all items given a set of neighbors.

#         Parameters
#         --------
#         neighbors: set of session ids

#         Returns
#         --------
#         out : list of tuple (item, score)
#         '''
#         # now we have the set of relevant items to make predictions
#         scores = dict()
#         item_set = set(current_session)
#         # iterate over the sessions
#         for session in neighbors:
#             # get the items in this session
#             items = self.items_for_session(session[0])
#             step = 1

#             for item in reversed(current_session):
#                 if item in items:
#                     decay = getattr(
#                         self, self.weighting_score + '_score')(step)
#                     break
#                 step += 1

#             for item in items:

#                 # d=0* (exclude items form the current session)
#                 if not self.remind and item in item_set:
#                     # dont want to remind user the item that is already is his current (extended) session
#                     continue
#                 # item score计算：跟idf相关，以及decay（跟在当前session中的位置有关）
#                 old_score = scores.get(item)
#                 new_score = session[1]
#                 new_score = new_score if not self.idf_weighting else new_score + (
#                     new_score * self.idf[item] * self.idf_weighting)
#                 new_score = new_score * decay

#                 if not old_score is None:
#                     new_score = old_score + new_score  # score是累加的

#                 scores.update({item: new_score})

#         return scores

#     def linear_score(self, i):
#         return 1 - (0.1 * i) if i <= 100 else 0

#     def same_score(self, i):
#         return 1

#     def div_score(self, i):
#         return 1 / i

#     def log_score(self, i):
#         return 1 / (np.log10(i + 1.7))

#     def quadratic_score(self, i):
#         return 1 / (i * i)

#     def linear(self, i, length):
#         return 1 - (0.1 * (length - i)) if i <= 10 else 0

#     def same(self, i, length):
#         return 1

#     def div(self, i, length):
#         return i / length

#     def log(self, i, length):
#         return 1 / (np.log10((length - i) + 1.7))

#     def quadratic(self, i, length):
#         return (i / length) ** 2

#     def clear(self):
#         self.session = -1
#         self.session_items = []
#         self.relevant_sessions = set()

#         self.session_item_map = dict()
#         self.item_session_map = dict()
#         self.session_time = dict()
#         self.session_user_map = dict()  # user_based

#     def support_users(self):
#         '''
#             whether it is a session-based or session-aware algorithm
#             (if returns True, method "predict_with_training_data" must be defined as well)

#             Parameters
#             --------

#             Returns
#             --------
#             True : if it is session-aware
#             False : if it is session-based
#         '''
#         return True

#     def predict_with_training_data(self):
#         '''
#             (this method must be defined if "support_users is True")
#             whether it also needs to make prediction for training data or not (should we concatenate training and test data for making predictions)

#             Parameters
#             --------

#             Returns
#             --------
#             True : e.g. hgru4rec
#             False : e.g. uvsknn
#             '''
#         return False

#     def extend_session_in_fit(self, row, index_user, index_item):
#         '''
#         构造self.last_user_items：映射user id到item list
#         其长度小于等于self.extend_session_length，若超过，取最后self.extend_session_length个

#         '''
#         if not row[index_user] in self.last_user_items:
#             # create a new list to save the user's last viewed items
#             self.last_user_items[row[index_user]] = []
#         self.last_user_items[row[index_user]].append(row[index_item])
#         if len(self.last_user_items[row[index_user]]) > self.extend_session_length:
#             self.last_user_items[row[index_user]] = self.last_user_items[row[index_user]][
#                 -self.extend_session_length:]

#     def extend_session_in_predict_next(self, items, input_user_id):
#         '''
#         为当前session补充item，使用用户最近的click
#         同时把包含补充的item对应的session加入到self.relevant_sessions
#         '''
#         if len(items) < self.extend_session_length:
#             # update the session with items from the users past
#             n = len(self.session_items)
#             addItems = self.extend_session_length - n
#             prev_items = self.last_user_items[input_user_id][-addItems:]
#             items = prev_items + self.session_items

#             # if it is beginning of the session => find relevant sessions for added items
#             if len(self.items_previous) == 0:
#                 for item in set(prev_items):
#                     self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(
#                         item)
#             # not beginning of the session, so we already retrieved neighbours for the extended session
#             elif self.refine_mode:
#                 # if the first item that was in the previous step, is not in the current step anymore => refine the self.relevant_sessions
#                 if not self.items_previous[0] in items:
#                     self.relevant_sessions = set()
#                     for item in set(items):
#                         self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(
#                             item)

#             # update self.items_previous
#             self.items_previous = items
#         # the session is long enough => refine the self.relevant_sessions to just consider current session's items
#         elif self.refine_mode and self.need_refine:
#             self.relevant_sessions = set()
#             for item in set(self.session_items):
#                 # then we can continue with just adding related sessions for the current item
#                 self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(
#                     item)
#             # refined once after reach to the defined length, no need to do refine anymore
#             self.need_refine = False

#         return items

#     def apply_boost(self, session, user_id, similarity):
#         if self.boost_own_sessions > 0.0 and self.session_user_map[session] == user_id:
#             similarity = similarity + (similarity * self.boost_own_sessions)
#         return similarity

#     def retrieve_past_neighbors(self, user_id):
#         for neighbor_sid in self.relevant_sessions:
#             if self.session_user_map[neighbor_sid] == user_id:
#                 for item in self.items_for_session(neighbor_sid):
#                     self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(
