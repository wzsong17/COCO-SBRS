# %%
from collections import defaultdict
from fileinput import lineno
# import datetime
# import warnings
# from tqdm.notebook import tqdm
from pathlib import Path as path
import pickle
import numpy as np
# import pandas as pd

# warnings.filterwarnings('ignore')

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# %% [markdown]
# ## load data

# %%
# dfr = {}

# %%
# for delicious douban epinions
DATA_DIR = path(__file__).resolve().parents[1]
with open(DATA_DIR/'session_data.pkl', 'rb') as f:
    dt = pickle.load(f)

datas = ['delicious']
# datas = ['ciao', 'lastfm', 'gowalla',
#          'epinions', 'douban', 'reddit']
# dfr['delicious'] = dt['delicious']

for data in datas:
    print(f'data {data}')
    # session开始时间
    dt[data]['start_time'] = dt[data].groupby(
        'sessionId').timestamp.transform('min')

# %%
# for tafeng
# DATA_DIR = path(r'E:\appdata\jianguoyun\2004shortsession\data')
# with open('E:\\appdata\\jianguoyun\\2004shortsession\\data\\session_tafeng.pkl', 'rb') as f:
#     df = pickle.load(f)
# datas = ['tafeng']
# dfr = {}
# dfr['tafeng'] = df

# %%
# for reddit
# DATA_DIR = path(r'E:\appdata\jianguoyun\2004shortsession\data')
# datas = ['reddit']
# data = 'reddit'
# with open(f'E:\\appdata\\jianguoyun\\2004shortsession\\data\\session_{data}.pkl', 'rb') as f:
#     df = pickle.load(f)
#     dfr[data] = df

# %% [markdown]
# ## preprocessing

# %%
# 去掉小于3个session的用户
# min_session_per_user = 3
# for data in datas:
#     print(data)
#     t = dfr[data].groupby('user').sessionId.nunique() >= 3
#     users = set(t[t].index)
#     print(f'before {len(dfr[data])}')
#     dfr[data] = dfr[data][dfr[data].user.isin(users)]
#     print(f'after {len(dfr[data])}')

# %%
# 日志按时间排序
# for data in datas:
#     dfr[data] = dfr[data].sort_values(by=['timestamp'])

# %% [markdown]
# ## split

# %% [markdown]
# ### spliting

# %%

# train_set/test_set: key是dataset字符串，value是dict
# value的key是数字0-4（代表split），value是dataframe
train_set = defaultdict(dict)
test_set = defaultdict(dict)

# splittype = 'pct'  # 若是last，每个用户最后一个session做测试集，若是pct，最后20%的session是测试集
# 切割训练集和测试集
# 按用户切割：
# 1. 若用户session数量

num_split = 5  # 每个数据集划分多少次
test_pct = 0.2  # 测试session数量占比

for data in datas:
    print(f'data {data}')
    df = dt[data]
    train_set[data] = {}
    test_set[data] = {}
    map_user_session = df.groupby('user')['sessionId'].unique()
    map_item_session = df.groupby('item')['sessionId'].unique()

    set_session = set(df['sessionId'].unique())
    num_session = len(set_session)
    num_test = int(test_pct * num_session)
    num_train = num_session - num_test

    # 每个数据集划分num_split次
    for i in range(num_split):
        print(f'split {i+1}')
        train_session = set()
        test_session = set()
        # 每个用户/item都随机选择一个session，放到训练集中
        select_session_user = map_user_session.apply(np.random.choice)
        select_session_item = map_item_session.apply(np.random.choice)
        # 将随机选择到的放到训练集
        train_session.update(select_session_user)
        train_session.update(select_session_item)
        # 训练集，在剩下的session中选择一部分再放到训练集中，满足训练集session个数
        set_to_choice = set_session - train_session
        session_select = np.random.choice(
            list(set_to_choice), (num_train - len(train_session)), replace=False)
        train_session.update(session_select)
        # 测试集，所有未选中的
        test_session.update(set_session-train_session)

        train_set[data][i] = df[df.sessionId.isin(
            train_session)][['user', 'item', 'timestamp', 'sessionId', 'start_time']]
        test_set[data][i] = df[df.sessionId.isin(
            test_session)][['user', 'item', 'timestamp', 'sessionId', 'start_time']]

        print(
            f'#sessions in train/test: {train_set[data][i].sessionId.nunique()}/{test_set[data][i].sessionId.nunique()}')
        print(
            f'#users in train/test: {train_set[data][i].user.nunique()}/{test_set[data][i].user.nunique()}')
        print(
            f'#items in train/test: {train_set[data][i].item.nunique()}/{test_set[data][i].item.nunique()}')
        print(
            f'#interactions in train/test: {len(train_set[data][i])}/{len(test_set[data][i])}')

    # gu = dfr[data].groupby('user')
    # train_sessions = set()
    # test_sessions = set()
    # for uid, df in gu:
    #     # if splittype == 'pct':
    #     n_sessions = df.sessionId.nunique()
    #     # if n_sessions < 2:
    #     #     continue
    #     split_point = int(0.2*n_sessions)
    #     if split_point < 1:  # 若用户的session的20%少于1一个，他不作为测试集/验证集
    #         print('User f{uid} with {n_sessions} sessions, skip')
    #         continue
    #     # elif splittype == 'last':
    #     #     split_point = -1
    #     # else:
    #     #     print('error no such splitype')
    #     session_ids = (df[['sessionId', 'start_time']].drop_duplicates(
    #     ).sort_values(by=['start_time']))['sessionId']

    #     train_set[data][uid] = list(session_ids[:split_point])
    #     test_set[data][uid] = list(session_ids[split_point:])
    #     train_sessions.update(list(session_ids[:split_point]))
    #     test_sessions.update(list(session_ids[split_point:]))
    # train_dfs[data] = dfr[data][dfr[data].sessionId.isin(train_sessions)]
    # test_dfs[data] = dfr[data][dfr[data].sessionId.isin(test_sessions)]

# %%
# 最后多少天的session作为测试集
# n_days_for_test = {
#     'gowalla': 30, # 共626天 20month
#     'delicious': 30 * 12,#2546天,84month
#     'epinions': 30 * 24,# 4326天,144month
#     'douban': 30 * 24,#4297天,143month
# }

# train_dfs = {}
# test_dfs = {}

# for data in datas:
#     # 每个session的start time
#     dfr[data]['start_time'] = dfr[data].groupby('sessionId').timestamp.transform('min')
#     # 分割时间点
#     split_point = dfr[data].timestamp.max() - n_days_for_test[data]*24*60*60
#     # 训练集
#     train_dfs[data] = dfr[data][dfr[data].start_time<split_point]
#     # 测试
#     test_dfs[data] = dfr[data][dfr[data].start_time>=split_point]

# %%
# 测试集中去掉没有出现在训练集中的item，再删除所有长度为1的session
# min_sess_length = 2
# for data in datas:
#     items = set(train_dfs[data].item.unique())
#     test_dfs[data] = test_dfs[data][test_dfs[data].item.isin(items)]
#     test_dfs[data] = test_dfs[data][test_dfs[data].groupby(
#         'sessionId').sessionId.transform('size') >= min_sess_length]
    # 从训练集中删除测试集中没有的user: IIRNN要求
    #users_in_testset = set(test_dfs[data].user.unique())
    #train_dfs[data] = train_dfs[data][train_dfs[data].user.isin(users_in_testset)]

# %%
# remap user/item id
user2id = {}
item2id = {}
sess2id = {}
for data in datas:
    print(f'data: {data}')
    # 获取映射表，每个split保持一致
    user2id[data] = dict(
        zip(dt[data].user.unique(), range(dt[data].user.nunique())))
    item2id[data] = dict(
        zip(dt[data].item.unique(), range(dt[data].item.nunique())))
    sess2id[data] = dict(zip(dt[data].sessionId.unique(), range(
        1, dt[data].sessionId.nunique() + 1)))

    # map_usr = {uid: idx for idx, uid in enumerate(
    #     train_dfs[data].user.unique())}
    # map_itm = {vid: idx for idx, vid in enumerate(
    #     train_dfs[data].item.unique())}
    for i in range(num_split):

        train_set[data][i].user = train_set[data][i].user.map(user2id[data])
        train_set[data][i].item = train_set[data][i].item.map(item2id[data])
        train_set[data][i].sessionId = train_set[data][i].sessionId.map(
            sess2id[data])

        test_set[data][i].user = test_set[data][i].user.map(user2id[data])
        test_set[data][i].item = test_set[data][i].item.map(item2id[data])
        test_set[data][i].sessionId = test_set[data][i].sessionId.map(
            sess2id[data])

    # print(
    #     f'#users in train/test {train_dfs[data].user.nunique()}/{test_dfs[data].user.nunique()}')
    # print(
    #     f'#items in train/test {train_dfs[data].item.nunique()}/{test_dfs[data].item.nunique()}')
    # print(
    #     f'#sessions in train/test set: {train_dfs[data].sessionId.nunique()}/{test_dfs[data].sessionId.nunique()}')
    # print(
    #     f'#interactions in train/test set: {len(train_dfs[data])}/{len(test_dfs[data])}')

# %% [markdown]
# ## save to files

# %%
# datas = ['delicious2']


#! format 0 my model
for data in datas:
    for i in range(num_split):
        with open(DATA_DIR / f'fm_{data}_{i}.pkl', 'wb') as f:
            pickle.dump((train_set[data][i], test_set[data][i]), f)


# %% [markdown]
#! ### format 1 (stamp/shan)
# print('>>fmt1')
# # %%
# # test_length = [2,3,4,5]
# max_sess_len_test = 5
#
# for data in datas:
#     print(data)
#     for i in range(num_split):
#
#         assert not train_set[data][i].user.isna().any()
#         assert not train_set[data][i].item.isna().any()
#         assert not test_set[data][i].user.isna().any()
#         assert not test_set[data][i].item.isna().any()
#
#         train_file = DATA_DIR / f'fm1_{data}_train_{i}.csv'
#         test_file = DATA_DIR / f'fm1_{data}_test_{i}.csv'
#     #     file_hd = {k:open(DATA_DIR / f'fm1_{data}_test_{k}.csv', 'w') for k in test_length}
#
#         with open(train_file, 'w') as trn_f:
#             # first line
#             n_user = train_set[data][i].user.nunique()
#             n_item = train_set[data][i].item.nunique()
#             trn_f.write(f'{n_user} {n_item}\n')
#
#             gu = train_set[data][i].groupby('user')
#             for uid, dfu in gu:
#                 gs = dfu.sort_values(by=['timestamp']).groupby('sessionId')
#                 sstr = []
#                 for sid, dfs in gs:
#                     d_ = dfs.sort_values(by=['timestamp'])
#                     sess_len = len(d_)
#                     if len(d_) < 2:
#                         continue
#     #                 for i in range(1,  len(d_)):
#     #                     sstr.append(':'.join(list(map(str,d_.iloc[:i].item))))
#                     sstr.append(':'.join(list(map(str, d_.item))))
#                 trn_f.write(f'{uid} {"@".join(sstr)}\n')
#
#         with open(test_file, 'w') as tst_f:
#             gu = test_set[data][i].groupby('user')
#             for uid, dfu in gu:
#                 gs = dfu.groupby('sessionId')
#                 for sid, dfs in gs:
#                     d_ = dfs.sort_values(by=['timestamp'])
#                     if len(d_) < 2:
#                         continue
#                     for i in range(2,  min(len(d_), max_sess_len_test)):
#                         tst_f.write(
#                             f'{uid} {":".join(list(map(str,d_.iloc[:i].item)))}\n')
# #                 for size in test_length:
# #                     if len(dfs) < size:
# #                         continue
# #                     else:
# #                         file_hd[size].write(f'{int(uid)} {":".join(list(map(str,map(int,dfs.item[:size]))))}\n')
#
# # for k,v in file_hd.items():
# #     v.close()
#
# # %% [markdown]
# #! ### format 2(insert/iirnn)
# print('>>fmt2')
# # %%
# max_session_length = 20
# for data in datas:
#     print(f'processing {data}')
#     for i in range(num_split):
#         result = defaultdict(dict)
#         PAD_ITEM_ID = int(train_set[data][i].item.max()+1)
#
#         for st in ['train', 'test']:
#             print(st)
#             df = eval(f'{st}_set[data][i]')
#             gu = df.groupby('sessionId')
#             result[f'{st}set'] = defaultdict(list)
#             result[f'{st}_session_lengths'] = defaultdict(list)
#             starttime = defaultdict(list)
#
#             for sid, dfu in gu:
#
#                 dfu_ = dfu.sort_values(by='timestamp')
#                 assert len(dfu_) > 1
#
#                 uid = int(dfu_.iloc[0]['user'])
#                 session = dfu[['timestamp', 'item']].values.astype(int)
#
#                 result[f'{st}_session_lengths'][uid].append(len(dfu)-1)
#                 starttime[uid].append(dfu_.timestamp.min())
#                 if len(dfu) < max_session_length:
#                     session = np.append(session, np.array(
#                         [[0, PAD_ITEM_ID]]*(max_session_length-len(session)), dtype=int), 0)
#                 result[f'{st}set'][uid].append(session)
#             print(len(starttime.keys()))
#             for uid in starttime.keys():  # 按sesson开始时间对session排序
#                 idx_sorted = np.argsort(starttime[uid])
#                 result[f'{st}_session_lengths'][uid] = [
#                     result[f'{st}_session_lengths'][uid][i] for i in idx_sorted]
#                 result[f'{st}set'][uid] = [result[f'{st}set'][uid][i]
#                                            for i in idx_sorted]
#     #             result.pop(f'{st}_session_starttime')
#
#         with open(DATA_DIR / f'fm2_{data}_{i}.pkl', 'wb') as f:
#             pickle.dump(result, f)
#
# # %%
# # 老脚本，可能出现长度为1的session
# # max_session_length = 20
# # for data in ['reddit4']:#datas:
# #     print(f'processing {data}')
# #     result = defaultdict(dict)
# #     PAD_ITEM_ID = int(train_dfs[data].item.max()+1)
#
# #     for st in ['train', 'test']:
# #         print(st)
# #         df = eval(f'{st}_dfs[data]')
# #         gu = df.groupby('user')
# #         for uid, dfu in tqdm(gu):
# #             dfu_ = dfu.sort_values(by='timestamp')
# #             last_sessionId = dfu_.iloc[0]['sessionId']
#
# #             user_sessions = []
# #             user_sessions_lengths = []
# #             session = []
# #             for row in dfu_.itertuples():
# #                 if row.sessionId == last_sessionId: # 同一个session的item
# #                     session.append([row.timestamp, row.item])
# #                 else:
# #                     assert len(session)>1
# #                     user_sessions_lengths.append(len(session)-1)
# #                     # padding last session
# #                     session.extend([[0, PAD_ITEM_ID]]*(max_session_length-len(session)))
# #                     user_sessions.append(session)
#
# #                     # new session
# #                     last_sessionId = row.sessionId
# #                     session = [[ row.timestamp, row.item]]
# #             # last session
# #             user_sessions_lengths.append(len(session)-1)
# #             session.extend([[0, PAD_ITEM_ID]]*(max_session_length-len(session)))
# #             user_sessions.append(session)
# #             result[f'{st}set'][uid] = user_sessions
# #             result[f'{st}_session_lengths'][uid] = user_sessions_lengths
#
# #     with open(DATA_DIR / f'fm2_{data}.pkl', 'wb') as f:
# #         pickle.dump(result, f)
#
# # %% [markdown]
# #! ### format 3 (CSRM/srgnn)
# print('>>fmt3')
# # %%
# max_sess_len = 20
# max_sess_len_test = 5
#
# for data in datas:
#     print(f'processing {data}...')
#     for i in range(num_split):
#         sessions = {
#             'train': [],
#             'test': []
#         }
#         sessions_lab = {
#             'train': [],
#             'test': []
#         }
#         for st in ['train', 'test']:
#             print(st)
#             df = eval(f'{st}_set[data][i]')
#             gs = df.groupby('sessionId')
#             for sid, dfs in gs:
#                 df = dfs.sort_values(by='timestamp')
#                 last_sessionId = -1
#                 cur_sess = []
#
#                 for row in df.itertuples():
#                     if row.sessionId == last_sessionId:
#                         if st == 'train' or (st == 'test' and len(cur_sess) <= max_sess_len_test-1):
#                             sessions[st].append(cur_sess.copy())
#                             # item id start from 1
#                             sessions_lab[st].append(row.item+1)
#                         cur_sess.append(row.item+1)
#                     else:
#                         cur_sess = [row.item+1]
#                     last_sessionId = row.sessionId
#         peridx = np.random.permutation(len(sessions['test']))
#         validx = set(peridx[:int(len(sessions['test'])/2)])
#         val = sessions['test']
#         val_lab = sessions_lab['test']
#         tst = sessions['test']
#         tst_lab = sessions_lab['test']
#     #     idx = 0
#     #     for sess, lab in zip(sessions['test'],sessions_lab['test']):
#     #         if idx in validx:
#     #             val.append(sess)
#     #             val_lab.append(lab)
#     #         else:
#     #             tst.append(sess)
#     #             tst_lab.append(lab)
#     #         idx += 1
#     #     tstidx = peridx[int(len(sessions['test'])/2):]
#     #     with open(DATA_DIR / f'fm3_{data}_train.pkl', 'wb') as f:
#     #         pickle.dump((sessions['train'],sessions_lab['train']), f)
#     #     with open(DATA_DIR / f'fm3_{data}_valid.pkl', 'wb') as f:
#     #         pickle.dump((val,val_lab), f)
#     #     with open(DATA_DIR / f'fm3_{data}_test.pkl', 'wb') as f:
#     #         pickle.dump((tst,tst_lab), f)
#         with open(DATA_DIR / f'fm3_csrm_{data}_{i}.pkl', 'wb') as f:
#             pickle.dump({'trn': (sessions['train'], sessions_lab['train']), 'val': (
#                 val, val_lab), 'tst': (tst, tst_lab)}, f)
#
# # %% [markdown]
# #! ### format 4 (format3 基础上加上user id和session id) index从0开始
#
# # %%
# # max_sess_len_train = 20
# # max_sess_len_test = 5
#
# # for data in datas:
# #     print(f'processing {data}...')
# #     sessions = {
# #         'train': {
# #             'sess_list' : [],
# #             'sess_label' : [],
# #             'sess_time' : [],
# #             'sess_id': [],
# #             'user_id' : []
# #         },
# #         'test' : {
# #             'sess_list' : [],
# #             'sess_label' : [],
# #             'sess_time' : [],
# #             'sess_id': [],
# #             'user_id' : []
# #         }
# #     }
#
# #     for st in ['train', 'test']:
# #         print(st)
# #         df = eval(f'{st}_dfs[data]')
# #         gs = df.groupby('sessionId')
#
# #         for sid, dfs in tqdm(gs):
# #             df = dfs.sort_values(by='timestamp')
# #             last_sessionId = -1
# #             cur_sess = []
#
# #             for row in df.itertuples():
# #                 if row.sessionId == last_sessionId:
# #                     if len(cur_sess) <= eval(f'max_sess_len_{st}')-1:
# #                         sessions[st]['sess_list'].append(cur_sess.copy())
# #                         sessions[st]['sess_label'].append(row.item)
# #                         sessions[st]['sess_id'].append(row.sessionId)
# #                         sessions[st]['sess_time'].append(row.start_time)
# #                         sessions[st]['user_id'].append(row.user)
# #                     cur_sess.append(row.item)
# #                 else:
# #                     cur_sess = [row.item]
# #                 last_sessionId = row.sessionId
# #     peridx = np.random.permutation(len(sessions['test']['sess_id']))
# #     validx = set(peridx[:int(len(sessions['test']['sess_id'])/2)])
# #     val = defaultdict(list)
# #     tst = defaultdict(list)
# #     idx = 0
# #     for sess, lab, sid, ss_time, user_id in zip(sessions['test']['sess_list'],sessions['test']['sess_label'],sessions['test']['sess_id'],\
# #                                          sessions['test']['sess_time'],sessions['test']['user_id']):
# #         if idx in validx:
# #             val['sess_list'].append(sess)
# #             val['sess_label'].append(lab)
# #             val['sess_id'].append(sid)
# #             val['sess_time'].append(ss_time)
# #             val['user_id'].append(user_id)
# #         else:
# #             tst['sess_list'].append(sess)
# #             tst['sess_label'].append(lab)
# #             tst['sess_id'].append(sid)
# #             tst['sess_time'].append(ss_time)
# #             tst['user_id'].append(user_id)
# #         idx += 1
# #     with open(DATA_DIR / f'fm4_{data}_train.pkl', 'wb') as f:
# #         pickle.dump(sessions['train'], f)
# #     with open(DATA_DIR / f'fm4_{data}_valid.pkl', 'wb') as f:
# #         pickle.dump(val, f)
# #     with open(DATA_DIR / f'fm4_{data}_test.pkl', 'wb') as f:
# #         pickle.dump(tst, f)
#
# # %% [markdown]
# #! ### format 5 (stan/sknn)
# print('>>fmt5')
# # %%
# max_sess_len_train = 20
# max_sess_len_test = 5
#
# for data in datas:
#     print(f'processing {data}...')
#     for i in range(num_split):
#         sessions = {
#             'train': defaultdict(list),
#             'test': defaultdict(list)
#         }
#         for st in ['train', 'test']:
#             print(st)
#             df = eval(f'{st}_set[data][i]')
#             gs = df.groupby('sessionId')
#
#             for sid, dfs in gs:
#                 df = dfs.sort_values(by='timestamp')
#                 last_sessionId = -1
#                 cur_sess = []
#
#                 for row in df.itertuples():
#                     if row.sessionId == last_sessionId:
#                         if len(cur_sess) <= eval(f'max_sess_len_{st}')-1:
#                             sessions[st]['sess_list'].append(cur_sess.copy())
#                             sessions[st]['sess_label'].append(
#                                 row.item+1)  # id from 1
#                             sessions[st]['sess_id'].append(row.sessionId)
#                             sessions[st]['sess_time'].append(row.start_time)
#                         cur_sess.append(row.item+1)  # id from 1
#                     else:
#                         cur_sess = [row.item+1]  # id from 1
#                     last_sessionId = row.sessionId
#         with open(DATA_DIR / f'fm4_stan_{data}_{i}.pkl', 'wb') as f:
#             pickle.dump(sessions, f)
#
# # %% [markdown]
# #! ### format 6 (hgru2rec/stamp官方)
#
# # %%
# print('>>fmt6')
# for data in datas:
#     print(data)
#     for i in range(num_split):
#         # remap session Id to int
#         # ssid_map = {}
#         # for sid in train_set[data][i].sessionId.unique():
#         #     ssid_map[sid] = len(ssid_map)
#         # for sid in test_set[data][i].sessionId.unique():
#         #     ssid_map[sid] = len(ssid_map)
#         # train_set[data][i]['sessionId'] = train_set[data][i]['sessionId'].map(
#         #     ssid_map)
#         # test_set[data][i]['sessionId'] = test_set[data][i]['sessionId'].map(
#         #     ssid_map)
#
#         # add position in groups
#         train_set[data][i]['position'] = train_set[data][i].groupby(
#             'sessionId').timestamp.apply(np.argsort)
#         test_set[data][i]['position'] = test_set[data][i].groupby(
#             'sessionId').timestamp.apply(np.argsort)
#
#         # filtering session items larger than position 4
#         test_set[data][i] = test_set[data][i][test_set[data][i].position < 5]
#
#         with open(DATA_DIR / f'fm6_hgru_{data}_{i}.pkl', 'wb') as f:
#             pickle.dump({'train': train_set[data][i][['user', 'item', 'timestamp', 'sessionId', 'position']],
#                         'test': test_set[data][i][['user', 'item', 'timestamp', 'sessionId', 'position']]}, f)
#
# # %%
