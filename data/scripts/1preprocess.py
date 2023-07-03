# %%
# from matplotlib.ticker import PercentFormatter
import datetime
# import warnings
from tqdm.notebook import tqdm
from pathlib import Path as path
import pickle
import pandas as pd
import numpy as np
import sys
# %pylab
# %matplotlib inline

# warnings.filterwarnings('ignore')

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# %%
# read data
dfr = {}
# dfn = {}
# ,'ciao','lastfm''gowalla','epinions','douban', 'reddit'
# datas = ['delicious', 'ciao', 'lastfm', 'gowalla',
#          'epinions', 'douban', 'reddit']
datas = ['delicious']
DATA_DIR = path(__file__).resolve().parents[1]
print(DATA_DIR)
RAW_DIR = DATA_DIR / 'raw_data'

print('=>read data')
print('=>read data', file=sys.stderr)
for data in datas:
    if data == 'gowalla':
        dfr[data] = pd.read_csv(RAW_DIR/data/'Gowalla_totalCheckins.txt', sep='\t', header=None, usecols=[
                                0, 1, 4], names=['user', 'checkintime', 'latitude', 'longitude', 'location'], parse_dates=[1])
        # dfn[data] = pd.read_csv(
        #     RAW_DIR/data/'Gowalla_edges.txt', sep='\t', header=None, names=['user', 'friend'])

        dfr[data]['timestamp'] = dfr[data].checkintime.values.astype(
            np.int64)//10**9
        dfr[data] = dfr[data].rename(columns={'location': 'item'})
        dfr[data] = dfr[data].drop(columns=['checkintime'])

        # t = dfn[data].values[:, [1, 0]]
        # dfn[data] = pd.DataFrame(t, columns=['user', 'friend']).append(
        #     dfn[data], ignore_index=True)

    if data == 'lastfm':
        dfr[data] = pd.read_csv(
            RAW_DIR/data/'user_taggedartists-timestamps.dat', sep='\t')
        # dfn[data] = pd.read_csv(RAW_DIR/data/'user_friends.dat', sep='\t')

        dfr[data] = dfr[data].rename(
            columns={'userID': 'user', 'tagID': 'item'})
        dfr[data] = dfr[data].drop(columns=['artistID'])
        dfr[data]['timestamp'] = dfr[data]['timestamp']//1e3
        dfr[data] = dfr[data][dfr[data].timestamp > 1.1e9]

        # dfn[data] = dfn[data].rename(
        #     columns={'userID': 'user', 'friendID': 'friend'})
        # t = dfn[data].values[:, [1, 0]]
        # dfn[data] = pd.DataFrame(t, columns=['user', 'friend']).append(
        #     dfn[data], ignore_index=True)

    if data == 'delicious':
        dfr[data] = pd.read_csv(
            RAW_DIR/data/'user_taggedbookmarks-timestamps.dat', sep='\t')
        # dfn[data] = pd.read_csv(
        #     RAW_DIR/data/'user_contacts-timestamps.dat', sep='\t')

        dfr[data] = dfr[data].rename(
            columns={'userID': 'user', 'tagID': 'item'})
#         dfr[data] = dfr[data].drop(columns=['bookmarkID'])
        dfr[data]['timestamp'] = dfr[data]['timestamp']//1e3

        # dfn[data] = dfn[data].rename(
        #     columns={'userID': 'user', 'contactID': 'friend'})
        # dfn[data] = dfn[data].drop(columns=['timestamp'])
        # t = dfn[data].values[:, [1, 0]]
        # dfn[data] = pd.DataFrame(t, columns=['user', 'friend']).append(
        #     dfn[data], ignore_index=True)

    if data == 'douban':
        dfr[data] = pd.read_csv(RAW_DIR/data / 'douban_movie.tsv', sep='\t',
                                dtype={0: np.int32, 1: np.int32, 2: np.int32, 3: np.float32})
        # dfn[data] = pd.read_csv(
        #     RAW_DIR/data / 'socialnet.tsv', sep='\t', dtype={0: np.int32, 1: np.int32})

        dfr[data] = dfr[data][dfr[data].Rating >= 4]
        dfr[data] = dfr[data].rename(
            columns={'UserId': 'user', 'ItemId': 'item', 'Timestamp': 'timestamp'})
        dfr[data] = dfr[data].drop(columns=['Rating'])

        # dfn[data] = dfn[data].rename(
        #     columns={'Follower': 'user', 'Followee': 'friend'})
        # dfn[data] = dfn[data].drop(columns=['Weight'])

    if data == 'epinions':
        dfr[data] = pd.read_csv(RAW_DIR/data/'rating_with_timestamp.txt', sep='  ', header=None, usecols=[0, 1, 3, 5], names=[
                                'user', 'item', 'domain', 'rating', 'helpfulness', 'timestamp'], dtype={'user': int, 'item': int, 'timestamp': int})
        # dfn[data] = pd.read_csv(RAW_DIR/data/'trust.txt', sep='  ', header=None,
        # names = ['user', 'friend'], dtype = {'user': int, 'friend': int})

        dfr[data] = dfr[data][dfr[data].rating >= 4]
        dfr[data] = dfr[data].drop(columns=['rating'])

    if data == 'ciao':
        dfr[data] = pd.read_csv(RAW_DIR/data/'rating_with_timestamp.txt', sep='  ', header=None, usecols=[0, 1, 3, 5], names=[
            'user', 'item', 'domain', 'rating', 'helpfulness', 'timestamp'], dtype={'user': int, 'item': int, 'timestamp': int})
        # dfn[data] = pd.read_csv(RAW_DIR/data/'trust.txt', sep=' ', header=None,
        # names=['user', 'friend'], dtype={'user': int, 'friend': int})

        dfr[data] = dfr[data][dfr[data].rating >= 4]
        dfr[data] = dfr[data].drop(columns=['rating'])

    if data == 'reddit':
        dfr[data] = pd.read_csv(RAW_DIR/data/'reddit_data.csv')
        dfr[data] = dfr[data].rename(
            columns={'utc': 'timestamp', 'username': 'user', 'subreddit': 'item'})

# %%
# 去重
print('=>drop duplicate lines')
print('=>drop duplicate lines', file=sys.stderr)
for data in datas:
    print(f'{data}')
    dfr[data].drop_duplicates(inplace=True)
#     dfn[data].drop_duplicates(inplace=True)

# %%
# 按时间范围过滤评分
date_setting = {}
# 查看动作按时间分布
for data in datas:
    date_data = {}
    date_setting[data] = date_data
    print('-'*20)
    print(f'data: {data}')

    df = dfr[data]
    total_seconds = (df['timestamp'].max()-df['timestamp'].min())
    n_days = total_seconds / (24*3600)
    n_weeks = total_seconds / (7*24*3600)
    n_months = total_seconds / (30*24*3600)
    date_data['min_dt'] = min_dt = datetime.datetime.fromtimestamp(
        df['timestamp'].min())
    date_data['max_dt'] = max_dt = datetime.datetime.fromtimestamp(
        df['timestamp'].max())

    print(f'Min date: {min_dt.year}-{min_dt.month}-{min_dt.day}')
    print(f'Max date: {max_dt.year}-{max_dt.month}-{max_dt.day}')

    print(f'#days {n_days}')
    print(f'#weeks {n_weeks}')
    print(f'#months {n_months}')

    # 过滤
    begin = datetime.datetime(
        min_dt.year, min_dt.month, min_dt.day+1).timestamp()
    end = datetime.datetime(max_dt.year, max_dt.month, max_dt.day).timestamp()
    dfr[data] = dfr[data][dfr[data]
                          ['timestamp'].between(left=begin, right=end)]

# fig, axs = plt.subplots(2, 2, figsize=(20, 20))
# for i, data in enumerate(datas):
#     dfr[data]['timestamp'].plot(
#         kind='hist', ax=axs[i//2][i % 2], bins=100, title=f'data {data}')

# %%
# 去除评分过少的user和item
print('=>remove users and items occured less 10 times')
print('=>remove users and items occured less 10 times', file=sys.stderr)

min_hist_for_usr = 10
min_hist_for_itm = 10

for data in datas:
    print('-'*20)
    print(data)
    print(f'Before: {len(dfr[data])}')

    dfr[data] = dfr[data][dfr[data].groupby(
        'user').timestamp.transform('size') >= min_hist_for_usr]
    dfr[data] = dfr[data][dfr[data].groupby(
        'item').timestamp.transform('size') >= min_hist_for_itm]

    print(f'After:  {len(dfr[data])}')

# %%
# 只保留rating log和network中重叠的用户
# for data in ['delicious']:
#     print('-'*20)
#     print(data)

#     usrs_in_net = set(dfn[data].user.unique())
#     usrs_in_net.update(set(dfn[data].friend.unique()))
#     usrs_in_log = set(dfr[data].user.unique())

#     print('filtering log')
#     print(f'Before: {len(dfr[data])}')
#     dfr[data] = dfr[data][dfr[data].user.isin(usrs_in_net)]
#     print(f'After:  {len(dfr[data])}')

#     print('filtering social links')
#     print(f'Before: {len(dfn[data])}')
#     dfn[data] = dfn[data][dfn[data].user.isin(usrs_in_log)|dfn[data].friend.isin(usrs_in_log)]
#     print(f'After:  {len(dfn[data])}')

# %% [markdown]
# ## session处理

# %% [markdown]
# ### 时间段：一段时间内作为session

# %%
# 获得session
# time_thresh = {
#     'gowalla': 3600*24,
#     'lastfm': 3600*24,
#     'douban': 3600*24,
#     'ciao': 3600*24,
#     'epinions': 3600*24,
#     'delicious': 3600*24,
# }

# for data in datas:
#     print('-'*20)
#     print(data)
#     min_timestamp = dfr[data]['timestamp'].min()
#     time_id = [int(math.floor((t-min_timestamp) / time_thresh[data])) for t in dfr[data]['timestamp']]
#     dfr[data]['timeId'] = time_id
#     if data == 'delicious':
#         session_id = [str(uid)+'_'+str(bid)+'_'+str(tid) for uid, bid, tid in zip(dfr[data]['user'], dfr[data]['bookmarkID'], dfr[data]['timeId'])]
#     else:
#         session_id = [str(uid)+'_'+str(tid) for uid, tid in zip(dfr[data]['user'], dfr[data]['timeId'])]
#     dfr[data]['sessionId'] = session_id

# %% [markdown]
# ### 第二种方式，inactive time

# %%
# 按照inactive time切割session的方式
print('=>generate session id')
print('=>generate session id', file=sys.stderr)
time_thresh = {
    'gowalla': 3600*6,
    'lastfm': 3600*6,
    'douban': 3600*6,
    'ciao': 3600*6,
    'epinions': 3600*6,
    'delicious': 3600*6,
    'reddit': 3600,
}

for data in datas:
    print('-'*20)
    print(data)
    diff_tm = []
    session_ids = []
    for usr, df in dfr[data].groupby('user'):
        diff = df.timestamp.sort_values().diff().fillna(0)
        sess_id = str(usr) + '_' + \
            (diff > time_thresh[data]).cumsum().astype(str)
        session_ids.append(sess_id)
    dfr[data]['sessionId'] = pd.concat(session_ids)
    assert not dfr[data].sessionId.isna().any()

    dfr[data].drop_duplicates(['user', 'item', 'sessionId'], inplace=True)

# %%
# 去掉过短过长的session
min_sess_length = {
    'gowalla': 2,
    'lastfm': 2,
    'douban': 2,
    'epinions': 2,
    'ciao': 2,
    'delicious': 2,
    'reddit': 2,
}
max_sess_length = {
    'gowalla': 20,
    'lastfm': 20,
    'douban': 20,
    'epinions': 20,
    'ciao': 20,
    'delicious': 20,
    'reddit': 20,
}
print('=>remove sessions with length less than 10 and larger than 20')
print('=>remove sessions with length less than 10 and larger than 20', file=sys.stderr)
for data in datas:
    print('-'*20)
    print(data)

    print(f'Before: {len(dfr[data])}')
    dfr[data] = dfr[data][dfr[data].groupby(
        'sessionId').sessionId.transform('size') >= min_sess_length[data]]
    dfr[data] = dfr[data][dfr[data].groupby(
        'sessionId').sessionId.transform('size') <= max_sess_length[data]]
    print(f'After: {len(dfr[data])}')

# %%
# 查看分割的session长度计数
# t = ['Delicious', 'Reddit']

# fig, axs = plt.subplots(1, 2, figsize=(20, 5))
# for i, data in enumerate(datas):
#     sess_size_list = dfr[data].groupby('sessionId').sessionId.size()
#     sess_size_list.plot(kind='hist', ax=axs[i], title=data, bins=list(range(2, 21)), xticks=list(
#         range(2, 21, 2)), weights=np.ones(len(sess_size_list)) / len(sess_size_list))
#     axs[i].set_title(t[i])
#     axs[i].title.set_size(40)
#     axs[i].set_xlabel("Session Length", fontsize=35)
#     if i == 0:
#         axs[i].set_ylabel("pct. of Sessions", fontsize=32)
#     else:
#         axs[i].set_ylabel("", fontsize=40)
#     axs[i].tick_params(axis='both', labelsize=40)
#     axs[i].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

# %%
# fig.savefig('sesslen.pdf', dpi=400, bbox_inches='tight')

# %% [markdown]
# ## 统计

# %%
# 数量
for data in datas:
    print('-'*20)
    print(data)
    print(f'#users:   {dfr[data].user.nunique()}')
    print(f'#items:   {dfr[data].item.nunique()}')
    print(f'#interactions:   {len(dfr[data])}')
    print(f'#sess:    {dfr[data].sessionId.nunique()}')
    print(
        f'sparsity:  {(len(dfr[data])/(dfr[data].user.nunique()*dfr[data].item.nunique())):2f}')
#     print(f'#links    {len(dfn[data])}')

# %%

# for data in datas:
#     print('-'*20)
#     print(data)
#     gs = dfr[data].groupby('sessionId')
#     sz = gs.size()
#     print(len(sz[sz <= 5])/len(sz))


# %%
# 平均
for data in datas:
    print('-'*20)
    print(data)
    gs = dfr[data].groupby('sessionId')
    # 长度
    avg_size = gs.size().mean()
    avg_itm_size = gs.item.unique().apply(len).mean()
    med_size = gs.size().median()
    print(f'avg size/session: {avg_size:.2f}')
    print(f'med size/session: {med_size:.2f}')
    print(f'avg items/session: {avg_itm_size:.2f}')

    gu = dfr[data].groupby('user')
    avg_itm_usr = gu.item.unique().apply(len).mean()
    avg_ses_usr = gu.sessionId.unique().apply(len).mean()
    print(f'avg items/user: {avg_itm_usr:.2f}')
    print(f'avg sess/user: {avg_ses_usr:.2f}')

    gv = dfr[data].groupby('item')
    avg_usr_itm = gv.user.unique().apply(len).mean()
    avg_ses_itm = gv.sessionId.unique().apply(len).mean()
    print(f'avg users/item: {avg_usr_itm:.2f}')
    print(f'avg sess/item: {avg_ses_itm:.2f}')

#     gl = dfn[data].groupby('user')
#     avg_link = gl.size().mean()
#     print(f'avg links/user: {avg_link:.2f}')
#     gl2 = dfn[data].groupby('friend')
#     avg_link2 = gl2.size().mean()
#     print(f'avg links/user: {avg_link2:.2f}')

# %% [markdown]
# 结论(1:gowalla, 2:delicious, 3:epinions, 4:douban)
# 1. 1和2的session中可能会出现重复的item，而3和4不会；
# 2. 2和4的user-item联系更多，1和3比较稀疏
# 3. 1和2的用户链接是双向的，而3和4是单向的
# 4. 3的用户关系是trust，作为truster/trustee的的概率相等。4的关系是follow，follow的好友数量远大于被follow的个数

# %% [markdown]
# ## 保存

# %%
print('=>save all data to pickle file session_data.pkl')
print('=>save all data to pickle file session_data.pkl', file=sys.stderr)
# DATA_DIR = path(r'E:\\appdata\\jianguoyun\\2004shortsession\\data')
with open(DATA_DIR/'session_data.pkl', 'wb') as f:
    pickle.dump(dfr, f)
# with open(DATA_DIR/'network.pkl', 'wb') as f:
#     pickle.dump(dfn, f)

# %%
