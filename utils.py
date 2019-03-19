import numpy as np
import itertools
import tqdm
from sqlalchemy import create_engine
import pandas as pd
import tensorflow as tf
import os


def prelu(inputs, name):
    alpha = tf.get_variable(name=name, shape=inputs.get_shape()[-1], dtype=inputs.dtype,
                            initializer=tf.zeros_initializer(), trainable=True)
    # alpha = tf.zeros_like(inputs, dtype = inputs.dtype)
    return alpha * tf.minimum(0.0, inputs) + tf.maximum(0.0, inputs)


def batch_iter(*data, batch_size, shuffle=True):
    if shuffle:
        shuffle_mask = np.random.permutation(len(data[0]))
    batch = []
    for d in data:
        if shuffle:
            d = d[shuffle_mask]
        batch.append(d)
    idx = 0
    length = len(data[0])
    while idx < length:
        if len(batch) == 1:
            yield batch[0][idx: idx + batch_size]
        else:
            yield tuple(map(lambda x: x[idx: idx + batch_size], batch))
        idx += batch_size
    return


class Early_stopping:
    '''
    Early_stopping)
        특정 에포크동안 계속 loss 가 상승하면 training stop

    Method)
        constructor_params :
                 - patience : 허용하는 에포크 수, patience 이상 연속적으로 loss가 상승하면 ealry stop
        validate : loss를 파라미터로 받아서 early stop을 할 것인지 여부 결정
        - (inputs : loss --> outputs : True or False )
    '''
    def __init__(self, patience=0):
        # step: 연속적으로 증가한 횟수
        self._step = 0
        # loss : 바로 전 loss (초기값 : inf)
        self._measure = np.inf
        self.patience = patience
    def __call__(self, measure):
        '''
        minimize measure
        '''
        # loss 함수가 바로 전 loss 함수보다 클경우 step 업데이트
        if self._measure < measure:
            self._step += 1
            # 연속적으로 증가한 횟수가 설정한 patience보다 높으면 early stop
            if self._step >= self.patience:
                print('Early stop training')
                return True
        # loss 함수가 바로 전 loss 함수보다 작을경우 step 초기화 및 loss 업데이트
        else:
            self._step = 0
            self._measure = measure
        return False



def set_save_path(model_name, start_date, end_date, model_num):
    '''
    model parameter를 넣으면 기간별 모델 따로 저장
    저장 경로 설정 및 생성
    params : 만들 파라미터

    '''
    # 저장 경로 설정 by model_name in parameter dict

    save_path = r'./training_result/{}/saved/{}/{}_{}/'.format(model_name,
                                                              model_num,
                                                              start_date.strftime('%Y%m%d'),
                                                              end_date.strftime('%Y%m%d'))
    # 저장용 directory 생성
    try:
        os.mkdir(r'./training_result/{}/saved/{}/'.format(model_name, model_num))
    except FileExistsError:
        print('Caution : Model directory already exists')
        pass

    try:
        os.mkdir(save_path)
    except FileExistsError:
        print('Caution : Model directory already exists')
        pass

    #     try:
    #         os.mkdir(params['save_path']+'best/')
    #     except FileExistsError:
    #         print('Caution : Model directory already exists')
    #         pass
    return save_path


def convert_type_in_params(params):
    # param안에 내포되어있는 것들
    # True, False, float, int, list, str, function
    for i in params:
        if params[i] == 'True' or params[i] == 'False':
            params[i] = True if 'True' else False
        else:
            try:
                if i in ['learning_rate', 'lambda', 'decay', 'l1_beta', 'l2_beta']:
                    params[i] = float(params[i])
                else:
                    params[i] = int(params[i])
            except ValueError:
                if '<function' in params[i]:
                    try:
                        params[i] = globals()[params[i].split(' ')[1]]
                    except KeyError:
                        params[i] = getattr(tf.nn, params[i].split(' ')[1])
                elif params[i][0] == '[':
                    if 'keep' in i:

                        params[i] = list(map(lambda x: float(x), params[i][1: -1].split(',')))
                    elif 'norm' in i:
                        params[i] = list(map(lambda x: True if x == 'True' else False, params[i][1: -1].split(',')))
                    else:
                        params[i] = list(map(lambda x: int(x), params[i][1: -1].split(',')))


def permutation_importance(model, sess, data, target, feature_name, score_fn=lambda x, y: np.mean(np.equal(x, y)), n_iters=2, best = True):
    # 먼저 기본 점수
    prediction = model.predict(sess, np.expand_dims(data, 1), restore=True, restore_best=best)
    # 일단 base_accuracy
    score = score_fn(target.values, prediction.squeeze())

    # 각 feature에 대해 반복
    feature_importances = {}
    for feat_num, feat in enumerate(feature_name):
        # 설정된 회수만큼 반복
        scores = []
        for _ in range(n_iters):
            # 각 feature에 대해서 shuffle된 데이터 구하기
            shuffled_data = data.copy()
            rng = np.random.permutation(len(data))
            shuffled_data[:, [feat_num]] = shuffled_data[:, [feat_num]][rng]
            # shuffle된 데이터로 accuracy 측정
            prediction = model.predict(sess, np.expand_dims(shuffled_data, 1), verbose=False)
            scores.append(score_fn(target.values, prediction.squeeze()))

        # 차이만큼이 feature importance
        feature_importances[feat] = score - scores
    return feature_importances


def get_scores(target, pred):
    target_ = (target > 0).astype(int)
    pred_ = (pred > 0).astype(int)
    return np.mean(np.equal(target_, pred_))


def unique_permutations(iterable, r=None):
    previous = tuple()
    for p in itertools.permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p

def grid_iter(setting, r):
    for i in itertools.combinations_with_replacement(setting, r):
        i = np.array(i, dtype = np.float32)
        if np.sum(i) == 1:
            for k in unique_permutations(list(i), r):
                yield list(k)


def cal_index_weights(start_date, end_date, ret_data):
    index_period_set = pd.date_range(start_date, end_date, freq='B')

    return_data = ret_data.copy()
    return_data.loc[start_date] = 0
    return_data = return_data.reindex(index_period_set.astype(str)).fillna(0) +1
    return_data.iloc[0] = 1
    return_data = return_data.cumprod(0)

    return return_data
def get_bounded_weights(x, lower_bounds, upper_bounds, eps = 0.0001):
    count = 0
    while np.clip(x, lower_bounds, upper_bounds).sum() != 1:
        x = np.clip(x, lower_bounds, upper_bounds) + eps
        x /= x.sum()
        count += 1

        if count > 5000:
            break
    return x

def cal_index_price(pf, start_date, end_date, ret_data):
    w = cal_index_weights(start_date, end_date, ret_data)
    try:
        price = w[pf.index] * pf.values.squeeze()
    except KeyError:
        for j in pf.index:
            if j not in w.columns:
                pf = pf.drop(j)
                price = w[pf.index] * pf.values.squeeze()
    return price

def migration(df, name, engine, index_label=None):
    '''
    parameters
    '''
    pbar = tqdm.trange(len(df) // 1000 + 1)
    pbar.set_description(name)
    for i in pbar:
        if i == 0:
            df.iloc[i * 1000:1000 + i * 1000].to_sql(name, engine, index=False, index_label=index_label,
                                                     if_exists='replace')
            if index_label is not None:
                if type(index_label) == list:
                    engine.execute('alter table {} add constraint idx_{} primary key({});'.format(name, name,
                                                                                                   ', '.join(
                                                                                                       index_label)))
                else:
                    engine.execute(
                        'alter table {} add constraint idx_{} primary key({});'.format(name, name, index_label))
        else:
            df.iloc[i * 1000:1000 + i * 1000].to_sql(name, engine, index=False, index_label=index_label,
                                                     if_exists='append')

def insert_data(df, name, engine):
    pbar = tqdm.trange(len(df) // 1000 + 1)
    pbar.set_description(f'append data in {name}')
    for i in pbar:
        df.iloc[i * 100:100 + i * 100].to_sql(name, engine, index = False, if_exists='append')

def update_data_back(df, name, engine, key1, key2 =None):
    pbar = tqdm.trange(len(df) // 1000 + 1)
    pbar.set_description(f'append data in {name}')
    key1_data = np.unique(df[key1])
    key1_data = "', '".join(key1_data)
    sql = f"delete from {name} where {key1} in ('{key1_data}')"
    if key2 is not None:
        key2_data = np.unique(df[key2])
        key2_data = "', '".join(key2_data)
        sql += f" and {key2} in ('{key2_data}')"
    engine.execute(sql)
    for i in pbar:
        df.iloc[i * 100:100 + i * 100].to_sql(name, engine, index = False, if_exists='append')

def update_data(df, name, engine, keys):
    pbar = tqdm.trange(len(df) // 1000 + 1)
    pbar.set_description(f'append data in {name}')
    
    
    if type(keys) == str:
        key_data = np.unique(df[keys])
        key_data = "', '".join(key_data)
        sql = f"delete from {name} where {keys} in ('{key_data}')"
    else:
        s = 0
        for key in keys:
            key_data = np.unique(df[key])
            key_data = "', '".join(key_data)
            try:
                sql += f" and {key} in ('{key_data}')"
            except NameError:
                sql = f"delete from {name} where {key} in ('{key_data}')"
            s+=1
    engine.execute(sql)
    for i in pbar:
        df.iloc[i * 1000:1000 + i * 1000].to_sql(name, engine, index = False, if_exists='append')

def update_missing_data(code):


    engine = create_engine('postgresql://postgres:WJDDNRtkfkd1@@127.0.0.1:5432/qpms')

    engine2 = create_engine('mssql+pymssql://sa:qpmsdb!@#@10.93.20.102:1433/EUMQNTDB')

    print('start_fetching')

    f = pd.read_sql("select * from market..mb where code in ('{}')".format("', '".join(code)), engine2)
    print('fetch done')
    update_data(f, 'close_price', engine, ['fin_prod_id', 'base_dt'])





def get_start_rebal_dateset(start_date, end_date, rebalancing_period = 3, fixed_length = None):
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    start_dates = []
    end_dates = []
    start_date += pd.tseries.offsets.BDay(0)
    if fixed_length:
        start_dates = pd.date_range(start_date, end_date, freq = 'B')
        end_dates = start_dates.shift(fixed_length)
        end_dates = end_dates[end_dates <= pd.to_datetime(end_date)]
        start_dates = start_dates[:len(end_dates)]
        start_dates = start_dates.astype(str)
        end_dates = end_dates.astype(str)
        return start_dates, end_dates

    rebal_date = start_date + pd.tseries.offsets.relativedelta(months = rebalancing_period) + pd.tseries.offsets.BDay(-1)
    while rebal_date.date() < end_date:
        start_dates.append(str(start_date.date()))
        end_dates.append(str(rebal_date.date()))
        rebal_date = start_date + pd.tseries.offsets.relativedelta(months = rebalancing_period) + pd.tseries.offsets.BDay(0)
        start_date += pd.tseries.offsets.BDay(1)
    return start_dates, end_dates

def get_start_rebal_iters(start_date, end_date, rebalancing_period = 3, fixed_length = None):
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()

    if fixed_length:
        start_dates = pd.date_range(start_date, end_date, freq = 'B')
        end_dates = start_dates.shift(fixed_length)
        end_dates = end_dates[end_dates <= pd.to_datetime(end_date)]
        start_dates = start_dates[:len(end_dates)]
        start_dates = start_dates.astype(str)
        end_dates = end_dates.astype(str)
        for i, j in zip(start_dates, end_dates):
            yield i, j
    else:
        start_date += pd.tseries.offsets.BDay(0)
        rebal_date = start_date + pd.tseries.offsets.relativedelta(months = rebalancing_period) + pd.tseries.offsets.BDay(-1)
        while rebal_date.date() < end_date:
            yield str(start_date.date()), str(rebal_date.date())
            rebal_date = start_date + pd.tseries.offsets.relativedelta(months = rebalancing_period) + pd.tseries.offsets.BDay(0)
            start_date += pd.tseries.offsets.BDay(1)

def get_start_rebal_dates(start_date, end_date, rebalancing_period=1):
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    start_dates = []
    rebalancing_dates = []

    if start_date > end_date:
        raise ValueError('날짜를 잘못 입력하셨습니다. / you inputs wrong date, start_date > end_date')
    if start_date <= end_date - pd.tseries.offsets.relativedelta(months=rebalancing_period):
        start_dates.append(start_date)
        start_date += pd.tseries.offsets.BMonthEnd(rebalancing_period)
        start_date = start_date.date()
        rebalancing_dates.append(start_date)
    else:
        return np.array([]), ([])

    while start_date <= end_date - pd.tseries.offsets.relativedelta(months=rebalancing_period):
        start_dates.append(start_date + pd.tseries.offsets.BMonthBegin(1))

        start_date += pd.tseries.offsets.BMonthEnd(rebalancing_period)
        start_date = start_date.date()
        rebalancing_dates.append(start_date)
    # print(start_date, end_date - pd.tseries.offsets.relativedelta(months = rebalancing_period))
    return np.array(start_dates).astype('datetime64[D]').astype(str), np.array(rebalancing_dates).astype('datetime64[D]').astype(str)



def indexing_data_with_date(data, date):
    temp_date = date
    while True:
        try:
            res = data.loc[temp_date]
            break
        except KeyError:
            temp_date = (pd.to_datetime(temp_date) - pd.tseries.offsets.BDay(1)).strftime('%Y%m%d')
    return res

        
        

