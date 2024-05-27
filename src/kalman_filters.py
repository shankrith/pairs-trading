import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from time import time

def KFSmoother(prices):

    kf = KalmanFilter(initial_state_mean = 0,
                      initial_state_covariance = 1,
                      transition_covariance = 0.05,
                      transition_matrices = np.eye(1),
                      observation_matrices = np.eye(1),
                      observation_covariance = 0.2)

    state_means, _ = kf.filter(prices.values)
    return pd.Series(state_means.flatten(), index = prices.index)

def KFHedgeRatio(x, y):
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis = 1)

    kf = KalmanFilter(n_dim_obs = 1,
                      n_dim_state = 2,
                      initial_state_mean = [0, 0],
                      initial_state_covariance = np.ones((2, 2)),
                      transition_matrices = np.eye(2),
                      observation_matrices = obs_mat,
                      observation_covariance = 1,
                      transition_covariance = trans_cov)

    state_means, _ = kf.filter(y.values)
    return -state_means

def estimate_half_life(spread):
    X = spread.shift().iloc[1:].to_frame().assign(const = 1)
    y = spread.diff().iloc[1:]
    beta = (np.linalg.inv(X.T @ X) @ X.T @ y).iloc[0]
    halflife = int(round(-np.log(2) / beta, 0))
    return max(halflife, 1)

def get_spread(candidates, prices):
    pairs = []
    half_lives = []

    periods = pd.DatetimeIndex(sorted(candidates['test_end'].unique()))
    start = time()
    for p, test_end in enumerate(periods, 1):
        start_iteration = time()

        period_candidates = candidates.loc[candidates['test_end'] == test_end, ['y', 'x']]
        trading_start = test_end + pd.DateOffset(days = 1)
        t = trading_start - pd.DateOffset(years = 2)
        T = trading_start + pd.DateOffset(months = 6) - pd.DateOffset(days = 1)
        max_window = len(prices.loc[t:T,:].index)
        for i, (y, x) in enumerate(zip(period_candidates['y'], period_candidates['x']), 1):
            if i % 1000 == 0:
                msg = f'{i:5.0f} | {time() - start_iteration:7.1f} | {time() - start:10.1f}'
                print(msg)
            pair = prices.loc[t:T, [y, x]]
            pair['{}_smooth'.format(y)] = KFSmoother(prices.loc[t:T, y])
            pair['{}_smooth'.format(x)] = KFSmoother(prices.loc[t:T, x])
            pair['hedge_ratio'] = KFHedgeRatio(y = pair['{}_smooth'.format(y)], x = pair['{}_smooth'.format(x)])[:,0]
            pair['spread'] = pair[y] + pair[x] * pair['hedge_ratio']
            half_life = estimate_half_life(pair['spread'].loc[t:test_end])
            spread = pair['spread'].rolling(window = min(2 * half_life, max_window))
            pair['z_score'] = (pair['spread'] - spread.mean())/spread.std()
            
            pairs.append(pair.loc[trading_start:T].assign(s1 = y, s2 = x, period = p, pair = i).drop([x, y, '{}_smooth'.format(y), '{}_smooth'.format(x)], axis = 1))
            half_lives.append([test_end, y, x, half_life])

    return pairs, half_lives