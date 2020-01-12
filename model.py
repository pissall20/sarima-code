from datetime import timedelta
from multiprocessing import cpu_count
from time import time
from warnings import catch_warnings
from warnings import filterwarnings

import pandas as pd
from joblib import Parallel
from joblib import delayed
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm


# one-step sarima forecast
def sarima_forecast(history, config, prediction_steps=4, walk_forward=False, verbose=False):
    if not walk_forward:
        print(f"Making forecast with the best found (order, seasonality, trend) config: {config}\n")
    if verbose:
        print(f"Trying the (order, seasonality, trend) config:{config}\n")
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    y_hat = model_fit.predict(prediction_steps)
    return y_hat


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    print("Splitting the data into train and test.")
    n_test = int(n_test)
    return data[:-n_test], data[-n_test:]


# walk-forward validation for univariate data
def walk_forward_validation(train, test, cfg):
    internal_train, internal_test = train, test
    predictions = list()
    # seed history with training dataset
    history = [x for x in internal_train]
    # step over each time-step in the test set
    for i in range(len(internal_test)):
        # fit model and make forecast for history
        y_hat = sarima_forecast(history, cfg, prediction_steps=1, walk_forward=True)
        # store forecast in list of predictions
        predictions.append(y_hat)
        # add actual observation to history for the next loop
        history.append(internal_test[i])
    # estimate prediction error
    error = measure_rmse(internal_test.tolist(), predictions)
    return error


# score a model, return None on failure
def score_model(train, test, cfg, debug=False, verbose=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(train, test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(train, test, cfg)
        except:
            error = None
    # check for an interesting result
    if result and verbose:
        print(' > Model[%s] %.3f' % (key, result))
    return key, result


# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    # split dataset
    train, test = train_test_split(data, n_test)
    print("Doing a grid search. May take a while.")
    grid_start_time = time()
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(train, test, cfg) for cfg in tqdm(cfg_list))
        scores = executor(tasks)
    else:
        scores = [score_model(train, test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1]]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    grid_end_time = time()
    print(f"Time taken for grid search: {round(grid_end_time - grid_start_time, 2)} seconds")
    print(scores)
    return scores


# create a set of sarima configs to try
def sarima_configs(seasonal=None):
    print("Creating p,d,q parameters grid")
    if not seasonal:
        seasonal = [0]
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    cap_p_params = [0, 1, 2]
    cap_d_params = [0, 1]
    cap_q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in cap_p_params:
                        for D in cap_d_params:
                            for Q in cap_q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models


def custom_sarimax_prediction_function(file_path, date_col, y_column, test_size=0.1, freq='W', prediction_steps=4):
    # load dataset
    data_set = pd.read_csv(file_path)
    # Convert date column to date
    data_set[date_col] = pd.to_datetime(data_set[date_col])
    # Set the date as index
    data_set = data_set.set_index(date_col)
    data_series = data_set[y_column]
    # data split
    test_length = len(data_set) * test_size
    # test_length = 1

    # model configs
    # Input the seasonality config in terms of months. In this example I've given yearly seasonality
    # For quarterly seasonality you would send [0,3,6,9,12]
    seasonality = [0]
    config_list = sarima_configs(seasonal=seasonality)
    # grid search
    scores_all = grid_search(data_series, config_list, test_length)
    print('Done finding the best config for this given time series.\n There are the best 3 configs:')
    # list top 3 configs
    for top_config, least_error in scores_all[:3]:
        print(top_config, least_error)

    # Use the top config for best prediction
    final_prediction_steps = prediction_steps
    final_prediction = sarima_forecast(data_series, eval(scores_all[0][0]), final_prediction_steps)
    if freq == 'D':
        prediction_dates = [(max(data_set.index) + timedelta(days=steps)) for steps in
                            range(1, final_prediction_steps + 1)]
    else:
        # Weekly prediction dates
        prediction_dates = [(max(data_set.index) + timedelta(days=7 * steps)) for steps in
                            range(1, final_prediction_steps + 1)]

    prediction_df = pd.DataFrame(zip(prediction_dates, final_prediction), columns=[date_col, y_column])
    return prediction_df


if __name__ == '__main__':
    file = 'dataset_energy_timeseries_4_features_20130101_20191218.csv'
    date_column = "eventdate"
    kpi_column = "ear"
    print(custom_sarimax_prediction_function(file, date_column, kpi_column, prediction_steps=4))
