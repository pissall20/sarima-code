# grid search sarima hyperparameters for monthly car sales dataset
from math import sqrt
from multiprocessing import cpu_count
from time import time
from warnings import catch_warnings
from warnings import filterwarnings

from joblib import Parallel
from joblib import delayed
from pandas import read_csv, to_datetime
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
def walk_forward_validation(train, test, n_test, cfg):
    predictions = list()
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        y_hat = sarima_forecast(history, cfg, walk_forward=True)
        # store forecast in list of predictions
        predictions.append(y_hat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error


# score a model, return None on failure
def score_model(train, test, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(train, test, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(train, test, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return key, result


# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    print("Doing a grid search. May take a while.")
    # split dataset
    train, test = train_test_split(data, n_test)
    grid_start_time = time()
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(train, test, n_test, cfg) for cfg in tqdm(cfg_list))
        scores = executor(tasks)
    else:
        scores = [score_model(train, test, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1]]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    grid_end_time = time()
    print(f"Time taken for grid search: {round(grid_end_time - grid_start_time, 2)} seconds")
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


if __name__ == '__main__':
    date_column = "eventdate"
    kpi_column = "ear"
    # load dataset
    data_set = read_csv('dataset_energy_timeseries_4_features_20130101_20191218.csv')
    data_set[date_column] = to_datetime(data_set[date_column])
    data_set = data_set.set_index(date_column)
    data_series = data_set[kpi_column]
    # data = series.values
    print(data_series.shape)
    # data split
    # test_length = len(data_set) * 0.2
    test_length = 4

    # model configs
    # Input the seasonality config in terms of months. In this example I've given a 6 month
    config_list = sarima_configs(seasonal=[0])
    # grid search
    scores_all = grid_search(data_series, config_list, test_length)
    print('Done finding the best config for this given time series.\n There are the best 3 configs:')
    # list top 3 configs
    for top_config, least_error in scores_all[:3]:
        print(top_config, least_error)

    final_pred = sarima_forecast(data_series, scores_all[0], 4)
    print(final_pred)
