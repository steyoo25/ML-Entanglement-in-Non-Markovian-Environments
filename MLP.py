# Authors: Stephen Yoon, Yifan Shi

import numpy as np
import pandas as pd
import optuna
import time
from display_graph import generate_graph
from display_CS import generate_CS
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def build_mlp(shuffled, train_amt, var_param, graph_type):
    # read the input
    df = pd.read_csv('input.csv')
    totalSamples = df.shape[0]
    all_input = df[['t','p']]
    tlist = df['t'].unique() # list of time vals
    all_conc = df['conc']

    if shuffled:
        X_train, X_test, y_train, y_test = train_test_split(all_input, all_conc, train_size = train_amt*0.01, random_state=42)
    else:
        # # train 15% from start, 15% from end
        # trainingSize = int(totalSamples * train_amt)//2
        # X_train = pd.concat([all_input.iloc[:trainingSize], all_input.iloc[-trainingSize:]])
        # y_train = pd.concat([all_conc.iloc[:trainingSize], all_conc.iloc[-trainingSize:]])
        # X_test = all_input.iloc[trainingSize:-trainingSize]
        # y_test = all_conc.iloc[trainingSize:-trainingSize]
        range1 = (all_input['t'] > 0) & (all_input['t'] <= train_amt)
        range2 = (all_input['t'] > tlist[-1]-train_amt) & (all_input['t'] <= tlist[-1])
        X_train = pd.concat([all_input[range1], all_input[range2]])
        y_train = pd.concat([all_conc.iloc[:all_input[range1].shape[0]], all_conc.iloc[-all_input[range2].shape[0]:]])
        X_test = all_input[(all_input['t'] > train_amt) & (all_input['t'] <= tlist[-1]-train_amt)]
        y_test = all_conc.iloc[all_input[range1].shape[0]:-all_input[range2].shape[0]]

    # measure time to train the model
    start_time = time.time()

    # hyperparameter tuning with OPTUNA
    def objective(trial):
        layer_sizes = tuple(trial.suggest_int(f'n_units_layer{i}', 1, 500) for i in range(trial.suggest_int('n_layers', 1, 10)))
        mlp = MLPRegressor(hidden_layer_sizes=layer_sizes, max_iter=100, random_state=42)
        mlp.fit(X_train, y_train)
        mape = mean_absolute_percentage_error(np.array(y_train), np.array(mlp.predict(X_train)))
        mse = mean_squared_error(np.array(y_train), np.array(mlp.predict(X_train)))
        return mse/totalSamples + 100*(mape/totalSamples)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    # select the best parameters
    best_params = study.best_params
    best_params_tuple = []
    layer_size = best_params['n_layers']
    for i in range(layer_size):
        best_params_tuple.append(best_params[f'n_units_layer{i}'])
    best_params_tuple = tuple(best_params_tuple)
    best_model = MLPRegressor(best_params_tuple, max_iter = 100, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # display score (R^2, RMSE, MAPE)
    print(f'Time (training only): {time.time() - start_time:.3f} seconds')
    print(f'''Scoring Metrics:
    R^2:{r2_score(y_test, y_pred)}
    MSE: {mean_squared_error(y_test, y_pred)}
    MAPE: {mean_absolute_percentage_error(np.array(y_test), np.array(y_pred))}''')

    if graph_type == 'g':
        generate_graph(df, X_train, best_model, var_param)
    else:
        generate_CS(df, best_model, var_param)