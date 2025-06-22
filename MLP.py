# Authors: Stephen Yoon, Yifan Shi

import numpy as np
import pandas as pd
import optuna
import time
from display_graph import generate_graph
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler



def build_mlp(df, shuffled, train_amt):

    all_input = df[['t', 'g']]
    all_target = df['Q']
    tlist = df['t'].unique() # list of time vals

    if shuffled:
        # Step 1: Split out the training dataset
        X_train, X_temp, y_train, y_temp = train_test_split(
            all_input,
            all_target,
            train_size=train_amt*0.01,
            random_state=42
        )

        # Step 2: Split validation and test sets from the remaining data
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.9,  # Test set accounts for 90% of the remaining data, validation set 10%
            random_state=42
        )
        shuffle_value = "SH"
    else:
        # Split the training dataset
        range1 = (all_input['t'] > 0) & (all_input['t'] <= train_amt)
        range2 = (all_input['t'] > tlist[-1] - train_amt) & (all_input['t'] <= tlist[-1])
        X_train = pd.concat([all_input[range1], all_input[range2]])
        y_train = pd.concat(
            [all_target.iloc[:all_input[range1].shape[0]], all_target.iloc[-all_input[range2].shape[0]:]])

        # Split the remaining data
        X_temp = all_input[(all_input['t'] > train_amt) & (all_input['t'] <= tlist[-1] - train_amt)]
        # y_temp = all_target.iloc[all_input[range1].shape[0]:-all_input[range2].shape[0]]
        range_temp = (all_input['t'] > train_amt) & (all_input['t'] <= tlist[-1] - train_amt)
        y_temp = all_target[range_temp]


        # Split validation and test sets from the remaining data
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.9, random_state=42)
        shuffle_value = "UNSH"

    # measure time to train the model
    start_time = time.time()

    # hyperparameter tuning with OPTUNA
    # BEFORE: neurons per layer - 1 to 500, total layers - 1 to 10, max_iter = 100
    def objective(trial):
        # layer_sizes = [trial.suggest_int(f'n_units_layer{i}', 1, 200) for i in range(trial.suggest_int('n_layers', 5, 20))]
        layer_sizes = [trial.suggest_int(f'n_units_layer{i}', 1, 200) for i in range(trial.suggest_int('n_layers', 5, 20))]

        # Suggest L2 regularization parameter
        alpha_l2 = trial.suggest_float('alpha_l2', 1e-6, 1e-3, log=True)

        learning_rate_init = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

        # mlp = MLPRegressor(hidden_layer_sizes=layer_sizes, max_iter=1000, random_state=42, alpha=alpha_l2, solver='adam', learning_rate_init=learning_rate_init)
        mlp = MLPRegressor(hidden_layer_sizes=layer_sizes, max_iter=1000, random_state=42, alpha=alpha_l2, solver='adam', learning_rate_init=learning_rate_init)
        
        mlp.fit(X_train, y_train)

        # Compute validation loss for hyperparameter tuning
        y_val_pre = mlp.predict(X_val)
        val_rmse = root_mean_squared_error(np.array(y_val), np.array(y_val_pre))
        # val_rmse = mean_squared_error(np.array(y_val), np.array(y_val_pre), squared=False)
        # val_rmse = np.sqrt(mean_squared_error(np.array(y_val), np.array(y_val_pre)))

        val_mape = mean_absolute_percentage_error(np.array(y_val), np.array(y_val_pre))

        # Return validation loss as the optimization target
        loss = 10*val_rmse + val_mape

        # Manually set an acceptable loss threshold
        max_loss_threshold = 1e-2  # Adjust this value as needed
        if loss <= max_loss_threshold:
            trial.study.stop()  # Manually stop further trials if desired loss is reached

        return loss
    # Create a study with a pruner to use early stopping
    pruner = SuccessiveHalvingPruner()
    sampler = TPESampler()  # Bayesian Optimization with TPE Sampler
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler=sampler)
    # Optimize with early stopping and manual threshold checking
    # study.optimize(objective, n_trials=1000)
    study.optimize(objective, n_trials=500)

    # select the best parameters
    best_params = study.best_params
    best_params_tuple = []
    layer_size = best_params['n_layers']
    for i in range(layer_size):
        best_params_tuple.append(best_params[f'n_units_layer{i}'])
    best_params_tuple = tuple(best_params_tuple)
    best_alpha_l2 = best_params['alpha_l2']
    # best_model = MLPRegressor(best_params_tuple, max_iter=500, random_state=42, alpha=best_alpha_l2, solver='adam')

    best_model = MLPRegressor(
        hidden_layer_sizes=best_params_tuple,
        max_iter=500,
        random_state=42,
        alpha=best_alpha_l2,
        solver='adam'
    )


    best_model.fit(X_train, y_train)

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    # display score (R^2, RMSE, MAPE)
    print(f'Time (training only): {time.time() - start_time:.3f} seconds')

    print(f'''Training Set Metrics:
    R^2: {r2_score(y_train, y_train_pred)}
    RMSE: {root_mean_squared_error(y_train, y_train_pred)}
    MAPE: {mean_absolute_percentage_error(y_train, y_train_pred)}''')

    print(f'''Validation Set Metrics:
    R^2: {r2_score(y_val, y_val_pred)}
    RMSE: {root_mean_squared_error(y_val, y_val_pred)}
    MAPE: {mean_absolute_percentage_error(y_val, y_val_pred)}''')

    print(f'''Test Set Metrics:
    R^2: {r2_score(y_test, y_test_pred)}
    RMSE: {root_mean_squared_error(y_test, y_test_pred)}
    MAPE: {mean_absolute_percentage_error(y_test, y_test_pred)}''')

    # Overall metrics (training + validation + test)
    X_all = pd.concat([X_train, X_val, X_test])
    y_all = pd.concat([y_train, y_val, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])

    overall_r2 = r2_score(y_all, y_all_pred)
    overall_rmse = root_mean_squared_error(y_all, y_all_pred)
    overall_mape = mean_absolute_percentage_error(y_all, y_all_pred)

    print(f'''Overall Metrics (Training + Validation + Test):
    R^2: {overall_r2}
    RMSE: {overall_rmse}
    MAPE: {overall_mape}''')

    # Generate performance metrics
    generate_graph(df, X_train, best_model)


    # Save test results
    # X_test['Q'] = y_test_pred

    # Predict Q values over the full input range (df)
    df_pred = df.copy()
    df_pred['Q'] = best_model.predict(df[['t', 'g']])

    output_path = f"C:/Users/raymo/OneDrive/Science/Projects/Summer 2023 instruction Entanglement Evolution in a NM Environment/Python/MLP_Jun_2025/gamma/SEP/data_1_{shuffle_value}.txt"
    with open(output_path, 'w') as f:
        f.write('t g Q\n')  # Header with space separation
        for index, row in df_pred.iterrows():
            f.write(f"{row['t']} {row['g']} {row['Q']}\n")  # Save data with space separation