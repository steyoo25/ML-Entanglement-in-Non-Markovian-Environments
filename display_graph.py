# Authors: Stephen Yoon, Yifan Shi

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_graph(df, X_train, best_model, var_param):
    tlist = df['t'].unique() # list of time vals
    plist = df['p'].unique() # list of system param vals

    X, Y = np.meshgrid(plist, tlist)

    Z1, Z2, Z3 = [], [], []
    for t in tlist:
        row1, row2, row3 = [], [], []
        for p in plist:
            
            anal_val = (df.loc[(df['t'] == t) & (df['p'] == p), 'conc']).values[0]
            row1.append(max(0, anal_val))
            
            pred = best_model.predict(pd.DataFrame({'t': [t], 'p': [p]}))[0]
            row2.append(max(0, pred))

            if ((X_train['t'] == t) & (X_train['p'] == p)).any():
                row3.append(anal_val)
            else:
                row3.append(None)

        Z1.append(row1)
        Z2.append(row2)
        Z3.append(row3)

    Z1, Z2, Z3 = map(np.array, (Z1, Z2, Z3))

    if var_param == 'o': name = 'Ω'
    elif var_param == 'g': name = 'γ'
    elif var_param == 'f': name = 'F'

    if name in 'ΩF': X, Y = Y, X

    # ANALYTICAL MODEL
    cmap = plt.get_cmap('Greys')
    colors_Z1 = np.array([[cmap(value) for value in row] for row in Z1])
    colors_Z1[Z1 == 0] = (1.0, 0.25, 0.0, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if name in 'ΩF':
        ax.set_ylabel(name)
        ax.set_xlabel('t')
    elif name in 'γ':
        ax.invert_yaxis()
        ax.set_xlabel(name)
        ax.set_ylabel('t')
    ax.plot_surface(X, Y, Z1, facecolors=colors_Z1, edgecolor='none')
    plt.title('No ML')

    # ML GRAPH
    cmap_ml = plt.get_cmap('plasma')
    colors_Z2 = np.array([[cmap_ml(value) for value in row] for row in Z2])
    colors_Z2[Z2 == 0] = (1.0, 0.25, 0.0, 1)
    fig_ML = plt.figure()  # plot graph with ML predictions
    ax = fig_ML.add_subplot(111, projection='3d')
    if name in 'ΩF':
        ax.set_ylabel(name)
        ax.set_xlabel('t')
    elif name in 'γ':
        ax.invert_yaxis()
        ax.set_xlabel(name)
        ax.set_ylabel('t')
    ax.plot_surface(X, Y, Z2, facecolors=colors_Z2, edgecolor='none')
    plt.title('ML')

    # GRAPH FOR JUST TRAINING DATA
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    if name in 'ΩF':
        ax.set_ylabel(name)
        ax.set_xlabel('t')
    elif name in 'γ':
        ax.invert_yaxis()
        ax.set_xlabel(name)
        ax.set_ylabel('t')
    ax.plot_surface(X, Y, Z3)
    plt.title('Training')

    # # CROSS SECTION GRAPH
    # p = 0.6

    # plt.figure()
    # Y1, Y2 = [], []
    # for t in tlist:
    #     Y1.append(df.loc[(df['t'] == t) & (df['p'] == p), 'conc'].values[0])
    #     Y2.append(best_model.predict(pd.DataFrame({'t': [t], 'p': [p]}))[0])

    # Y1, Y2 = map(np.array, (Y1, Y2))

    # Y1[Y1 < 0] = 0
    # Y2[Y2 < 0] = 0  # all negative values represent ESD.

    # plt.plot(tlist, Y1, label='Analytical')
    # plt.plot(tlist, Y2, label='MLP')

    # plt.xlabel('t')
    # plt.ylabel('Concurrence')
    # plt.title(f'Analytical vs. MLP @ {name} = {p:.1f}')
    # plt.legend()

    plt.show()