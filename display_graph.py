# Authors: Stephen Yoon, Yifan Shi

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_graph(df, X_train, best_model):
    tlist = df['t'].unique()  # list of time values
    glist = df['g'].unique()  # list of system parameter values

    X, Y = np.meshgrid(glist, tlist)

    Z1, Z2, Z3 = [], [], []
    for t in tlist:
        row1, row2, row3 = [], [], []
        for g in glist:
            anal_val = (df.loc[(df['t'] == t) & (df['g'] == g), 'Q']).values[0]
            row1.append(max(0, anal_val))

            pred = best_model.predict(pd.DataFrame({'t': [t], 'g': [g]}))[0]
            row2.append(max(0, pred))

            if ((X_train['t'] == t) & (X_train['g'] == g)).any():
                row3.append(anal_val)
            else:
                row3.append(None)

        Z1.append(row1)
        Z2.append(row2)
        Z3.append(row3)

    Z1, Z2, Z3 = map(np.array, (Z1, Z2, Z3))

    # ANALYTICAL MODEL
    cmap = plt.get_cmap('Greys')
    colors_Z1 = np.array([[cmap(value) for value in row] for row in Z1])
    colors_Z1[Z1 == 0] = (1.0, 0.25, 0.0, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('t')
    ax.set_ylabel('g')
    ax.set_zlabel('conc')
    ax.plot_surface(X, Y, Z1, facecolors=colors_Z1, edgecolor='none')
    plt.title('Analytical conc(t, g)')

    # ML GRAPH
    cmap_ml = plt.get_cmap('plasma')
    colors_Z2 = np.array([[cmap_ml(value) for value in row] for row in Z2])
    colors_Z2[Z2 == 0] = (1.0, 0.25, 0.0, 1)
    fig_ML = plt.figure()  # plot graph with ML predictions
    ax = fig_ML.add_subplot(111, projection='3d')
    ax.set_xlabel('g')
    ax.set_ylabel('t')
    ax.set_zlabel('conc')
    ax.plot_surface(X, Y, Z2, facecolors=colors_Z2, edgecolor='none')
    plt.title('ML Prediction of conc(t, g)')

    plt.show()
