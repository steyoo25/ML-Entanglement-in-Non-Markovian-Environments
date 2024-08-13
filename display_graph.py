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

            row2.append(max(0, best_model.predict(pd.DataFrame({'t': [t], 'p': [p]}))[0]))

            row3.append(anal_val if ((X_train['t'] == t) & (X_train['p'] == p)).any() else None)

        Z1.append(row1)
        Z2.append(row2)
        Z3.append(row3)

    Z1, Z2, Z3 = map(np.array, (Z1, Z2, Z3))

    if var_param == 'o': x_name = 'Ω'
    elif var_param == 'g': x_name = 'γ'

    cmap = plt.get_cmap('Greys')
    colors_Z1 = np.array([[cmap(value) for value in row] for row in Z1])
    colors_Z1[Z1 == 0] = (1.0, 0.25, 0.0, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z1, facecolors=colors_Z1, edgecolor='none')
    ax.invert_yaxis()
    ax.set_ylabel('t')
    ax.set_xlabel(x_name)
    if x_name == 'Ω': ax.invert_xaxis()
    plt.title('No ML')
    
    cmap_ml = plt.get_cmap('plasma')
    colors_Z2 = np.array([[cmap_ml(value) for value in row] for row in Z2])
    colors_Z2[Z2 == 0] = (1.0, 0.25, 0.0, 1)
    fig_ML = plt.figure()  # plot graph with ML predictions
    ax = fig_ML.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z2, facecolors=colors_Z2, edgecolor='none')
    ax.invert_yaxis()
    ax.set_xlabel(x_name)
    ax.set_ylabel('t')
    if x_name == 'Ω': ax.invert_xaxis()
    plt.title('ML')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z3)
    ax.invert_yaxis()
    ax.set_ylabel('t')
    ax.set_xlabel(x_name)
    if x_name == 'Ω': ax.invert_xaxis()
    plt.title('Training')
    
    plt.show()