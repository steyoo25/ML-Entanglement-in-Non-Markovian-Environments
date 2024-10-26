# Authors: Stephen Yoon, Yifan Shi

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_CS(df, best_model, var_param):    
    tlist = df['t'].unique()
    
    if var_param == 'o': y_name = 'Ω'
    elif var_param == 'g': y_name = 'γ'
    elif var_param == 'f': y_name = 'F'

    Y1, Y2 = [], []
    p = 0.6037735849059057

    for t in tlist:
        Y1.append(df.loc[(df['t'] == t) & (df['p'] == p), 'conc'].values[0])
        Y2.append(best_model.predict(pd.DataFrame({'t': [t], 'p': [p]}))[0])

    Y1, Y2 = map(np.array, (Y1, Y2))

    Y1[Y1 < 0] = 0
    Y2[Y2 < 0] = 0  # all negative values represent ESD.

    plt.plot(tlist, Y1, label='Analytical')
    plt.plot(tlist, Y2, label='MLP')

    plt.xlabel('t')
    plt.ylabel('Concurrence')
    plt.title(f'Analytical vs. MLP @ {y_name} = {p:.1f}')
    plt.legend()

    plt.show()