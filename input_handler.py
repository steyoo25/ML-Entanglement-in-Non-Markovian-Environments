# Authors: Stephen Yoon, Yifan Shi

import numpy as np
from math import sqrt

# system parameters
G = 1 # Γ
L = 0.5 # λ
y = 1 # default γ
O = 0.5 # default Ω
F = 0.75
i = 1j # imaginary number

def ent_2s(J,F):
    return abs((1-4*F)/3*np.exp(-J))-abs(2/3*(1-F))
def ent_2c(J,F):
    return abs((1-4*F)/3*np.exp(-2*J))-abs(2/3*(1-F))

def get_ent(t, p, mode_sel, var_param):
    if var_param == 'f':
        J = (y*G*(-1+np.exp(-t*(y-i*O))+t*(y-i*O)))/(2*((y-i*O)**2))\
            +(y*G*(-1+np.exp(-t*(y+i*O))+t*(y+i*O)))/(2*((y+i*O)**2))

        if mode_sel == '2s':
            ent = np.real(ent_2s(J,p))
        elif mode_sel == '2c':
            ent = np.real(ent_2c(J,p)) 
    
    else:
        if var_param == 'g':
            J = (p*G*(-1+np.exp(-t*(p-i*O))+t*(p-i*O)))/(2*((p-i*O)**2))\
                +(p*G*(-1+np.exp(-t*(p+i*O))+t*(p+i*O)))/(2*((p+i*O)**2))
        elif var_param == 'o':
            J = (y*G*(-1+np.exp(-t*(y-i*p))+t*(y-i*p)))/(2*((y-i*p)**2))\
                +(y*G*(-1+np.exp(-t*(y+i*p))+t*(y+i*p)))/(2*((y+i*p)**2))
        
        if mode_sel == '2s':
            ent = np.real(ent_2s(J,F))
        elif mode_sel == '2c':
            ent = np.real(ent_2c(J,F))
    return ent

def generate_input(mode_sel, var_param, p_range, p_step, t_range, t_step):
    p0, pf = p_range[0], p_range[1]
    t0, tf = t_range[0], t_range[1]
    NUM_TIME_VALS, NUM_PARAM_VALS = (pf-p0)/p_step, (tf-t0)/t_step # adjust to change the # of samples
    plist = np.linspace(max(p0, 1e-12), pf*G, int(NUM_PARAM_VALS)) # param list
    tlist = np.linspace(max(t0, 1e-12), tf*(1/G), int(NUM_TIME_VALS)) # time list
    
    all_input = [[t, p] for t in tlist for p in plist]

    file = open('input.csv', 'w')
    file.write('t,p,conc\n')
    for t, p in all_input:
        ent = get_ent(t,p,mode_sel,var_param)
        file.write(f'{t},{p},{ent}\n')