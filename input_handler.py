# Authors: Stephen Yoon, Yifan Shi

import numpy as np
from math import sqrt

# system parameters
G = 1 # Γ
L = 1/2 # λ
y = 0.8 # default γ
O = 1 # default Ω
i = 1j # imaginary number

def ent_2s(J,a):
    return 2/3*(abs(np.exp(-4*(L**2)*J))-sqrt(a*(1-a)))
def ent_2c(J,F):
    return 2*(abs((1-4*F)/6*np.exp(-8*(L**2)*J))-(1-F)/3)
def ent_3s(J,F):
    return (1-F)/2*np.exp(-6*(L**2)*J)-F/8
def ent_3c(J,F):
    return (1-F)/2*np.exp(-12*(L**2)*J)-F/8

def get_ent(t, p, mode_sel, var_param):
    if var_param == 'g': J = (p*G*(-1+np.exp(-t*(p-i*O))+t*(p-i*O)))/(2*((p-i*O)**2))+(p*G*(-1+np.exp(-t*(p+i*O))+t*(p+i*O)))/(2*((p+i*O)**2))
    elif var_param == 'o': J = (y*G*(-1+np.exp(-t*(y-i*p))+t*(y-i*p)))/(2*((y-i*p)**2))+(y*G*(-1+np.exp(-t*(y+i*p))+t*(y+i*p)))/(2*((y+i*p)**2))

    if mode_sel == '2s': ent = np.real(ent_2s(J,1/3)) # alpha = 1/3
    elif mode_sel == '2c': ent = np.real(ent_2c(J,0.75)) # F = 0.75 (negativity constant)
    elif mode_sel == '3s': ent = np.real(ent_3s(J,0.5)) # F = 0.5
    elif mode_sel == '3c': ent = np.real(ent_3c(J,0.5)) # F = 0.5
    
    return ent

def generate_input(mode_sel, var_param, p_range, t_range):
    NUM_TIME_VALS, NUM_PARAM_VALS = 80, 80 # adjust to change the # of samples
    tlist = np.linspace(max(t_range[0], 1e-12), t_range[1]*(1/G), NUM_TIME_VALS) # time list
    plist = np.linspace(max(p_range[0], 1e-12), p_range[1]*G, NUM_PARAM_VALS) # param list
    
    all_input = [[t, x] for t in tlist for x in plist]

    file = open('input.csv', 'w')
    file.write('t,p,conc\n')
    for t, p in all_input:
        ent = get_ent(t,p,mode_sel,var_param)
        file.write(f'{t},{p},{ent}\n')