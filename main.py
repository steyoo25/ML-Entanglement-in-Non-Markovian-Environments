# Authors: Stephen Yoon, Yifan Shi

# Importing functions from other local files
from input_handler import generate_input
from MLP import build_mlp

# Main function to run MLP
def main():
    print('Please enter your selection as indicated in the parentheses (e.g., \'2s\', \'g\')')

    # Ask user to select 
    mode_sel = input('Enter environment/bath: 2-qubit separate (2s), 2-qubit common (2c): ')
    var_param = input('Enter varying system parameter, γ(g), Ω(o), or F(f): ')
    p_range = [float(x) for x in input(f'Enter the range for {var_param}, separated by a space (e.g. 0 5): ').split()]
    p_step = float(input(f'Enter the step size for {var_param}: '))
    t_range = [float(x) for x in input('Enter the range for time, separated by a space (e.g. 0 4): ').split()]
    t_step = float(input(f'Enter the step size for time: '))

    # mode_sel = '2s'
    # var_param = 'f'
    # p_range = [0.25, 1]
    # p_step = 0.005
    # t_range = [0, 4]
    # t_step = 0.025

    # Generate analytical data
    generate_input(mode_sel, var_param, p_range, p_step, t_range, t_step)

    # Building MLP & Graphing
    shuffled = ((input('Shuffle dataset? (y/n): ')).upper() == 'Y')
    if shuffled:
        train_amt = float(input('Enter training amount in percentage (0-100): '))
    else:
        train_amt = float(input(f'Enter training amount in time constant to be taken from both ends ({t_range[0]}-{t_range[1]}): '))

    # shuffled = False
    # train_amt = 0.6
    
    build_mlp(shuffled, train_amt, var_param)

if __name__ == '__main__':
    main()