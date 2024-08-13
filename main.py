# Authors: Stephen Yoon, Yifan Shi

# Importing functions from other local files
from input_handler import generate_input
from MLP import build_mlp

# Main function to run MLP
def main():
    print('Please enter your selection as indicated in the parentheses (e.g., \'2s\', \'g\'')

    # Ask user to select 
    mode_sel = input('Enter environment/bath: 2-qubit separate (2s), 2-qubit common (2c), 3-qubit separate (3s), 3-qubit common(3c): ')
    var_param = input('Enter varying system parameter, γ(g) or Ω(o): ')
    p_range = [float(x) for x in input('Enter the range for the varying parameter, separated by a space (e.g. 0 5): ').split()]
    t_range = [float(x) for x in input('Enter the range for time, separated by a space (e.g. 0 4): ').split()]

    # Generate analytical data
    generate_input(mode_sel, var_param, p_range, t_range)

    # Building MLP & Graphing
    graph_type = input('3D Graph(g) or 2D cross section(c): ')
    shuffled = ((input('Shuffle dataset? (y/n): ')).upper() == 'Y')
    train_amt = float(input('Enter training amount - time constant(unshuffled)/percentage(shuffled): '))

    build_mlp(shuffled, train_amt, var_param, graph_type)

if __name__ == '__main__':
    main()