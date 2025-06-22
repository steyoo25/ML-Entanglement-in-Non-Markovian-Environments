# Authors: Stephen Yoon, Yifan Shi

# Importing functions from other local files

import pandas as pd
from MLP import build_mlp


def main():

    # Construct the file name
    csv_file = "data_1.csv"

    try:
        # Read the CSV file
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        print("Data loaded successfully!")

        # Choose whether to shuffle the dataset
        shuffled = (input('Shuffle dataset? (y/n): ').strip().upper() == 'Y')

        # Set training dataset proportion
        if shuffled:
            train_amt = float(input('Enter training amount in percentage (0-100): '))
        else:
            train_amt = float(input('Enter training amount as a time value: '))

        # Run MLP training
        build_mlp(df, shuffled, train_amt)

    except FileNotFoundError:
        print(f"Error: File {csv_file} not found. Please check your input values.")


if __name__ == '__main__':
    main()