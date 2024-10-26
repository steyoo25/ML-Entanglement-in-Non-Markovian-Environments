import pandas as pd

# Read the file, skipping the first line with headers
df = pd.read_csv('graph_txt/2s_o.txt', delim_whitespace=True)

# Replace negative values in the rightmost column with 0
df['conc'] = df['conc'].apply(lambda x: 0 if x < 0 else x)

# Save the modified data back to a file
df.to_csv('graph_txt/2s_o.txt', sep=' ', index=False)