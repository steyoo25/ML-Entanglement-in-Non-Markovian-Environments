# Authors: Stephen Yoon, Yifan Shi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from input_handler import get_ent  # Analytical solution function

# **Select bath and var_param**
bath = "c"  # "s" or "c"
var_param = "f"  # "g", "o", "f"

# **Automatically generate file paths**
base_path = "C:/shiyifan/PythonProject/MLproject/"
files = [
    f"{base_path}2{bath}_{var_param}_pred_shuffled.txt",
    f"{base_path}2{bath}_{var_param}_pred_unshuffled.txt"
]

# **Define corresponding labels and colors**
labels = ["Shuffled", "Unshuffled", "Analytical"]
colors = ["b", "r", "k"]  # Blue (Shuffled), Red (Unshuffled), Black (Analytical)

# **Target p value**
target_p = 0.75
tolerance = 1e-3  # Allowed error margin

plt.figure(figsize=(7, 5))  # Canvas size

# **Iterate through the two machine learning files**
for file, label, color in zip(files, labels[:2], colors[:2]):
    try:
        data = pd.read_csv(file, delim_whitespace=True)  # Read data
    except FileNotFoundError:
        print(f"File not found: {file}")
        continue

    filtered_data = data[np.abs(data['p'] - target_p) <= tolerance]  # Filter data close to target_p

    if not filtered_data.empty:
        filtered_data = filtered_data.sort_values(by='t')  # Sort by t value
        conc_values = np.maximum(filtered_data['conc'], 0)  # **Correct conc values**

        plt.plot(
            filtered_data['t'], conc_values,
            marker='o', linestyle='--', color=color,
            markersize=2, label=label
        )  # **Adjust point size & change line style to '--'**
    else:
        print(f"No data points close to {var_param} = {target_p} in {file}, try adjusting tolerance!")

# **Compute Analytical solution data**
t_values = np.linspace(0, 4, 50)
analytical_conc = [get_ent(t, target_p, mode_sel=f'2{bath}', var_param=var_param) for t in t_values]
analytical_conc = np.maximum(analytical_conc, 0)  # **Correct conc values**

# **Plot Analytical solution curve**
plt.plot(
    t_values, analytical_conc,
    marker='o', linestyle='--', color=colors[2],
    markersize=2, label="Analytical"
)

# **Add legend and labels**
plt.xlabel("t")
plt.ylabel("C")
plt.title(f"Section Graph ({var_param}={target_p}, bath={bath})")
plt.legend()
plt.grid()
plt.show()