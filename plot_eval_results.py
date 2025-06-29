from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing evaluation result files
eval_dir = Path('eval_results')
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

# List all files in the eval_results directory
files = sorted([f for f in eval_dir.iterdir() if f.is_file()])

# Print files with numbers
for idx, fpath in enumerate(files, 1):
    print(f"{idx}: {fpath.name}")

# Await user input
num = input("Enter the number of the file to plot: ")
try:
    num = int(num)
    if not (1 <= num <= len(files)):
        raise ValueError
except ValueError:
    print("Invalid input.")
    exit(1)

# Get the selected file
selected_file = files[num - 1]
stem = selected_file.stem

# Read the file as a pandas dataframe
df = pd.read_csv(selected_file)

# Columns to plot
validate_columns = [
    'degree_validate',
    'clustering_validate',
    'orbits4_validate'
]
test_columns = [
    'degree_test',
    'clustering_test',
    'orbits4_test'
]

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot validate columns
for col in validate_columns:
    if col in df.columns:
        axes[0].plot(df['epoch'], df[col], label=col)
    else:
        print(f"Warning: Column '{col}' not found in {selected_file.name}")
axes[0].set_ylabel('validate value')
axes[0].set_title(f"{stem} - Validate")
axes[0].legend()
axes[0].grid(True)

# Plot test columns
for col in test_columns:
    if col in df.columns:
        axes[1].plot(df['epoch'], df[col], label=col)
    else:
        print(f"Warning: Column '{col}' not found in {selected_file.name}")
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('test value')
axes[1].set_title(f"{stem} - Test")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

# Save plot to figures directory
output_file = figures_dir / f"{stem}.png"
plt.savefig(output_file)
print(f"Plot saved as {output_file}")
plt.show()
