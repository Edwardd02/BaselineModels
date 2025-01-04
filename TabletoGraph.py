import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data from the LaTeX table
data = {
    'Method': ['Transformer', 'Transformer', 'MissForest', 'Mtsdi', 'SKLearn', 'MLP'],
    'Dataset': ['Filled SMP Data', 'Original SMP Data', 'Original SMP Data', 'Original SMP Data', 'Original SMP Data', 'Filled SMP Data'],
    'RMSE': [0.00876, 0.0112, 0.0119, 0.0151, 0.0176, 0.0256],
    'MAE': [0.00502, 0.00667, 0.00856, 0.00748, 0.0126, 0.0219],
    'MAPE': [0.0156, 0.0207, 0.0269, 0.0226, 0.0405, 0.0710]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a new column combining Method and Dataset for x-axis labels
df['Method-Dataset'] = df['Method'] + '-' + df['Dataset']

# Melt the DataFrame to long format for easier plotting
df_melted = df.melt(id_vars=['Method-Dataset'], value_vars=['RMSE', 'MAE', 'MAPE'], var_name='Metric', value_name='Value')

# Set the style for the plot
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(12, 8))

# Create a barplot
sns.barplot(x='Method-Dataset', y='Value', hue='Metric', data=df_melted, palette="Set2")

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add titles and labels
plt.title('Error Metrics for Different Methods and Datasets', fontsize=16)
plt.xlabel('Method-Dataset', fontsize=14)
plt.ylabel('Value', fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()
