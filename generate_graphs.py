import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure the output directory exists
output_dir = 'paper_outputs'
os.makedirs(output_dir, exist_ok=True)

# Load existing CSV files from paper_outputs folder
df_pred = pd.read_csv(os.path.join(output_dir, 'table3_prediction.csv'))
df_dataset = pd.read_csv(os.path.join(output_dir, 'dataset_description.csv'))

# --- GRAPH 1: Model Error Comparison ---
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
x = np.arange(len(df_pred['Model']))
width = 0.35

rects1 = ax.bar(x - width/2, df_pred['MAE (min)'], width, label='MAE (min)', color='#1f77b4')
rects2 = ax.bar(x + width/2, df_pred['RMSE (min)'], width, label='RMSE (min)', color='#ff7f0e')

ax.set_ylabel('Error in Minutes (Lower is Better)')
ax.set_title('Wait-Time Prediction Error Across Models', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_pred['Model'], rotation=15)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig1_model_error_comparison.png'))
plt.close()

# --- GRAPH 2: Latency vs Error ---
df_filtered = df_pred[df_pred['Model'] != 'Naive Baseline']

plt.figure(figsize=(9, 5), dpi=300)
plt.scatter(df_filtered['Pred Time (s)'], df_filtered['MAE (min)'], color='teal', s=120)

for _, row in df_filtered.iterrows():
    plt.annotate(row['Model'], (row['Pred Time (s)'], row['MAE (min)']),
                 textcoords="offset points", xytext=(5,5), ha='left')

plt.xlabel('Inference Latency per Sequence (seconds)')
plt.ylabel('Mean Absolute Error (minutes)')
plt.title('Trade-off Between Speed and Prediction Accuracy', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig2_latency_vs_error.png'))
plt.close()

# --- GRAPH 3: Dataset Metrics ---
df_dataset['Base_Seq'] = df_dataset['Sequence'].apply(lambda x: x.split('-')[0] + '-' + x.split('-')[1])
df_grouped = df_dataset.groupby('Base_Seq').mean(numeric_only=True).reset_index()

x = np.arange(len(df_grouped['Base_Seq']))
width = 0.25

fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
ax1.bar(x - width, df_grouped['Max Queue Size'], width, label='Max Queue Size', color='#2ca02c')
ax1.bar(x, df_grouped['Mean Queue Size'], width, label='Mean Queue Size', color='#1f77b4')

ax2 = ax1.twinx()
ax2.bar(x + width, df_grouped['Avg Track (sec)'], width, label='Avg Dwell Time (s)', color='#d62728', alpha=0.8)

ax1.set_xlabel('MOT17 Benchmark Video Sequences')
ax1.set_ylabel('Queue Size (Persons)')
ax2.set_ylabel('Average Dwell Time (Seconds)')
ax1.set_title('MOT17 Queue Dynamics Across Sequences', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df_grouped['Base_Seq'])

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3_mot17_dataset_metrics.png'))
plt.close()

print("All figures saved directly into the 'paper_outputs' directory!")