import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#manually changed indices and scores for each graph
exp_and_scores = {
    'Zero-Shot': [6, 3, 2, 1, 3, 2, 8, 8, 2, 10, 10, 3, 10, 2, 2, 8, 7, 4, 2, 5, 10, 10, 4, 10, 3, 3, 10, 2, 1, 2, 9, 10, 8, 8, 3, 10, 6, 3, 6, 8, 8, 2, 6, 6, 8, 3, 2, 5, 8, 2, 10, 1, 8, 10, 6, 7, 7, 8, 8, 5, 6, 10, 10, 4],
    'Two-Shot': [6, 4, 1, 7, 2, 1, 8, 10, 10, 10, 10, 2, 2, 10, 3, 6, 7, 7, 10, 10, 10, 8, 10, 10, 9, 10, 10, 10, 10, 10, 8, 10, 6, 7, 6, 3, 4, 3, 4, 4, 7, 8, 8, 6, 10, 3, 10, 10, 10, 10, 10, 3, 4, 5, 7, 6, 7, 10, 8, 9, 10, 10, 10, 3],
    'Four-Shot': [5, 8, 1, 2, 3, 2, 10, 10, 2, 10, 6, 2, 1, 2, 4, 10, 7, 10, 10, 8, 10, 10, 9, 10, 10, 8, 10, 10, 10, 10, 7, 10, 6, 4, 3, 10, 4, 3, 7, 4, 7, 2, 8, 4, 8, 4, 10, 10, 10, 6, 10, 1, 6, 4, 9, 4, 3, 7, 8, 10, 10, 10, 10, 3],
    'Six-Shot': [4, 2, 10, 10, 4, 2, 8, 10, 10, 10, 3, 3, 2, 10, 3, 10, 6, 10, 10, 10, 10, 9, 10, 10, 10, 8, 10, 10, 10, 10, 7, 10, 6, 7, 2, 3, 3, 1, 4, 6, 9, 5, 8, 4, 10, 5, 10, 8, 10, 3, 10, 3, 6, 3, 4, 6, 5, 10, 10, 10, 10, 10, 10, 2]
}

data = []
for experiment, scores in exp_and_scores.items():
    score_array = np.array(scores)
    sorted_array = np.sort(score_array)
    for individual_score in sorted_array:
        data.append({'CoT Prompting Style': experiment, 'Correctness Scores': individual_score})

df = pd.DataFrame(data)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
#creates the box plots
ax = sns.boxplot(x='CoT Prompting Style', y='Correctness Scores', data=df, width=0.5, fliersize=0)
#inserting data points
sns.stripplot(x='CoT Prompting Style', y='Correctness Scores', data=df, jitter=0.23, color='black', size=6, alpha=0.7)
for i, (experiment, scores) in enumerate(exp_and_scores.items()):
    q1_label = np.percentile(scores, 25)
    ax.text(i, q1_label, f'Q1: {q1_label:.2f}', ha = 'center', va = 'bottom', color = 'white', fontsize = 12)

plt.ylim(0, 11)
plt.title('Zero, Two, Four, and Six-Shot CoT Prompting Score Distribution for Scenarios of Large Files')
plt.show() 