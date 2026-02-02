import matplotlib.pyplot as plt
import numpy as np

# Experimental Data
metrics = ['Assignee Accuracy', 'Action Similarity', 'Deadline Accuracy']
baseline_zero = [1.1, 0.9, 0.0]   # Failed Baseline (Zero-Shot)
baseline_one = [7.4, 11.8, 6.9]   # Baseline with One-Shot Example
fine_tuned = [56.7, 59.3, 35.1]   # Our Fine-Tuned Model

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the Bars
rects1 = ax.bar(x - width, baseline_zero, width, label='Baseline (Zero-Shot)', color='#d3d3d3') # Light Gray
rects2 = ax.bar(x, baseline_one, width, label='Baseline (One-Shot)', color='#87CEEB')       # Sky Blue
rects3 = ax.bar(x + width, fine_tuned, width, label='Fine-Tuned (Ours)', color='#2E8B57')   # Forest Green

# Styling & Layout
ax.set_ylabel('Performance Score (%)')
ax.set_title('Impact of Fine-Tuning on Hebrew Action Extraction')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 70) # Add some headroom for labels

# Add Value Labels above Bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('results_chart.png', dpi=300)
print("Graph saved as results_chart.png")
plt.show()