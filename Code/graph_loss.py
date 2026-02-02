import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../Results/3_finetuning/loss_history.csv")

plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['loss'], label='Training Loss', color='#1f77b4', linewidth=2)
plt.title('Training Loss Convergence', fontsize=14, fontweight='bold')
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()