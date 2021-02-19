import numpy as np
import matplotlib.pyplot as plt

cm = np.load('confusion_matrix.npy')
print(cm)

x = np.arange(cm.shape[0])
y = np.arange(cm.shape[0])

# Plotting:
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="jet")

# Show all ticks and labels:
ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))
ax.set_xticklabels(x)
ax.set_yticklabels(y)

# Colorbar:
cbar = ax.figure.colorbar(im, ax=ax)

# Set ticks position
ax.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)

# Turn spines off and create white grid
for edge, spine in ax.spines.items():
	spine.set_visible(False)

ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

# Print data inside the heatmap-matrix
for i in range(len(x)):
    for j in range(len(y)):
        text = ax.text(j, i, cm[i, j].round(3),
                       ha="center", va="center", color="w")

ax.set_title("Confusion matrix\n", fontweight='bold')
fig.tight_layout()
plt.show()
