import matplotlib.pyplot as plt
import numpy as np


# Data comes from training logs or project github repo
mf = [0.116, 0.459, 0.663, 0.778]
emcdr = [0.1597, 0.4779, 0.6265, 0.6915]
jaccard = [0.0419, 0.3944, 0.6954, 0.833]
popularity = [0.0487, 0.2866, 0.556, 0.7292]
sota = [0.021, 0.106, 0.212, 0.319]

x = np.array([2, 4, 6, 8])
width = 0.2
plt.bar(x - 3 * width, mf, width=width, label="Matrix Factorization")
plt.bar(x - 2.0 * width, emcdr,  width=width, label="EMCDR")
plt.bar(x - 1 * width, jaccard,  width=width, label="Jaccard")
plt.bar(x, popularity,  width=width, label="Popularity Model")
plt.bar(x + 1 * width, sota,  width=width, label="SOTA")
plt.xticks([1.8, 3.8, 5.8, 7.8], ["hits1", "hits5", "hits10", "hits15"])
plt.xlabel("Metrics")
plt.title("Performance for different approaches", fontsize=14)
plt.legend()
plt.savefig("images/model_compare.jpg", dpi=300, bbox_inches="tight")
plt.show()
