import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee

np.random.seed(0)

filename = "tutorial.h5"
sampler = emcee.backends.HDFBackend(filename)
ndim=3

labels = ["amp", "period", "d"]
def plot_chain(sampler, burn=0, thin=1, title=""):
    samples = sampler.get_chain(discard=burn, thin=thin)
    # Plotting the sampling chain for each walker 
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    axes[0].set_title(title)

plot_chain(sampler, title="All Samples")

plt.show()