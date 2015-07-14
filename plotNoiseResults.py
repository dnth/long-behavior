import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

lstmAccuracyData = np.loadtxt("lstm/AccuracyData.txt")
rnnAccuracyData = np.loadtxt("rnn/AccuracyData.txt")
mlpAccuracyData = np.loadtxt("mlp/AccuracyData.txt")
tdnn10AccuracyData = np.loadtxt("tdnn/AccuracyData.txt")
tdnn100AccuracyData = np.loadtxt("tdnn-100/AccuracyData.txt")
tdnn200AccuracyData = np.loadtxt("tdnn-200/AccuracyData.txt")

fig, ax = plt.subplots(figsize=(10,10))

plt.plot(lstmAccuracyData, linewidth=2, label="LSTM")
plt.plot(rnnAccuracyData, linewidth=2, label="RNN")
plt.plot(mlpAccuracyData, linewidth=2, label="MLP")
plt.plot(tdnn10AccuracyData, linewidth=2, label="TDNN-10")
plt.plot(tdnn100AccuracyData, linewidth=2, label="TDNN-100")
plt.plot(tdnn200AccuracyData, linewidth=2, label="TDNN-200")
plt.legend()
plt.grid(True)
plt.ylabel("Classification Accuracy (%)")
plt.xlabel(r"$\sigma$")
loc = plticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
loc = plticker.MultipleLocator(base=10)
ax.yaxis.set_major_locator(loc)


labels = np.arange(-0.1,2.1,0.1).tolist()
labels[1] = 'no noise'
ax.set_xticklabels(labels, rotation=60)
ax.tick_params(axis='x', labelsize=9)
plt.show()