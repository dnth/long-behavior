import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

result = np.loadtxt("AccuVsBehaviorLength.txt")
fig, ax = plt.subplots()
plt.plot(result[:19], linewidth=2)
plt.ylim([60,105])
plt.grid(True)
plt.title("Recognition rate vs Sequence Length")

loc = plticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
loc = plticker.MultipleLocator(base=10)
ax.yaxis.set_major_locator(loc)
ax.set_xticklabels(np.arange(100,2400,100), rotation=90)
ax.tick_params(axis='x', labelsize=9)
plt.xlabel("Sequence Length, L")
plt.ylabel("Recognition Rate (%)")
plt.show()

