import numpy as np
import matplotlib.pyplot as plt
import pickle
with open("hilo-converge-results.pkl",'rb') as f:
    results = pickle.load(f)

def a_avg(arrs):
    return np.stack(arrs).mean(axis=0)

def a_std(arrs):
    return np.stack(arrs).std(axis=0)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

n_steps = results["hilo_steps"]
test_every = results["test_every"]
# xs = np.array(list(range(0, n_steps, test_every)))
xs = np.array(list(range(0, n_steps, test_every)) + ([] if (n_steps-1)%test_every==0 else [n_steps-1]))
l_mm, l_mc, l_cm, l_cc = results["losses"]
ax1.plot(xs, a_avg(l_mm), color='b', label="mnist->minst")
ax1.plot(xs, a_avg(l_mc), color='yellow', label="mnist->casia")
ax1.plot(xs, a_avg(l_cm), color='green', label="casia->mnist")
ax1.plot(xs, a_avg(l_cc), color='r', label="casia->casia")
ax1.set_xlabel("Duels")
ax1.set_ylabel("Average Loss")
ax1.set_yscale('log')
ax1.set_yticks([0.05, 0.1, 0.25, 1], labels =[str(i) for i in [0.05, 0.1, 0.25, 1]])
ax1.legend()

ax2.errorbar(xs, a_avg(l_mm), a_std(l_mm), color='b', label="mnist->mnist")
ax2.errorbar(xs, a_avg(l_mc), a_std(l_mc), color='yellow', label="mnist->casia")
ax2.errorbar(xs, a_avg(l_cm), a_std(l_cm), color='green', label="casia->mnist")
ax2.errorbar(xs, a_avg(l_cc), a_std(l_cc), color='r', label="casia->casia")
ax2.set_xlabel("Duels")
ax2.set_ylabel("Loss: average ± standard deviation")
ax2.set_yscale('log')
ax2.set_yticks([0.05, 0.1, 0.25, 1], labels =[str(i) for i in [0.05, 0.1, 0.25, 1]])
ax2.legend()

ax3.plot(xs, a_avg(l_mm), color='b', label="mnist->minst")
ax3.fill_between(xs, a_avg(l_mm)-a_std(l_mm), a_avg(l_mm)+a_std(l_mm), color='b',alpha=0.3)
ax3.plot(xs, a_avg(l_cm), color='green', label="casia->mnist")
ax3.fill_between(xs, a_avg(l_cm)-a_std(l_cm), a_avg(l_cm)+a_std(l_cm), color='green',alpha=0.3)
print(a_avg(l_cm))
print(a_avg(l_cm)-a_std(l_cm))
print(a_avg(l_cm)+a_std(l_cm))
ax3.set_xlabel("Duels")
ax3.set_ylabel("Loss for target mnist")
ax3.set_yscale('log')
ax3.set_yticks([0.05, 0.1, 0.25, 1], labels =[str(i) for i in [0.05, 0.1, 0.25, 1]])
ax3.legend()

ax4.plot(xs, a_avg(l_cc), color='r', label="casia->casia")
ax4.fill_between(xs, a_avg(l_cc)-a_std(l_cc), a_avg(l_cc)+a_std(l_cc), color='r',alpha=0.3)
ax4.plot(xs, a_avg(l_mc), color='yellow', label="mnisa->casia")
ax4.fill_between(xs, a_avg(l_mc)-a_std(l_mc), a_avg(l_mc)+a_std(l_mc), color='yellow',alpha=0.3)
ax4.set_xlabel("Duels")
ax4.set_ylabel("Loss for target casia")
ax4.set_yscale('log')
ax4.set_yticks([0.05, 0.1, 0.25, 1], labels =[str(i) for i in [0.05, 0.1, 0.25, 1]])
ax4.legend()

plt.suptitle("Performing HILO on 5 patients for 50 steps with MNIST and CASIA")
plt.show()