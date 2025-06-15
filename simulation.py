import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0, mu, sigma, T, dt):
    N = int(T/dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return t, S

# Parameters
S0 = 100
mu = 0.1
sigma = 0.2
T = 1
dt = 0.01

t, S = simulate_gbm(S0, mu, sigma, T, dt)

plt.plot(t, S)
plt.title("Stock Price Simulation using GBM")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.grid(True)
plt.savefig("simulation.png")
plt.show()
