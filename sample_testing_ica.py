# Import the random module.
import random

# Import the numpy module.
import numpy as np

# Import the matplotlib.pyplot module.
import matplotlib.pyplot as plt

# Import the signal module from scipy.
from scipy import signal

# Import the FastICA and PCA classes from sklearn.decomposition.
from sklearn.decomposition import FastICA, PCA

# Import the make_interp_spline and CubicSpline classes from scipy.interpolate.
from scipy.interpolate import make_interp_spline, CubicSpline

# Specifies the total number of cycles or time interval
time_interval = 10

def generate_signal(num_cycle, rate):
    signals = []  # Use a list to store the signals

    for i in range(0, num_cycle, rate):  # Iterate over the cycles with the given rate
        curr_bit = np.random.randint(low=0, high=2)  # Generate a random bit (0 or 1)

        if i + rate > num_cycle:
            repeat_time = num_cycle - i  # Calculate the remaining cycles if less than the given rate
        else:
            repeat_time = rate  # Use the given rate if enough cycles are remaining

        for _ in range(repeat_time):  # Repeat the current bit for the calculated number of cycles
            signals.append(curr_bit)  # Append the current bit to the signals list

    return np.array(signals)  # Convert the list to a NumPy array before returning

# Generate the jammer signal using the generate_signal function and specified time interval
jammer_rate = np.random.randint(low=2, high=1001)
jammer_signal = generate_signal(time_interval, jammer_rate)

# Generate the sender signal using the generate_signal function and specified time interval
sender_rate = np.random.randint(low=2, high=1001)
sender_signal = generate_signal(time_interval, sender_rate)

# Combine the jammer signal and sender signal to create the mixed signal
mixed_signal = jammer_signal + sender_signal

S = np.c_[mixed_signal, mixed_signal]
A = np.array([[1, 1], [0.5, 2]])
X = np. dot(S, A.T)
ica = FastICA()
S_ = ica.fit(X).transform(X)
A_ =  ica.mixing_
np.allclose(X, np.dot(S_, A_.T))

processed_signal = filter(sender_signal, jammer_signal)
# processed_signal = S_[:, 0]
# processed_signal += 1

print(sender_signal)
# print(processed_signal)