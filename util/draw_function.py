import matplotlib.pyplot as plt
import numpy as np


def alpha_bar(time_step):
    return np.cos((time_step + 0.008) / 1.008 * np.pi / 2) ** 2

# Define your 1D function
def my_function(x):
    # return np.power(1 - x, 0.5)
    # return np.power((1-x)*x, 1.0)
    # return 1 / (np.sqrt(2 * x * (1 - x)))
    # return 1.4142 * (2 * (x - 1) * np.sqrt(x) + np.power((1 - x), 2.0) / (2.0 * np.sqrt(x)))
    # return np.power((x), 4.5)
    # return np.power((x+0.1), -1) / 10
    # return np.cos((1-x) * np.pi / 2) ** 2
    return np.log(x+1)
    # t1 = x
    # t2 = x+0.01
    # return np.clip(1 - alpha_bar(t2) / alpha_bar(t1), 0.0, 0.999) ** 0.5
    # return 1 / (x * (1 - x) + 1e-4)
    # A = 0.0
    # K = 1.0
    # B = 1.0
    # Q = 0.9
    # v = 0.5
    # M = 0.0
    # C = 1.0
    # return A + (K-A) / np.power(C+Q*np.exp(-B*(5*2*(x-0.5)-M)), 1/v)


# Generate x values
x_values = np.linspace(0.00, 0.99, 100000)  # Adjust the range and number of points as needed

# Compute corresponding y values
y_values = my_function(x_values)

# Plot the function
plt.plot(x_values, y_values, label='y = sin(x)')
plt.title('Plot of y = sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()