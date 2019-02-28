"""
This module contains a gen() function which can be used to generate noisy
sine data based on given parameters, all of which have default values.
The gen() function is automatically run when this module is imported with
the data being stored in the td.td variable.

February 19, 2019
"""

import numpy as np
import random

def gen(s=1, m=0, sd=0.0, freq=2, phase=1.5, num_points=1000, a=0, b=2*np.pi):
    """
    Generates a sine curve with Gaussian noise according to input
    parameters. Defaults are given for all values as follows:

    s = 1             # Seed the generator for reproducibility
    m = 0             # Mean of Gaussian noise
    sd = 0.1          # Std. Dev. of Gaussian noise
    freq = 2          # Frequency of sine trend
    phase = 1.5       # Phase of sine trend
    num_points = 1000 # Number of data points
    a = 0             # Start of input interval
    b = 2*np.pi       # End of input interval
    """

    # Generate noisy sine curve data
    random.seed(s)
    x = list(np.arange(a, b, (b-a)/num_points))
    arg = [freq*i+phase for i in x]
    y = np.sin(arg)
    y = [point+random.gauss(m,sd) for point in y]
    data = [list(point) for point in zip(x,y)]

    return(data)

td = gen()

"""
A plot of the data in td.td can be created using the following code:

import matplotlib.pyplot as plt
x = [point[0] for point in td.td]
y = [point[1] for point in td.td]
plt.plot(x,y)
plt.show()
"""
