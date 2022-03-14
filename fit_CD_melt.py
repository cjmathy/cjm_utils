import sys
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x, m, Tm, mf, mu, yf, yu):
    y = ((yf + mf*x) + (yu + mu*x)*np.exp(m*(1/Tm - 1/x))) / (1 + np.exp(m*(1/Tm - 1/x)))
    return (y)

def fit_sigmoid(x, y):

    # initial guesses in order: m, Tm, mf, mu, yf, yu
    p0 = [0.4, 55, 0, 0, -14, -8]

    popt, pcov = sp.optimize.curve_fit(
        f=sigmoid,
        xdata=x,
        ydata=y,
        p0=p0,
        method='lm')
    return popt

def estimate_derivative(x, y, w):
    'windowsize w'
    'calculate an estimated derivate for y, then the max of yprime is a good first estimate for your Tm'
    
    y_prime = np.zeros(len(y))
    for i in list(range(w, len(y)-w)):
      y_prime[i] = (y[i+w]-y[i-w])/(x[i+w]-x[i])
    return(y_prime)


datafile = 'example_data/CD_melt.txt'

df = pd.read_csv(datafile, delimiter = '\t', names=['temperature','CD','HV'], skiprows=19)


x = df.temperature
y = df.CD

params = fit_sigmoid(x, y)
ypred = sigmoid(x, *params)

plt.plot(x, y)
plt.plot(x, ypred)
