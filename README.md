# BayesGaussian
Code implementing Thompson Sampling with Gaussian distribution (Bayesian Machine Learning - AB Testing)

The model of each machine is using Online Machine learning -
the model is improved with each following sample.

In the experiment we start with multiple machines, each returns
a sample from Gaussian distribution. The goal is to find and then use
only the machine, which returns the highest numbers.
In order to do it, we need to find the parameters of
each machine (parameters of its Gaussian distribution) and
use the machine with the highest mean.

Three variations of the problem are presented:
* Distribution approximation: Gaussian unknown mean, known variance (GaussianMean.py)
* Distribution approximation: Gaussian known mean, unknown variance (GaussianVariance.py)
* Distribution approximation: Gaussian unknown mean, unknown variance (GaussianMeanVariance.py)
