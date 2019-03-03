'''
Code implementing Thompson Sampling with Gaussian distribution.

The model of each machine is using Online Machine learning -
the model is improved with each following sample.

In the experiment we start with multiple machines, each returns
a sample from Gaussian distribution (with unknown
mean, and variance). The goal is to find and then use
only the machine, which returns the highest numbers.
In order to do it, we need to find the parameters of
each machine (parameters of its Gaussian distribution) and
use the machine with the highest mean.


Distribution approximation: Gaussian unknown mean and variance
Conjugate prior: Normal Gamma
Wikipedia: http://en.wikipedia.org/wiki/Normal-gamma_distribution

author: kuba.rachwalski@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma,norm

def sn_calculate(old_mean,new_mean,number):
    #function used to iteratively calculate new s, given:
    # - old mean (without current number)
    # - new mean(with current number)
    # - new sample

    #in order to get variance, divide sn by n
    
    sn = (number-new_mean)*(number-old_mean)
    return sn

class Machine:
    def __init__(self,mu,tau):
        
        #Real parameters
        self.mu = mu
        self.tau = tau
        
        # Make a very simple prior
        #Prior parameters:
        # mu - 1x1 - prior mean
        # n_mu - 1x1 - number of observations of mean
        # tau - 1x1 - prior precision (1 / variance)
        # n_tau - 1x1 - number of observations of tau

        mu = 0.
        n_mu = 1.
        tau = 1.
        n_tau = 1.


        #Prior parameters after conversion:
        # mu0 - prior mean
        # lambda0 - pseudo observations for prior mean
        # alpha0 - inverse gamma shape
        # beta0 - inverse gamma scale

        self.mu0 = mu
        self.lambda0 = n_mu
        self.alpha0 = n_tau * 0.5
        self.beta0 = ((0.5 * n_tau) / tau)

        #Parameters of posterior(initially)
        self.model_mu = self.mu0
        self.model_lambda = self.lambda0
        self.model_alpha = self.alpha0
        self.model_beta = self.beta0

        #additional parameters
        self.sn = 0
        self.n = 0
        self.mean = 0

        
    def useMachine(self):
        #use the real machine (that we try to model)
        draw = np.random.normal(self.mu, np.sqrt(1/self.tau))
        return draw
        
    def sampleParameters_tau(self):
        #returns sample of the precission (from gamma distribution)
        #that is used in the model of the machine
        tau =  np.random.gamma(shape=self.model_alpha, scale=(1. / self.model_beta))
        return tau
    
    def sampleParameters_mean(self,tau):
        #returns sample of the mean (from gaussian distribution)
        #that is used in the model of machine
        var = 1. / (self.model_lambda * tau)
        mean = np.random.normal(loc=self.model_mu, scale=np.sqrt(var))       
        return mean
           
    def sampleFromOurModelOfMachine(self):        
        #returns sample from our model of machine (Gaussian Dist)
        tau = self.sampleParameters_tau()
        mean = self.sampleParameters_mean(tau)
        gaus_number = np.random.normal(mean,np.sqrt(1/tau))
        return gaus_number

    
    def updateOurModelOfMachine(self,MachineReturned):
        #Update the parameters of model
        
        #additional parameters
        old_mean = self.mean
        self.n += 1
        self.mean += (MachineReturned-self.mean)/self.n
        self.sn += sn_calculate(old_mean,self.mean,MachineReturned)
        
        #Parameters of posterior(initially)
        self.model_lambda += 1
        self.model_mu += (MachineReturned-self.model_mu)/self.model_lambda
        self.model_alpha += 0.5
        prior_disc = self.lambda0 * self.n * ((self.mean - self.mu0) ** 2) / self.model_lambda
        self.model_beta = self.beta0 + 0.5 * (self.sn + prior_disc)

        
    def printTau(self,max= 10):
        #Print PDF of the assesed tau parameter  (precission)
        x = np.linspace(-2,max,200000)
        y = gamma.pdf(x, a=self.model_alpha, scale=(1/self.model_beta))
        plt.plot(x,y,label="Tau: "+str(self.tau))
        print("a: "+str(self.model_alpha))
        print("scale: "+str(1/self.model_beta))
        plt.legend()
        plt.ylim((0,10))
        plt.title("Assessing parameter tau of Gaussian distribution")
        
        
    def printMu(self,max = 200000):
        #Print PDF of the assesed parameter mu
        x = np.linspace(-2,3,max)
        tau = self.sampleParameters_tau()
        y = norm.pdf(x,self.model_mu,np.sqrt(1/self.tau))
        print("Assessed mu: "+str(self.model_mu))
        print("Assessed Sampled variance:"+str((1/tau)))
        plt.plot(x,y,label="norm "+str(self.mu))
        plt.legend()
        plt.ylim((0,10))
        plt.title("Assessing parameter mu of Gaussian distribution")
        
        
    def printModel(self,max= 10):
        #Print PDF of the modeled machine
        x = np.linspace(-2,max,200000)
        tau = self.sampleParameters_tau()
        mu = self.sampleParameters_mean(tau)
        print("assessed mean: "+str(mu))
        print("assessed precission " + str(tau))
        y = norm.pdf(x,mu,np.sqrt(1/tau))
        plt.plot(x,y,label="mean "+str(self.mu))
        plt.legend()
        plt.ylim((0,10))
        plt.title("Models of Machines - Gaussian distribution")
        
    def printReal(self,max= 10):
        #Print PDF of the real machine
        x = np.linspace(-2,max,200000)
        tau = self.tau
        mu = self.mu
        print("Real mean: "+str(mu))
        print("Real precission " + str(tau))
        y = norm.pdf(x,mu,np.sqrt(1/tau))
        plt.plot(x,y,label="mean "+str(self.mu))
        plt.legend()
        plt.ylim((0,10))
        plt.title("Models of Machines - Real Values")

def experiment(parameters,N):
    #Run experiment of modeling machines described by "parameters"
    #Experiment is repeated "N" times
    
    #To save all machines
    machines = []
    
    #Create machine for each set of parameters
    for p in parameters:
        (mn, ta) = p
        machines.append(Machine(mn,ta))
        
    #Place to save all the results of using the best machine (according to our models)
    results_DB = np.empty(N)

    #execute N experiments
    for i in range(N):
        best_machine = None
        sample = 0
        
        #choosing the best machine (according to the highest mean)
        best_machine = np.argmax([single_machine.sampleParameters_mean(single_machine.sampleParameters_tau()) for single_machine in machines])
        #Get real sample of the best machine
        sample = machines[best_machine].useMachine()
        #Update the model of this machine based on the sample
        machines[best_machine].updateOurModelOfMachine(sample)        
        #And save the sample to results_DB
        results_DB[i] = sample
        
    #print results
    print("Tau assessment")
    plt.figure()
    for single_machine in machines:
        single_machine.printTau()
        print("======")
    
    print("")
    plt.figure() 
    print("Mean assessment")
    for single_machine in machines:
        single_machine.printMu()
        print("======")
    
    print("")
    plt.figure()
    print("Full model calcualtion")
    for single_machine in machines:
        single_machine.printModel()
        print("======")
    
    print("")    
    plt.figure() 
    print("Real values")
    for single_machine in machines:
        single_machine.printReal()
        print("======")
    
    #Return DB with all results   
    return results_DB

if __name__ == "__main__":
    
    #number of experiments
    N = 10000
    
    #Parameters of machines
    #Mean
    m1 = 0.5
    m2 = 2
    m3 = 1
    #Precission
    t1 = 4
    t2 = 2
    t3 = 1

    #Run the experiemtns
    results = experiment([(m1,t1),(m2,t2),(m3,t3)],N)
    
    
    #Draw the efficiency of the Thompson Algorithm (moving average)
    cumulative_average = np.cumsum(results) / (np.arange(N) + 1)
    
    #draw results
    plt.figure()
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.title("Performance of the Thompson Algorithm")
    plt.show()
