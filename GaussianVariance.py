'''
Code implementing Thompson Sampling with Gaussian distribution.

The model of each machine is using Online Machine learning -
the model is improved with each following sample.

In the experiment we start with multiple machines, each returns
a sample from Gaussian distribution (with known
mean, but unknown variance). The goal is to find and then use
only the machine, which returns the highest numbers.
In order to do it, we need to find the parameters of
each machine (parameters of its Gaussian distribution) and
use the machine with the highest mean.


This case is not really valid for the problem because 
mean is known and the posterior function is distribution
of variance. In our problem we need to maximize mean (known),
not variance. 


Distribution approximation: Gaussian known mean, unknown variance
Conjugate prior: Gamma
Wikipedia: https://en.wikipedia.org/wiki/Gamma_distribution

author: kuba.rachwalski@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma,norm

class Machine:
    def __init__(self,mu,tau):
        
        #Real parameters
        self.mu = mu
        self.tau = tau
        
        # Make a very simple prior
        self.model_alpha = 1
        self.model_beta = 1
        self.model_mu = 0
        self.model_tau = 0.0001
        
        #Additional supporting parameters
        self.suma = 0
        
    def useMachine(self):
        #use the real machine (that we try to model)
        draw = np.random.normal(self.mu, np.sqrt(1/self.tau))
        return draw
        
    def sampleParameters(self):
        #returns sample of the precission (from gamma distribution)
        #that is used in the machine model
        return np.random.gamma(self.model_alpha , (1/self.model_beta))
    
    def sampleFromOurModelOfMachine(self):
        #returns sample from our model of machine (Gaussian Dist)
        tau = self.sampleParameters()
        gaus_number = np.random.normal(self.mu,np.sqrt(1/tau))
        return gaus_number
    
    def updateOurModelOfMachine(self,MachineReturned):
        #Update the parameters of model (Gamma Distribution)      
        self.model_alpha += 1/2
        self.model_beta += ((MachineReturned - self.mu)**2)/2
        self.suma += MachineReturned        
        
    #Print PDF of the assesed tau parameter  (precission)
    def printTau(self,max= 10):
        x = np.linspace(-2,max,200000)
        y = gamma.pdf(x, a=self.model_alpha, scale=(1/self.model_beta))
        plt.plot(x,y,label="variance "+str(self.tau))
        print("a: "+str(self.model_alpha))
        print("scale: "+str(1/self.model_beta))
        plt.legend()
        plt.ylim((0,10))
        plt.title("Assessing parameter tau of Gaussian distribution")
    
    #Print PDF of the modeled machine
    def printModel(self,max= 10):
        x = np.linspace(-2,max,200000)
        mu = self.mu
        tau = self.sampleParameters()
        print("known mu: "+str(mu))
        print("assessed precission " + str(tau))
        y = norm.pdf(x,mu,np.sqrt(1/tau))
        plt.plot(x,y,label="mean "+str(mu))
        plt.legend()
        plt.ylim((0,10))
        plt.title("Models of Machines- Gaussian distribution")
        
        
        
    def gettau(self):
        return np.random.gamma(self.model_alpha, (1/self.model_beta))
    
    def getmu(self):
        return self.mu
        

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
        
        #choosing the best machine (using known parametr mu)
        best_machine = np.argmax([single_machine.getmu() for single_machine in machines])
        #Get real sample of the best machine
        sample = machines[best_machine].useMachine()
        #Update the model of this machine based on the sample
        machines[best_machine].updateOurModelOfMachine(sample)        
        #And save the sample to results_DB
        results_DB[i] = sample
        
    #print results
    
    print("Assessing parameter tau of Gaussian distribution")
    plt.figure()
    for single_machine in machines:
        single_machine.printTau()
        print("======")
    
    print("")
    print("Assessing Machines Modeling - Gaussian distribution")
    plt.figure() 
    for single_machine in machines:
        single_machine.printModel()
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
