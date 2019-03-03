'''
Code implementing Thompson Sampling with Gaussian distribution.

The model of each machine is using Online Machine learning -
the model is improved with each following sample.

In the experiment we start with multiple machines, each returns
a sample from Gaussian distribution (with unknown
mean, but known variance). The goal is to find and then use
only the machine, which returns the highest numbers.
In order to do it, we need to find the parameters of
each machine (parameters of its Gaussian distribution) and
use the machine with the highest mean.


Distribution approximation: Gaussian unknown mean, known variance
Conjugate prior: Normal
Wikipedia: https://en.wikipedia.org/wiki/Normal_distribution

author: kuba.rachwalski@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



class Machine:
    def __init__(self,mu,tau):
        
        #Real parameters
        self.mu = mu
        self.tau = tau
        
        # Make a very simple prior
        self.mu_model = 0
        self.tau_model = 0.00001
        
        #Additional supporting parameters
        self.starting_mutau = self.mu_model * self.tau_model
        self.suma = 0
        
    def useMachine(self):
        #use the real machine (that we try to model)
        draw = np.random.normal(self.mu, np.sqrt(1/self.tau))        
        return draw
   
    def sampleParameters(self):
        #returns sample of the mean (from gaussian distribution)
        #that is used in the machine model
        return np.random.normal(self.mu_model, np.sqrt(1/self.tau_model))
          
    def sampleFromOurModelOfMachine(self):
        #returns sample from our model of machine
        mu = self.sampleParameters()
        tau = self.tau
        gaus_number = np.random.normal(mu,np.sqrt(1/tau))
        return gaus_number
       
    def updateOurModelOfMachine(self,MachineReturned):        
        #Update the parameters of modeled gaussian distribution.
        self.suma += MachineReturned
        self.tau_model += self.tau
        self.mu_model = (self.tau * self.suma + self.starting_mutau)/(self.tau_model)
            
    #Print PDF of the assesed mu parameter    
    def printMu(self,max = 200000):
        x = np.linspace(-2,3,max)
        y = norm.pdf(x,self.mu_model,np.sqrt(1/self.tau_model))
        print("Assessed mu: "+str(self.mu_model))
        print("Precission of assessment:"+str((self.tau_model)))
        plt.plot(x,y,label="norm "+str(self.mu))
        plt.legend()
        plt.ylim((0,10))
        plt.title("Assessing parameter mu of Gaussian distribution")
     
    #Print PDF of the modeled machine
    def printModel(self,max= 10):
        x = np.linspace(-2,max,200000)
        mu = self.sampleParameters()
        tau = self.tau
        print("Assessed mu: "+str(mu))
        print("Known Precission:"+str(tau))
        y = norm.pdf(x,mu,np.sqrt(1/tau))
        plt.plot(x,y,label="norm "+str(self.mu))
        plt.legend()
        plt.ylim((0,10))
        plt.title("Models of Machines- Gaussian distribution")
        

        
        
def experiment(parameters,N):
    #Run experiment of modeling machines described by "parameters"
    #Experiment is repeated "N" times
    
    #To save all machines
    machines = []
    
    #Create machine for each set of parameters
    for p in parameters:
        (mn, ta) = p
        machines.append(Machine(mn,ta))
    
    #We will save all the results of using the best machine (according to our models)
    results_DB = np.empty(N)
    
    #execute N experiments
    for i in range(N):
        best_machine = None
        sample = 0
        
        #choosing the best machine (max of random sample of mu assessment)
        best_machine = np.argmax([single_machine.sampleParameters() for single_machine in machines])
        #Get real sample of the best machine
        sample = machines[best_machine].useMachine()
        #Update the model of this machine based on the sample
        machines[best_machine].updateOurModelOfMachine(sample)        
        #And save the sample to results_DB
        results_DB[i] = sample
        
    #print results
    
    print("Assessing parameter mu of Gaussian distribution")
    plt.figure()
    for single_machine in machines:
        single_machine.printMu()
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
    t1 = 1
    t2 = 1
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
