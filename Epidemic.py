import numpy as np; import matplotlib.pyplot as plt;
from scipy.stats import bernoulli; import pandas as pd; 
import random; from scipy import stats;
%matplotlib inline


def epidemic(popsize, initial_infections, mean_interactions, mean_prob_inf, mortality_rate,
                    infectious_period, recovery_period, sigma_prob_inf=None,verbose=1):

    """ Parameters: 
        -- popsize describes the population size
        -- initial_infections describes number of initial infections within the population
        -- mean_interactions describes the mean number of interactions a person in the population 
           has each day. This will be the scale parameter for an exponential distribution
        -- mean_prob_inf describes the mean probability of infection for an interaction. Actual            infection probabilities will be drawn from a normal distribution with this as the mean. 
        -- mortality_rate describes the probability of death over the recovery period 
        -- infectious_period describes the number of days a person is infectious for 
        -- recovery_period describes the number of days a person is non-infectious but symptomatic for. 
           This represents the period of time in which a person may die with the disease and if they outlast it they will have recovered. 
        -- sigma_prob_inf descrinbes the sd for the normal distribution from which probabilities of         infection are drawn from and is usually set to 10% of the mean. 
    """

    """ Return:
        -- Firstly returns the daily death count, Ndeath, as a list. Can calculate total deaths implictly from this.
        -- Secondly returns the daily infection count, Ninf, as a list. 
    """
    # Initialise sigma 
    if not sigma_prob_inf:
        sigma_prob_inf = mean_prob_inf/10

    # Initialise data structures 
    Population = pd.DataFrame(np.array([0]*popsize),columns=['State'])
    Population['Infection time'] = np.array([0]*popsize)
    Ndeaths = []
    Ninf = [initial_infections]

    # Initialise time and DONE conditions
    t=0
    DONE=False

    # Convert mortality rate into a daily probability i.e. convert a mortality rate into a 
    # daily probability rate such that they are equivalent. 

    PdeathDaily = 1 - (1-mortality_rate)**(1/recovery_period)

    # Select random initial infections 
    start_infections = [random.sample(list(Population.index),1)[0] for i in range(initial_infections)]
    for i in start_infections:
        Population.iloc[i,0] = 1
        Population.iloc[i,1] = 0

    # Bulk of the program exists in this while loop 

    while DONE==False:

    # Deaths and recoveries
    # This moves people from non-infectious to non-infectious  
        for i in Population[Population['State']==1].index:
            if (t - Population.iloc[i,1]) >= infectious_period:
                Population.iloc[i,0] = 2

        # For non-infectious, symptomatic people this is the mechanism for them dying with certain prob
        # by generating bernoulli random variables
        for i in Population[Population['State']==2].index:
            if bernoulli.rvs(PdeathDaily)== 1:
                Population.iloc[i,0] = -1 
            # If they have been symptomatic for 14 days or more they are set to recovered 
            elif (t-Population.iloc[i,1]) >= recovery_period:
                Population.iloc[i,0] = 3


        # New infections 

        # This generates a number for each infectious person from a exponential distribution which 
        # denotes how many people each person comes into contact with on that day 
        n_int = [round(np.random.exponential(mean_interactions,size=1)[0]) for i in range(len(Population[Population['State']==1]))]
        # Total number of interactions between infectious peoples and rest of the population 
        new_int = int(sum(n_int))
        # Sample with replacement from everyone who is alive to denote who the infectious people come into contact with
        # dead people 
        people_int = random.choices(list(Population[Population['State']!=-1].index),k=new_int)

        # For each person interacted with this describes the transmission mechanism - probabilistic transmission 
        # The probability of transmission is drawn from a normal distribution with some mean and sd 
        new = 0
        for i in people_int:
            if Population.iloc[i,0] == 0:
                # The actual transmission is described by sampling from bernoulli using p~N(mu,sigma)
                brn = bernoulli.rvs(stats.norm.rvs(mean_prob_inf,sigma_prob_inf,size=1))
                # If the bernoulli RV is 1 the person becomes infected
                if brn == 1:
                    Population.iloc[i,0] = 1
                    Population.iloc[i,1] = t
                    new += 1    
            # Anyone who isn't susceptible denoted by a 0 remains unchanged. 
            else:
                pass 

        # Done conditions - when no active infections and no symptomatic people 
        if (len(Population[Population['State']==1]) == 0) & (len(Population[Population['State']==2]) == 0):
            DONE = True

        # Update various counts 
        Ndeaths.append(len(Population[Population['State']==-1]))
        Ninf.append(new)

        # Increase time - i.e. move to next time period 
        t += 1

        if verbose==1:
            print('Time:',t,'; No. dead:',len(Population[Population['State']==-1]),'; Total Infections:',np.cumsum(Ninf[:t])[-1])    
    
    return Ndeaths, np.cumsum(Ninf)




