'''
Created on 13 dic 2021

@author: emilio
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st


def  directMethod(stoich_matrix, propensity_fcn, tspan, x0, params=None, output_fcn=None, MAX_OUTPUT_LENGTH=None):
    
    """ Implementation of the Direct Method variant of the Gillespie algorithm. Python porting of https://github.com/nvictus/Gillespie

    :param timestamp: formatted date to display
    :type timestamp: str or unicode
    :param priority: priority number
    :type priority: str or unicode
    :param priority_name: priority name
    :type priority_name: str or unicode
    :param message: message to display
    :type message: str or unicode
    :returns: formatted string
    :rtype: str or unicode
    """
    
    
    
    if MAX_OUTPUT_LENGTH is None:
        MAX_OUTPUT_LENGTH = 1000000;
    if output_fcn is None:
        output_fcn = [];
    if params is None: 
        params = [];
    
    # Initialize
    num_rxns = stoich_matrix.shape[0]
    num_species = stoich_matrix.shape[1]
    T = np.zeros([MAX_OUTPUT_LENGTH, 1])
    X = np.zeros([MAX_OUTPUT_LENGTH, num_species])
    T[0] = tspan[0]
    X[0,:] = x0;
    rxn_count = 0;
    
    # MAIN LOOP
    while T[rxn_count] < tspan[1]:        
        # Calculate reaction propensities
        a = propensity_fcn(X[rxn_count,:], params);
        
        
        
        #Sample earliest time-to-fire (tau)
        a0 = np.sum(a);
        
        r = np.random.rand(1,2);
        
        tau = -np.log(r[0,0])/a0; #(1/a0)*log(1/r(1));
        
        
        # Sample identity of earliest reaction channel to fire (mu)
        mu=0; s=a[0]; r0=r[0,1]*a0;
        while s < r0:
            mu = mu + 1;
            s = s + a[mu];
            
        if rxn_count + 1 > MAX_OUTPUT_LENGTH:
            t = T[0:rxn_count]
            x = X[0:rxn_count,:];
            raise ValueError("SSA:ExceededCapacity",
                             "Number of reaction events exceeded the number pre-allocated. Simulation terminated prematurely.");
            
        #Update time and carry out reaction mu
        T[rxn_count+1]   = T[rxn_count]   + tau;
        X[rxn_count+1,:] = X[rxn_count,:] + stoich_matrix[mu,:];    
        rxn_count = rxn_count + 1;
        
        # if len(output_fcn)>0:
        #     stop_signal = feval(output_fcn, T(rxn_count), X(rxn_count,:)');
        #     if stop_signal
        #         t = T(1:rxn_count);
        #         x = X(1:rxn_count,:);
        #         warning('SSA:TerminalEvent',...
        #                 'Simulation was terminated by OutputFcn.');
        #         return;
        #     end 
        # end
    
    
    t = T[0:rxn_count+1];
    x = X[0:rxn_count+1,:];
    
    if t[-1] > tspan[1]:
        t[-1] = tspan[1]
        x[-1,:] = X[rxn_count-1,:]
    
    
    return [t,x]
        
if __name__ == '__main__':
    sm=np.matrix([[-1,1,0],
                  [-1,0,1],
                  [1,-1,0],
                  [1,0,-1]])
    

    
    prop=lambda x,p:np.array([0,0,0,0])
    
    p={"MU":np.matrix([1,2,2]),
       "S":np.matrix([np.infty,1,1]),
       "P":np.matrix([[0,0.5,0.5],
                     [1,0,0],
                     [1,0,0]])}
    
    prop=lambda x,p:[p["P"][0,1]*p["MU"][0,0]*np.minimum(x[0],p["S"][0,0]),
                     p["P"][0,2]*p["MU"][0,0]*np.minimum(x[0],p["S"][0,0]),
                     p["P"][1,0]*p["MU"][0,1]*np.minimum(x[1],p["S"][0,1]),
                     p["P"][2,0]*p["MU"][0,2]*np.minimum(x[2],p["S"][0,2])]

    e=np.infty
    
    #dimensione di un batch
    K=30
    #numrto di batch
    N=30
    B=[]
    dt=0.1
    x0=[100,0,0]
    while(e>0.5*(10**-1)):
        [t,x]=directMethod(stoich_matrix=sm, propensity_fcn=prop, tspan=[0,K*(N+1)*dt], x0=x0, params=p)
        td = pd.Series(index=[pd.Timedelta(value=t[i,0],unit="sec") for i in range(t.shape[0])],data=x[:,0])
        td=td.resample('%.3fS'%(dt)).fillna(method="ffill")
        x0=x[-1,:]
        
        X=td.to_numpy()
        
        if(len(B)>0):
            B=np.vstack((B,np.array([X[K*n:K*(n+1)] for n in range(N+1)])))
        else:
            B=np.array([X[K*n:K*(n+1)] for n in range(N+1)])

        Bm=np.mean(B,axis=1,keepdims=True)
        CI=st.t.interval(0.95, len(Bm[1:])-1, loc=np.mean(Bm[1:]), scale=st.sem(Bm[1:]))
        e=abs(np.mean(Bm[1:])-CI[0])
            
        print(e)
    
    print(CI[0],np.mean(Bm[1:]),CI[1])