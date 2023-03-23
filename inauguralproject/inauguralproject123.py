
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:
    

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        
        if par.sigma == 0:
            H=np.minimum(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H=((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma)**(par.sigma/(par.sigma-1)))
    

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=True):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        pass    

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        pass

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass




model = HouseholdSpecializationModelClass

# opgave 1)



#define alpha and simga

# define a range of values for alpha and sigma
alpha_vals = [0.25, 0.5, 0.75]
sigma_vals = [0.5, 1.0, 1.5]

# create a grid of alpha and sigma values
alpha_grid, sigma_grid = np.meshgrid(alpha_vals, sigma_vals)

# compute the ratio HF/HM for each combination of alpha and sigma
ratios = []
for alpha, sigma in zip(alpha_grid.ravel(), sigma_grid.ravel()):
    model = HouseholdSpecializationModelClass()
    model.par.alpha = alpha
    model.par.sigma = sigma
    opt = model.solve_discrete(do_print=False)
    ratios.append(opt.HF / opt.HM)

# reshape the ratios into a grid
ratio_grid = np.array(ratios).reshape(alpha_grid.shape)

# create the heatmap plot
plt.imshow(ratio_grid, cmap='coolwarm', origin='lower', 
           extent=[alpha_vals[0], alpha_vals[-1], sigma_vals[0], sigma_vals[-1]])
plt.colorbar(label='HF/HM')
plt.xlabel('alpha')
plt.ylabel('sigma')
plt.title('Ratio of HF to HM')
plt.show()



#opgave 2 


# create a new model object
model = HouseholdSpecializationModelClass()

# set wF values
model.par.wF_vec = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

# solve for each wF value
for i, wF in enumerate(model.par.wF_vec):
    model.par.wF = wF
    opt = model.solve_discrete(do_print=False)
    model.sol.LM_vec[i] = opt.LM
    model.sol.HM_vec[i] = opt.HM
    model.sol.LF_vec[i] = opt.LF
    model.sol.HF_vec[i] = opt.HF

# calculate log ratios
log_HF_HM = np.log(model.sol.HF_vec / model.sol.HM_vec)
log_wF_wM = np.log(model.par.wF_vec / model.par.wM)

# plot log ratios against log wF/wM
plt.plot(log_wF_wM, log_HF_HM, 'o')
plt.xlabel('log(wF/wM)')
plt.ylabel('log(HF/HM)')
plt.show()


#Ogpave 3




model = HouseholdSpecializationModelClass()

# set wF values
model.par.wF_vec = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

# solve for each wF value
for i, wF in enumerate(model.par.wF_vec):
    model.par.wF = wF
    opt = model.solve_discrete(do_print=True)
    model.sol.LM_vec[i] = opt.LM
    model.sol.HM_vec[i] = opt.HM
    model.sol.LF_vec[i] = opt.LF
    model.sol.HF_vec[i] = opt.HF

# calculate log ratios
log_HF_HM = np.log(model.sol.HF_vec / model.sol.HM_vec)
log_wF_wM = np.log(model.par.wF_vec / model.par.wM)


# plot log ratios against log wF/wM
plt.plot(log_wF_wM, log_HF_HM, 'o')
plt.xlabel('log(wF/wM)')
plt.ylabel('log(HF/HM)')
plt.show()



