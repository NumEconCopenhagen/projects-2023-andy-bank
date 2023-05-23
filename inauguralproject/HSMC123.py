from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:
    def __init__(self):
        """ setup model """
        # Create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # Set parameter values
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5
        par.alpha = 0.5
        par.sigma = 1.0
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8, 1.2, 5)
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # Initialize solution variables
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)
        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self, LM, HM, LF, HF):
        """ Calculate utility """
        par = self.par
        sol = self.sol

        # Calculate consumption of market goods
        C = par.wM * LM + par.wF * LF

        # Calculate home production
        if par.sigma == 0:
            H = np.minimum(HM, HF)
        elif par.sigma == 1:
            H = HM**(1 - par.alpha) * HF**par.alpha
        else:
            H = ((1 - par.alpha) * HM**((par.sigma - 1) / par.sigma) +
                 par.alpha * HF**((par.sigma - 1) / par.sigma)**(par.sigma / (par.sigma - 1)))

        # Calculate total consumption utility
        Q = C**par.omega * H**(1 - par.omega)
        utility = np.fmax(Q, 1e-8)**(1 - par.rho) / (1 - par.rho)

        # Calculate disutility of work
        epsilon_ = 1 + 1 / par.epsilon
        TM = LM + HM
        TF = LF + HF
        disutility = par.nu * (TM**epsilon_ / epsilon_ + TF**epsilon_ / epsilon_)

        return utility - disutility

    def solve_discrete(self, do_print=True):
        """ Solve model discretely """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # All possible choices
        x = np.linspace(0, 24, 49)
        LM, HM, LF, HF = np.meshgrid(x, x, x, x)
        LM = LM.ravel()
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # Calculate utility
        u = self.calc_utility(LM, HM, LF, HF)

        # Set to minus infinity if constraint is broken
        I = (LM + HM > 24) | (LF + HF > 24)
        u[I] = -np.inf

        # Find maximizing argument
        j = np.argmax(u)

        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # Print
        if do_print:
            for k, v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self, do_print=False):
        """ Solve model continuously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        def objective(x):
            return self.calc_utility(x[0], x[1], x[2], x[3])

        obj = lambda x: -objective(x)
        constraints = (
            {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 24},
            {'type': 'ineq', 'fun': lambda x: x[2] + x[3] - 24}
        )
        guess = [4] * 4
        bounds = [(0, 24)] * 4

        # Optimizer
        result = optimize.minimize(
            obj, guess, method='SLSQP', bounds=bounds, constraints=constraints
        )

        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]

        return opt

    def solve_wF_vec(self, discrete=False):
        """ Solve model for vector of female wages """
        pass

    def run_regression(self):
        """ Run regression """
        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec / sol.HM_vec)
        A = np.vstack([np.ones(x.size), x]).T
        sol.beta0, sol.beta1 = np.linalg.lstsq(A, y, rcond=None)[0]

    def estimate(self, alpha=None, sigma=None):
        """ Estimate alpha and sigma """
        pass


# Create an instance of the model
model = HouseholdSpecializationModelClass()
