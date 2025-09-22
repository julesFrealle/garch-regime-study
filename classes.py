import matplotlib.pyplot as plt
from scipy.stats import t, laplace, uniform
from utils import *
from math import exp

class GARCH:
    def __init__(self, alpha, beta, omega, eta='gaussian', eta_params=None):
        """
        Initialise les paramètres du modèle GARCH.

        Parameters:
        alpha (list): Liste des coefficients alpha.
        beta (list): Liste des coefficients beta.
        omega (float): Terme constant.
        eta (str): Type de bruit pour le processus ('gaussian', 'student', 'laplace', 'uniform').
        eta_params (dict): Paramètres additionnels pour la distribution de bruit (ex : {'df': 5} pour 'student').
        """
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.eta = eta
        self.eta_params = eta_params or {}
        self.results_df = None  # DataFrame pour stocker les résultats

    def recursion(self, vol, eps, t):
        """
        Calcule la volatilité conditionnelle au temps t+1 en fonction des valeurs passées.

        Parameters:
        vol (list): Liste de la volatilité jusqu'à l'instant t.
        eps (list): Liste des perturbations jusqu'à l'instant t.
        t (int): Instant de temps.

        Returns:
        float: Nouvelle volatilité au temps t+1.
        """
        q = len(self.alpha)
        p = len(self.beta)
        new_vol = self.omega
        for i in range(min(q, t + 1)):
            new_vol += self.alpha[i] * eps[t - i]**2
        for j in range(min(p, t + 1)):
            new_vol += self.beta[j] * vol[t - j]
        return new_vol

    def simulate(self, length=1000):
        """
        Simule une trajectoire de processus GARCH.

        Parameters:
        length (int): Longueur de la série temporelle.

        Returns:
        None. Les résultats sont stockés dans self.results_df sous forme de DataFrame.
        """

        # Génération du bruit selon la distribution choisie
        if self.eta == 'gaussian':
            eta = np.random.randn(length)
        elif self.eta == 'student':
            df = self.eta_params.get('df', 5)
            eta = t.rvs(df, size=length)
        elif self.eta == 'laplace':
            loc = self.eta_params.get('loc', 0)
            scale = self.eta_params.get('scale', 1)
            eta = laplace.rvs(loc, scale, size=length)
        elif self.eta == 'uniform':
            low = self.eta_params.get('low', -1)
            high = self.eta_params.get('high', 1)
            eta = uniform.rvs(low, high - low, size=length)
        else:
            raise ValueError("Bruit non supporté. Choisissez parmi 'gaussian', 'student', 'laplace', 'uniform'.")

        vol = [self.omega] + [0] * (length-1)
        eps = [vol[0]*eta[0]] + [0] * (length-1)


        # Simulation du processus
        for t in range(length - 1):
            vol[t + 1] = self.recursion(vol, eps, t)
            eps[t + 1] = sqrt(vol[t + 1]) * eta[t + 1]

        sigma = np.sqrt(vol)

        # Stockage des résultats dans un DataFrame
        self.results_df = pd.DataFrame({
            'vol': vol,
            'sigma': sigma,
            'eps': eps
        })

    def plot(self, plot=['vol', 'sigma', 'eps']):
        """
        Affiche les séries temporelles spécifiées dans le paramètre plot.

        Parameters:
        plot (list): Liste des noms de séries à tracer. Options possibles : 'vol', 'sigma', 'eps'.
        """
        if self.results_df is None:
            raise ValueError("Vous devez d'abord exécuter la simulation avec la méthode `simulate`.")

        # Vérifie que les séries demandées existent
        for p in plot:
            if p not in self.results_df.columns:
                raise ValueError(f"La série '{p}' n'est pas disponible. Options possibles : 'vol', 'sigma', 'eps'.")

        # Tracé des séries sélectionnées
        self.results_df[plot].plot(subplots=True, layout=(len(plot), 1), figsize=(10, 6), title="Simulation du processus GARCH")
        plt.tight_layout()
        plt.show()
        
#    def compute_log_likelihood(params)
        
    def calibrate(self,p,q,eps=None):
        if eps is None:
            eps = self.eps
        if q is None:
            q = len(self.alpha)
        if p is None:
            p = len(self.beta)
        
        
class HMM:
    def __init__(self,P,omega,eta='gaussian',eta_params=None,initial_state=0):
        self.omega = np.array(omega)
        self.P = P
        self.initial_state = initial_state
        self.eta = eta
        self.eta_params = eta_params
        self.space_size = len(P)
        
    
    def simulate(self,length=1000):
        if type(self.initial_state) is int:
            self.states = [self.initial_state] 
        elif type(self.initial_state) is list:
            self.states = np.random.choice(d,p=self.initial_state)
        else:
            raise TypeError('Mauvais format pour l\'état initial, initial_state doit être une liste ou un entier')
        for i in range(length-1):
            self.states.append(np.random.choice(self.space_size,p=self.P[self.states[-1]]))
        
    
                # Génération du bruit selon la distribution choisie
        if self.eta == 'gaussian':
            eta = np.random.randn(length)
        elif self.eta == 'student':
            df = self.eta_params.get('df', 5)
            eta = t.rvs(df, size=length)
        elif self.eta == 'laplace':
            loc = self.eta_params.get('loc', 0)
            scale = self.eta_params.get('scale', 1)
            eta = laplace.rvs(loc, scale, size=length)
        elif self.eta == 'uniform':
            low = self.eta_params.get('low', -1)
            high = self.eta_params.get('high', 1)
            eta = uniform.rvs(low, high - low, size=length)
        self.eta_ = eta
        
        self.vol = self.omega[self.states]
        self.sigma = np.sqrt(self.vol)
        self.eps = eta*self.sigma       
        self.results_df = pd.DataFrame({
            'vol': self.vol,
            'sigma': self.sigma,
            'eps': self.eps,
            'eps2': self.eps**2,
            'states': self.states
        })

    def plot(self, plot=['vol', 'sigma', 'eps']):
        """
        Affiche les séries temporelles spécifiées dans le paramètre plot.

        Parameters:
        plot (list): Liste des noms de séries à tracer. Options possibles : 'vol', 'sigma', 'eps'.
        """
        if self.results_df is None:
            raise ValueError("Vous devez d'abord exécuter la simulation avec la méthode `simulate`.")

        # Vérifie que les séries demandées existent
        for p in plot:
            if p not in self.results_df.columns:
                raise ValueError(f"La série '{p}' n'est pas disponible. Options possibles : 'vol', 'sigma', 'eps'.")

        # Tracé des séries sélectionnées
        self.results_df[plot].plot(subplots=True, layout=(len(plot), 1), figsize=(10, 6), title="Simulation du processus HMM")
        plt.tight_layout()
        plt.show()

class MS_ARCH:
    def __init__(self, true_params, eta='gaussian', eta_params=None, initial_state=0):
        """
        Modèle Markov Switching-ARCH.

        P : Matrice de transition de Markov
        omega : Liste des variances inconditionnelles par état
        alpha : Liste des coefficients ARCH pour chaque lag, de la forme alpha[lag][state]
        eta : Type de bruit utilisé (par défaut 'gaussian')
        eta_params : Dictionnaire contenant les paramètres spécifiques pour les distributions non gaussiennes
        initial_state : État initial ou distribution initiale des probabilités
        """
        
        self.P = np.array(true_params['P'])
        self.omega = np.array(true_params['omega'])
        self.alpha = np.array(true_params.get('alpha',None))  # alpha de taille (nb_lags, nb_states)
        self.initial_state = initial_state
        self.eta = eta
        self.eta_params = eta_params
        self.space_size = len(self.P)
        self.exogen_additive = true_params.get('exogen_additive',None)
        self.mu = np.array(true_params.get('mu',None))
        self.exogen_transition = true_params.get('exogen_transition',None)
        self.beta = true_params.get('beta',np.zeros((self.space_size,self.space_size)))
        self.transition_f = true_params.get('transition_f','softmax')
        if 'seed' in true_params:
            np.random.seed(true_params['seed'])
        
    def simulate(self, length=1000):
        # Initialisation de la chaîne de Markov
        if type(self.initial_state) is int:
            self.states = [self.initial_state]
        elif type(self.initial_state) is list:
            self.states = [np.random.choice(self.space_size, p=self.initial_state)]
        else:
            raise TypeError('Mauvais format pour l\'état initial, initial_state doit être un entier ou une liste')

        if self.exogen_transition is None:
            for i in range(length - 1):
                self.states.append(np.random.choice(self.space_size, p = self.P[self.states[-1]]))
        else:
            for i in range(length - 1):
                new_proba = get_transition_matrix(self.P,self.exogen_transition[i],self.beta,kind=self.transition_f)
                self.states.append(np.random.choice(self.space_size, p = new_proba[self.states[-1]]))

        # Génération du bruit
        if self.eta == 'gaussian':
            eta = np.random.randn(length)
        elif self.eta == 'student':
            df = self.eta_params.get('df', 5)
            eta = t.rvs(df, size=length)
        elif self.eta == 'laplace':
            loc = self.eta_params.get('loc', 0)
            scale = self.eta_params.get('scale', 1)
            eta = laplace.rvs(loc, scale, size=length)
        elif self.eta == 'uniform':
            low = self.eta_params.get('low', -1)
            high = self.eta_params.get('high', 1)
            eta = uniform.rvs(low, high - low, size=length)
        self.eta_ = eta

        # Initialisation de la volatilité et des chocs
        self.sigma = np.zeros(length)
        self.eps = np.zeros(length)
        self.vol = np.zeros(length)
        
        # Simulation de la série ARCH
        for t in range(length):
            state = self.states[t]
            vol_t = self.omega[state]
            # Calcul de la volatilité pour chaque état (modèle ARCH)
            if t > 0:
                q = len(self.alpha)
                for j in range(min(t, q)):
                    vol_t += self.alpha[j][state] * (self.eps[t - 1 - j] ** 2)
                if self.exogen_additive is not None:
                    if self.mu.ndim ==1:
                        vol_t += self.exogen_additive[t-1]@self.mu
                    elif self.mu.ndim ==2:
                        vol_t += self.exogen_additive[t-1]@self.mu[:,state]
                    else:
                        raise TypeError('Mauvais format pour mu, mu doit être de dimension 1 ou 2')
                        
                    
            self.vol[t] = vol_t
            self.sigma[t] = np.sqrt(vol_t)
            self.eps[t] = eta[t] * self.sigma[t]
        # Stockage des résultats
        self.results_df = pd.DataFrame({
            'vol': self.vol,
            'sigma': self.sigma,
            'eps': self.eps,
            'eps2': self.eps ** 2,
            'states': self.states
        })

    def plot(self, plot=['vol', 'sigma', 'eps']):
        """
        Affiche les séries temporelles spécifiées dans le paramètre plot.

        Parameters:
        plot (list): Liste des noms de séries à tracer. Options possibles : 'vol', 'sigma', 'eps'.
        """
        if self.results_df is None:
            raise ValueError("Vous devez d'abord exécuter la simulation avec la méthode `simulate`.")

        # Vérifie que les séries demandées existent
        for p in plot:
            if p not in self.results_df.columns:
                raise ValueError(f"La série '{p}' n'est pas disponible. Options possibles : 'vol', 'sigma', 'eps'.")

        # Tracé des séries sélectionnées
        self.results_df[plot].plot(subplots=True, layout=(len(plot), 1), figsize=(10, 6), title="Simulation du processus MS-ARCH")
        plt.tight_layout()
        plt.show()
        
        
def AR(phi,length=1000,x0=1):
    X = np.zeros((length))
    X[0] = x0
    U = np.random.randn(length-1)
    arch_order = len(phi)
    for i in range(length-1):
        for lag in range(min(arch_order,i+1)):
            X[i+1] += X[i-lag]*phi[lag]
        X[i+1] += U[i]
    return np.array(X)