import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize, LinearConstraint
from copy import copy
from collections import defaultdict
from utils import *
import numpy as np
        
"""   
CONVENTION : pi commence en [1,0], on a donc besoin d'une longueur n+1 pour aller en pi[n,n] et en pi[1,0], or on va de eps[0] à eps[n-1]. 
On rappelle que pi[n,n] correspond à delta[n-1] et à eps[n-1], puisque pi[1,0] correspond à delta[0] (et à eps[vide]).
Ensuite, pi[t,t] fait intervenir phis[t-1]. Qui correspond à eps[t-1], ça c'est ok. Pareillement, pi[t,t] correspond à delta[t-1], donc à exogen_transition[t-2], c'est logique puisque Delta comment à delta[0],
qui ne dépend pas de X, puis delta[1], qui correspond à pi[2] et à exogen_transition[0]. pour la passe forward, on est sur p_t, donc sur delta[t], donc sur exogen_transition[t-1]. pour la passe backward on commence à
jointpi[n|n-1], on se réfère donc à delta[n-1], donc à exogen_transition[n-2], c'est logique puisque exogen_transition va jusqu'à n-1 (longueur n), et que la dernière observation est useless, ainsi pour
jointpi[n-t+1], on se réfère à delta[n-t] puis à exogen_transition[n-t-1]
        
"""

def MS_hamilton_filter(eps, params, i = None, backward_pass=True):  
    if i is None:
        params = { 0 : copy(params)}
        i = 0
    pi_0 = params[i].get('pi_0')
    P = params[i].get('P')
    
    exogen_transition = params[0].get('exogen_transition',None)
    transition = exogen_transition is not None
    k = exogen_transition.shape[-1] if transition else 1

    n = len(eps)
    d = len(pi_0)    
    # Matrices initialisées
    pi = np.zeros((n+1, n+1, d))  # Probabilités filtrées
    # Initialisation
    pi[1, 0] = pi_0

    # Forward Pass (vectorisé)
    phis = phi_vectorized(eps,params[i]['omega'],params[i].get('alpha',None),params[0].get('exogen_additive',None),params[i].get('mu',None)) ### Fonction @njit
    for t in range(1, n):
        if transition and t>0:
            P_new = get_transition_matrix(copy(P),exogen_transition[t-1],params[i].get('beta',np.zeros((d,d,k))),params[0].get('transition_f','softmax'))
        else:
            P_new = copy(P)
            
        pi[t, t] = pi[t, t-1] * phis[t-1,:]
        pi[t, t] /= pi[t, t].sum()  # Normalisation
        pi[t+1, t] = P_new.T @ pi[t, t]  # Utilisation de P.T pour vectoriser

    # Dernière mise à jour de pi
    pi[n, n] = pi[n, n-1] * phis[-1,:]
    pi[n, n] /= pi[n, n].sum()

    if not backward_pass:
        return pi

    jointpi = np.zeros((n+1, d, d))  # Probabilités conjointes
    # Backward Pass (vectorisé)
    for t in range(1, n+1):
        if transition and t<n:
           P_new = get_transition_matrix(P,exogen_transition[n-t-1],params[i].get('beta',np.zeros((d,d,k))),params[0].get('transition_f','softmax'))
        else:
            P_new = copy(P)
        jointpi[n-t+1] = P_new * np.outer(pi[n-t, n-t], pi[n-t+1, n] / pi[n-t+1, n-t])
        pi[n-t, n] = jointpi[n-t+1].sum(axis=1)
    
    return pi,jointpi



def MS_EM_Pass(pi,jointpi,eps,params,i):#eps, pi, jointpi, omega, alpha, exogen_additive):
    
    params[i+1] = {}
    n, d = pi.shape[1:]  # Extract shape information
    q = len(params[i].get('alpha',[]))  # Number of ARCH lags
    n -= 1  # Adjust for 0-based indexing

    # 1. Compute the transition matrix P using vectorization.
    if params[0].get('exogen_transition',None) is not None:
        k = params[0]['exogen_transition'].shape[-1]
        def to_minimize_a2(params_):
            new_p = params_[:d**2].reshape(d,d)
            new_beta = reconstruct_beta(params_[d**2:], d, k, constraint=params[0].get('beta_constraint', 'general'))
            return -get_a2(new_p,new_beta,jointpi,params[0]['exogen_transition'],params[0].get('transition_f','softmax')) + params[0].get('lambda',0)*np.linalg.norm(params_[d**2:].ravel(),1)
        
        
        beta_minimal = minimize_beta_form(params[i]['beta'], constraint=params[0].get('beta_constraint', 'general'))
        params_a2 = np.concatenate([params[i]['P'].reshape(-1),beta_minimal])
        bounds_a2 = [(1e-8,1-1e-8)] * (d**2) + [(-10,10)] * len(beta_minimal)
        constraints_a2 = create_linear_constraints(d, k, constraint=params[0].get('beta_constraint','general'))
        result = minimize(to_minimize_a2, x0 = params_a2, bounds=bounds_a2, constraints=constraints_a2)
        new_params_a2 = result.x
        
        params[i+1]['P'] = new_params_a2[:d**2].reshape(d,d)
        params[i+1]['beta'] = reconstruct_beta(new_params_a2[d**2:], d, k, constraint=params[0].get('beta_constraint', 'general'))
    else:
        params[i+1]['P'] = calculate_P_optimized(jointpi, pi, n, d)




    # 2. Define the function to minimize (log-likelihood, negative log-likelihood) for omega and alpha.
    def to_minimize_a1(params_):
        # Split params_ into omega and alpha
        new_omega = params_[:d]
        new_alpha = params_[d:d+q*d].reshape((q, d)) if q>0 else None
        new_mu = params_[d*(q+1):].reshape((-1,d)) if params[0].get('mu_depends_on_delta',False) else params_[d*(q+1):]
        # Compute the negative log-likelihood
        return -get_a1(new_omega, eps, pi, new_alpha, params[0].get('exogen_additive',None), new_mu) + params[0].get('lambda',0)*np.linalg.norm(new_mu.ravel(),1)

    # 3. Flatten omega and alpha for optimization input and setup bounds.
    params_a1 = np.concatenate([params[i]['omega'], params[i].get('alpha',np.array([[]])).ravel(),params[i].get('mu',np.array([[]])).ravel()])  # Flatten omega, alpha, and mu
    c = np.size(params[i].get('mu',[]))
    bounds_a1 = [(1e-3, None)] * d + [(0, None)] * (q * d + c) if q*d+c>0 else [(1e-3, None)] * d 
    # 4. Minimize the negative log-likelihood to get new omega and alpha.
    result = minimize(to_minimize_a1, x0=params_a1, bounds=bounds_a1, method='L-BFGS-B')
    # 5. Retrieve the optimized omega and alpha values from the result.
    new_params_a1 = result.x
    
    
    params[i+1]['omega'] = new_params_a1[:d]
    if q>0:
       params[i+1]['alpha'] = new_params_a1[d:d+q*d].reshape((q, d))
    if 'exogen_additive' in params[0].keys():
        params[i+1]['mu'] = new_params_a1[d*(q+1):].reshape(-1,d) if params[0].get('mu_depends_on_delta',False) else new_params_a1[d*(q+1):]

    # 6. Update the initial state probabilities (pi_0) for the next iteration.
    params[i+1]['pi_0'] = np.copy(pi[1,-1])
    if abs(params[i+1]['pi_0'][1]) < 1e-10:
        params[i+1]['pi_0'] = np.array([1-1e-10,1e-10])
    elif abs(params[i+1]['pi_0'][0]) < 1e-10:
        params[i+1]['pi_0'] = np.array([1e-10,1-1e-10])
    # if params[0].get('exogen_transition',None) is not None:
    #     params[i]['Q_start'] = to_minimize_a1(params_a1)+to_minimize_a2(params_a2)
    #     params[i]['Q_mid'] = to_minimize_a1(new_params_a1) + to_minimize_a2(params_a2)
    #     params[i]['Q_end'] = to_minimize_a1(new_params_a1)+to_minimize_a2(new_params_a2)
    #     params_2 = deepcopy(params)
    #     params_2[i] = {key:(val if key not in ['omega','alpha','mu'] else params[i+1].get(key,None)) for key,val in params[i].items()}        
    #     params[i]['log-likelihood-mid'] = compute_log_likelihood(eps,params_2,i,pi)
    # else:
    #     params[i]['Q_start'] = to_minimize_a1(params_a1) - get_a2(params[i]['P'],np.zeros(d),jointpi,np.zeros(n))
    #     params[i]['Q_mid'] = to_minimize_a1(new_params_a1) - get_a2(params[i]['P'],np.zeros(d),jointpi,np.zeros(n))
    #     params[i]['Q_end'] = to_minimize_a1(new_params_a1) - get_a2(params[i+1]['P'],np.zeros(d),jointpi,np.zeros(n))
    #     params_2 = deepcopy(params)
    #     params_2[i] = {key:(val if key not in ['omega','alpha','mu'] else params[i+1].get(key,None)) for key,val in params[i].items()}        
    #     params[i]['log-likelihood-mid'] = compute_log_likelihood(eps,params_2,i)        
    return

def MS_EM_algorithm(eps,params,niter=10,return_pi=False,reorder=True):
    params = {0:copy(params)}
    
    if params[0].get('mu',np.array([])).ndim == 2:
        print('mu of dimension 2: will depend on delta')
        params[0]['mu_depends_on_delta'] = True
        
    # Iterative EM process
    for i in tqdm(range(niter)):
        # Run filtering and maximization steps
        pi,jointpi = MS_hamilton_filter(eps,params,i)
        params[i]['log-likelihood'] = compute_log_likelihood(eps,params,i,pi)
        MS_EM_Pass(pi,jointpi,eps,params,i)
    params[niter]['log-likelihood'] = compute_log_likelihood(eps,params,niter)
    if return_pi:
        return params,pi
    if reorder:
        params_keys = [key for key in ['omega','pi_0','alpha','P','mu','beta'] if (key in params[niter].keys())]                       
        return reorder_states(params,params_keys)
    else:
        return params



@njit
def phi_vectorized(eps, omega, alpha=None, exogen_additive=None,mu=None):
    """
    Vectorized version of phi to compute the results for all time steps `t` and all states `i`.
    
    Parameters:
    - eps: Array of residuals (size n)
    - omega: Array of unconditional variances for each state (size d)
    - alpha: ARCH coefficients (size qxd) (optional)
    - exogen: Exogenous variance component (size n) (optional)
    
    Returns:
    - pdf_values: Matrix of shape (n, d), where each element is the normal PDF value for each time step `t` and state `i`.
    """
    sigmas = get_sigma(eps,np.arange(len(eps)),omega,alpha,exogen_additive,mu)

    eps = replicate_eps_broadcast(eps,len(omega))
     # Calculate the normal PDF values
    pdf_values = normal_pdf_vectorized(eps, sigmas)  # Shape (n, d)
    
    return pdf_values

def phi(t,eps,i,omega,alpha=None,exogen_additive=None,mu=None):
    var = omega[i]
    if t>0:
        if alpha is not None:
            r = min(t,len(alpha))
            for j in range(r):
                var += alpha[j][i]*eps[t-1-j]**2
        if exogen_additive is not None:
            var += mu@exogen_additive[t-1]
    return normal_pdf(eps[t],np.sqrt(var))

@njit
def log_phi(eps, omega, alpha=None, exogen_additive=None,mu=None):
    n = len(eps)
    d = len(omega)
    vars = get_sigma(eps, np.arange(0, n), omega,alpha,exogen_additive,mu,var=True)
    # Calculer log_phi à partir des variances
    log_phi_values = -0.5 * (np.log(2 * np.pi * vars) + (eps[:, np.newaxis] ** 2) / vars)
    return log_phi_values


@njit
def get_a1(omega, eps, pi, alpha=None,exogen_additive=None,mu=None,lambda_=0):
    n = len(eps)
    log_phi_values = log_phi(eps,omega,alpha,exogen_additive,mu)

    a1 = np.sum(log_phi_values * pi[1:n + 1, -1, :]) # On va de pi[1] à pi[n], pour eps[0] à eps[n-1], puisque pi est décalié de 1

    return a1

def compute_log_likelihood(eps, params, i=None, pi=None,cheat=False):
    """
    Compute the log-likelihood given the parameters and data.
    
    Parameters:
    - eps: List of residuals
    - pi_0: Initial state probabilities (1D array)
    - omega: Unconditional variances for each state (1D array)
    - P: Transition matrix (2D array)
    - alpha: ARCH coefficients (2D array, optional)
    - exogen_additive: Exogeneous component (list, optional)
    
    Returns:
    - Log-likelihood of the observed data.
    """
    
    # Step 1: Get filtered probabilities using the Hamilton filter
    
    # Step 2: Compute log-likelihood for each time step
    if i is None:
        if pi is None:
            if cheat:
                pi = MS_hamilton_filter(eps, params, backward_pass=True)[0]
            else:
                pi = MS_hamilton_filter(eps, params, backward_pass=False)
        phis = phi_vectorized(eps, params['omega'], params.get('alpha',None), params.get('exogen_additive',None),params.get('mu',None))  # Matrix (n, d) of likelihoods for each state and time
    else:
        if pi is None:
            if cheat:
                pi = MS_hamilton_filter(eps, params, i ,backward_pass=True)[0]
            else:
                pi = MS_hamilton_filter(eps, params, i ,backward_pass=False)
        phis = phi_vectorized(eps, params[i]['omega'], params[i].get('alpha',None), params[0].get('exogen_additive',None),params[i].get('mu',None))  # Matrix (n, d) of likelihoods for each state and time        
    return weighted_log_sum(pi,phis,cheat=cheat)



@njit
def calculate_P_optimized(jointpi, pi, n, d):
    """
    Optimized version of P matrix computation using Numba and vectorized operations.
    
    Parameters:
    - jointpi : Array of shape (n+1, d, d), joint probability of transitions
    - pi : Array of shape (n+1, n+1, d), state probabilities
    - n : Number of time steps
    - d : Number of states
    
    Returns:
    - P : Array of shape (d, d), transition probability matrix
    """
    P = np.zeros((d, d))  # Initialize the P matrix
    
    # Precompute slices over t=2 to n+1 for jointpi and pi
    jointpi_sum = np.sum(jointpi[1:], axis=0)  # Sum over time for jointpi (d, d)
    pi_sum = np.sum(pi[1:n, n, :], axis=0)  # Sum over time for pi (d,)

    # Compute P[i, j] = jointpi_sum[i, j] / pi_sum[i] in a vectorized manner
    for i in range(d):
        if pi_sum[i] > 0:  # Avoid division by zero
            P[i, :] = jointpi_sum[i, :] / pi_sum[i]
        else:
            P[i, :] = 1.0 / d  # Handle cases where pi_sum[i] is zero (uniform distribution)

    return P



def create_linear_constraints(d, k ,constraint='general'):
    """
    Crée les contraintes linéaires pour le problème.

    Arguments :
    - d : dimension de la matrice.
    - constraint : type de contrainte ('general', 'equal_lines', 'symmetric').

    Retourne :
    - LinearConstraint pour le problème d'optimisation.
    """
    if constraint == 'general':
        n = d**2 + k * (d**2)  # P aplati + beta aplati
        A = np.zeros((d + d * k, n))
        for i in range(d):
            A[i, i*d:(i+1)*d] = 1  # Contraintes sur P
        for i in range(k*d):
            A[d + i, d**2 + i * d:d**2 + (i+1) * d] = 1  # Contraintes sur beta
        b = np.concatenate([np.ones(d), np.zeros(d*k)])
        return LinearConstraint(A, b, b)

    elif constraint == 'equal_lines':
        n = d**2 + d * k  # P aplati + beta réduit
        A = np.zeros((d + 1, n))
        for i in range(d):
            A[i, i*d:(i+1)*d] = 1  # Contraintes sur P
        A[d, d**2:] = 1  # Contraintes sur beta
        b = np.concatenate([np.ones(d), [0]])
        return LinearConstraint(A, b, b)

    elif constraint == 'symmetric':
        
        triu_indices = np.triu_indices(d)
        n_reduced = len(triu_indices[0])  # Nombre d'éléments dans la forme réduite
        n = d**2 + n_reduced  # P aplati + beta aplati
        A = np.zeros((d + d, n))
            
        # Indices triangulaires supérieurs (incluant la diagonale)
        
        # Matrice de contraintes A        
        # Construire les contraintes ligne par ligne
        for i in range(d):
            A[i, i*d:(i+1)*d] = 1  # Contraintes sur P
            for idx, (row, col) in enumerate(zip(*triu_indices)):
                if row == i or col == i:  # Si l'élément appartient à la ligne i
                    A[i+d, d**2+idx] += 1
        # Les limites inférieures et supérieures (somme nulle)
        b = np.concatenate([np.ones(d), np.zeros(d*k)])
        
        # Retourner les contraintes linéaires
        return LinearConstraint(A, b, b)
    
    else:
        raise ValueError(f"Unknown constraint type: {constraint}")

@njit
def get_a2(p,beta,jointpi,exogen_transition,kind='softmax'):
    d = len(p)
    c = np.zeros((d,d))
    for t in range(1,len(jointpi)-1):
        if exogen_transition is not None:
            P = get_transition_matrix(p,exogen_transition[t-1],beta,kind=kind)
        else:
            P = p
        c = c + np.log(P)*jointpi[t+1]
    return np.sum(c)

@njit
def weighted_log_sum(pi,phis,cheat=False):
    log_likelihood = 0
    n = len(phis)
    # Calculate the log-likelihood using the filtered probabilities and phi values
    for t in range(n):
        # Weighted sum across states
        if cheat:
            weighted_sum = np.dot(pi[t+1, n], phis[t])
        else:           
            weighted_sum = np.dot(pi[t+1, t], phis[t])
        
        # Add the log of the weighted sum to the total log-likelihood
        log_likelihood += np.log(weighted_sum)
    return log_likelihood

@njit
def replicate_eps_broadcast(eps, d):
    """
    Replicate a 1D array `eps` of size `n` into a 2D array of shape `(n, d)` using broadcasting.

    Parameters:
    - eps: 1D array of size `n`
    - d: Number of states (second dimension size)

    Returns:
    - A 2D array of shape `(n, d)` where each row is a copy of `eps[t]`.
    """
    n = len(eps)
    eps_2d = np.zeros((n, d))

    # Broadcasting the 1D array across the second dimension
    for j in range(d):
        eps_2d[:, j] = eps

    return eps_2d


def minimize_beta_form(beta, constraint='general'):
    """
    Extrait la forme minimale de la matrice beta en fonction de la contrainte.

    Arguments :
    - beta : matrice (d x d).
    - constraint : type de contrainte ('general', 'equal_lines', 'symmetric').

    Retourne :
    - vecteur aplati correspondant à la forme minimale de beta.
    """
    d = beta.shape[0]
    if constraint == 'general':
        return beta.ravel()  # Pas de réduction, on garde tous les paramètres
    elif constraint == 'equal_lines':
        return beta[0, :]  # Une seule ligne suffit
    elif constraint == 'symmetric':
        triu_indices = np.triu_indices(d)  # Indices pour la partie triangulaire supérieure
        return beta[triu_indices]  # On extrait uniquement la partie triangulaire supérieure
    else:
        raise ValueError(f"Unknown constraint type: {constraint}")

def reconstruct_beta(beta_flat, d, k, constraint='general'):
    """
    Reconstruit la matrice beta à partir de son vecteur aplati.
    
    Arguments :
    - beta_flat : vecteur aplati contenant les paramètres de beta.
    - d : dimension de la matrice beta (d x d).
    - constraint : type de contrainte ('general', 'equal_lines', 'symmetric').

    Retourne :
    - beta : matrice beta reconstruite.
    """
    if constraint == 'general':
        return beta_flat.reshape((d, d, k))
    elif constraint == 'equal_lines':
        import numpy as np
        return np.tile(beta_flat, (d, 1))  # Répéter la ligne d fois
    elif constraint == 'symmetric':
        import numpy as np
        beta = np.zeros((d, d, k))
        triu_indices = np.triu_indices(d)  # Indices pour la partie triangulaire supérieure
        beta[triu_indices] = beta_flat  # Remplit la partie supérieure
        beta = beta + beta.T - np.diag(np.diag(beta))  # Symétrise
        return beta
    else:
        raise ValueError(f"Unknown constraint type: {constraint}")
    
def sort_parameters(em_output, param_keys):
    """
    Trie les paramètres d'un dictionnaire EM en fonction de 'omega'.
    
    Arguments :
    - em_output : dict, dictionnaire contenant les résultats de l'algorithme EM.
    - param_keys : list, liste des clés représentant les paramètres du modèle, dont 'omega'.
    
    Retourne :
    - dict, dictionnaire avec les paramètres triés.
    """
    if 'omega' not in param_keys:
        raise ValueError("'omega' doit être dans param_keys pour trier les paramètres.")

    # Récupérer l'array 'omega' et les indices de tri
    omega = em_output['omega']
    if len(omega.shape) != 1:
        raise ValueError("'omega' doit être un array 1D.")

    sorted_indices = np.argsort(omega)  # Indices pour trier les paramètres

    # Trier les paramètres en respectant les indices de 'omega'
    sorted_em_output = em_output.copy()
    for key in param_keys:
        param = em_output[key]

        # Cas 1D : Trier selon sorted_indices
        if len(param.shape) == 1:
            sorted_em_output[key] = param[sorted_indices]

        # Cas particulier : 2D mais avec une seule ligne
        elif len(param.shape) == 2 and param.shape[0] == 1:
            sorted_em_output[key] = param[:, sorted_indices]

        # Cas 2D général : Trier les lignes selon sorted_indices
        elif len(param.shape) == 2:
            sorted_em_output[key] = param[sorted_indices, :]

        else:
            raise ValueError(f"Les paramètres dans {key} doivent être 1D ou 2D.")

    return sorted_em_output


def reorder_states_old(dico, param_keys=None):
    """
    Réordonne les états d'un dictionnaire de résultats de l'EM (multi-itérations)
    en utilisant l'ordonnancement de la dernière itération (n). Seuls les paramètres spécifiés
    dans param_keys sont réordonnés.

    Pour les matrices 2D, à la fois les lignes et les colonnes sont réordonnées.

    Arguments :
    - dico : dict, où chaque clé correspond à une itération (1, 2, ..., n), et chaque valeur est un dictionnaire
      contenant les résultats de l'EM pour cette itération.
    - param_keys : list ou None, les clés des paramètres à réordonner. Si None, tous les paramètres
      présents dans la dernière itération seront utilisés.

    Retourne :
    - dict, avec tous les dictionnaires réordonnés selon l'itération n.
    """
    # Récupérer le dictionnaire de la dernière itération
    last_iteration = max(dico.keys())
    last_dict = dico[last_iteration]

    # Déterminer les clés des paramètres à modifier
    if param_keys is None:
        param_keys = list(last_dict.keys())

    # Récupérer l'ordre des états à la dernière itération
    omega_last = last_dict['omega']
    sorted_indices = np.argsort(omega_last)

    # Fonction de tri pour une itération donnée
    def sort_single_iteration(em_output, sorted_indices, param_keys):
        sorted_em_output = em_output.copy()
        for key in param_keys:
            param = em_output[key]

            # Cas 1D
            if len(param.shape) == 1:
                sorted_em_output[key] = param[sorted_indices]

            # Cas particulier : 2D mais avec une seule ligne
            elif len(param.shape) == 2 and param.shape[0] == 1:
                sorted_em_output[key] = param[:, sorted_indices]

            # Cas 2D général : Trier les lignes et les colonnes
            elif len(param.shape) == 2:
                sorted_em_output[key] = param[sorted_indices, :][:, sorted_indices]

            else:
                raise ValueError(f"Les paramètres dans {key} doivent être 1D ou 2D.")

        return sorted_em_output

    # Réordonner toutes les itérations
    reordered_dico = {}
    for i in dico.keys():
        reordered_dico[i] = sort_single_iteration(dico[i], sorted_indices, param_keys)

    return reordered_dico

import numpy as np

def reorder_states(dico, param_keys=None):
    """
    Réordonne les états d'un dictionnaire de résultats de l'EM (multi-itérations)
    en utilisant l'ordonnancement de la dernière itération (n). Seuls les paramètres spécifiés
    dans param_keys sont réordonnés.

    Pour les matrices 2D, à la fois les lignes et les colonnes sont réordonnées.
    Pour les tenseurs 3D, seule la première dimension (les états) est réordonnée.

    Arguments :
    - dico : dict, où chaque clé correspond à une itération (1, 2, ..., n), et chaque valeur est un dictionnaire
      contenant les résultats de l'EM pour cette itération.
    - param_keys : list ou None, les clés des paramètres à réordonner. Si None, tous les paramètres
      présents dans la dernière itération seront utilisés.

    Retourne :
    - dict, avec tous les dictionnaires réordonnés selon l'itération n.
    """
    last_iteration = max(dico.keys())
    last_dict = dico[last_iteration]

    if param_keys is None:
        param_keys = list(last_dict.keys())

    omega_last = last_dict['omega']
    sorted_indices = np.argsort(omega_last)

    def sort_single_iteration(em_output, sorted_indices, param_keys):
        sorted_em_output = em_output.copy()
        for key in param_keys:
            param = em_output[key]

            if param.ndim == 1:
                sorted_em_output[key] = param[sorted_indices]

            elif param.ndim == 2:
                if param.shape[0] == 1:
                    sorted_em_output[key] = param[:, sorted_indices]
                else:
                    sorted_em_output[key] = param[sorted_indices, :][:, sorted_indices]

            elif param.ndim == 3:
                sorted_em_output[key] = param[sorted_indices, :, :]

            else:
                raise ValueError(f"Paramètre '{key}' a une forme non supportée ({param.shape}).")

        return sorted_em_output

    reordered_dico = {
        i: sort_single_iteration(dico[i], sorted_indices, param_keys)
        for i in dico.keys()
    }

    return reordered_dico





def viterbi_continuous(eps, params):
    """
    Algorithme de Viterbi pour un HMM (Hidden Markov Model).

    Args:
        eps (list): Liste des observations.
        d (int): nombre d'états
        start_prob (dict): Probabilités initiales de chaque état.
        trans_prob (dict of dict): Probabilités de transition entre états.
        emit_prob (dict of dict): Probabilités d'émission pour chaque état.

    Returns:
        list: Séquence la plus probable d'états cachés.
    """
    V = [{}]  # Viterbi matrix
    path = {}  # Meilleur chemin menant à chaque état
    
    d = len(params['pi_0'])
    emit_prob = phi_vectorized(eps,params.get('omega',None),params.get('alpha',None),params.get('exogen_additive',None),params.get('mu',None))

    transition = params.get('exogen_transition',None) is not None    
    if transition:
        trans_prob = [get_transition_matrix(params['P'],params['exogen_transition'][t-1],params['beta'],params.get('transition_f','softmax')) for t in range(1,len(eps))]
    else:
        trans_prob = [params['P']]*(len(eps)-1)
        print('No exogen transition')
    # Initialisation
    start_prob = params.get('pi_0',np.array([1/d]*d))
    for state in range(d):
        V[0][state] = np.log(start_prob[state]) + np.log(emit_prob[0,state])
        path[state] = [state]

    # Récursion    
    for t in range(1, len(eps)):
        V.append({})
        new_path = {}

        for curr_state in range(d):
            # Chercher le meilleur état précédent menant à curr_state
            (prob, prev_state) = max(
                (V[t - 1][prev_state] + np.log(trans_prob[t-1][prev_state][curr_state]) + np.log(emit_prob[t,curr_state]), prev_state)
                for prev_state in range(d)
            )
            V[t][curr_state] = prob
            new_path[curr_state] = path[prev_state] + [curr_state]

        path = new_path

    # Terminaison : choisir le chemin ayant la plus grande probabilité finale
    n = len(eps) - 1
    (final_prob, final_state) = max((V[n][state], state) for state in range(d))
    return np.array(path[final_state])


def get_QMLE_ARCH(eps,params):
    n = len(eps)
    vars = get_sigma(eps,np.arange(0,n),np.array([params.get('omega')]),params.get('alpha',np.array([0])),params.get('exogen_additive',None),params.get('mu',None),var=True).reshape(-1)
    QMLE = eps**2/vars + np.log(vars)
    return QMLE


def weighted_calibration(eps,params,weights,calibrate_mu):
   
    q = len(params.get('alpha',[]))
    def to_minimize(x):
        omega = x[0]
        alpha = x[1:1+q].reshape((q,1)) if q>0 else None
        mu = x[q+1:].reshape((-1,1)) if calibrate_mu else None
        p_ = {'omega':omega,'alpha':alpha,'mu':mu,'exogen_additive':params.get('exogen_additive',None)}
        QMLE = get_QMLE_ARCH(eps,p_)
        return sum(QMLE*weights)
    
    x0 = np.concatenate([params['omega'],params.get('alpha',np.array([0])).reshape(-1),np.array([0]) if calibrate_mu else np.array([])])
    bounds = [(1e-3, None)]
    if q>0:
        bounds += [(0, 1)] * q
    if params.get('mu',None) is not None:
        bounds += [(0,None)]
    result = minimize(to_minimize,x0,method='L-BFGS-B',bounds=bounds).x
    store_results = {'omega':result[0],'alpha':result[1:1+q].reshape((q,1)) if q>0 else None,'mu':result[q+1:].reshape((-1,1)) if calibrate_mu else None}
    return store_results
    


def get_initial_conditions(eps,initial_params={},d=2,q=1,niter=200):
    # Initial_params doit contenir les informations suivantes:
    ### 'exogen_transition et/ou 'exogen_additive' si besoin
    # On peut aussi spécifier les paramètres suivants, sans quoi ils seront initialisés selon des valeurs standard
    ### 'omega' - array de taille d (optionnel)
    ### 'P' - array de taille dxd (optionnel)
    ### 'pi_0' - array de taille d (optionnel)

        
    calibrate_mu = initial_params.get('exogen_additive',None) is not None
    transition = initial_params.get('exogen_transition',None) is not None
    if calibrate_mu:
        initial_params['exogen_additive'] = initial_params['exogen_additive'].reshape(-1,1)
    if transition:
        k = initial_params['exogen_transition'].shape[-1]
        initial_params['beta'] = np.zeros((d,d,k))
    
    hmm_initial_params = {
                'pi_0' : initial_params.get('pi_0',np.array([1/d]*d)),\
                'omega' :initial_params.get('omega',np.array([0.3,0.7]) if d==2 else np.array([0.5]*d)),\
                'P' : initial_params.get('P',np.ones((d,d))/d),\
                'exogen_transition' : initial_params.get('exogen_transition',None),\
                'beta' : initial_params.get('beta',None),\
                #'beta_constraint' : 'general',\
            }
    
    hmm_results = MS_EM_algorithm(eps,hmm_initial_params,niter=niter,return_pi=False)

    hmm_params = hmm_results[niter]

    
    
    #### Viterbi Algorithm
    
    states_sequence = viterbi_continuous(eps,hmm_params)
    plt.plot(states_sequence,label='Predicted states',color='blue',alpha=0.5,linestyle='',marker='o',markersize=0.1,linewidth=0.1)
    
    ### Estimation of the ARCH parameters
    
    weights = {}
    initial_arch_params = defaultdict(lambda:{})
    arch_params = defaultdict(lambda:{})
    for idx in range(d):
        #weights[idx] = hmm_pi[:,-1,idx][1:].argmax(axis=1)
        weights[idx] = (idx == states_sequence)
        initial_arch_params[idx] = {'omega':np.array([hmm_params['omega'][idx]]),'alpha':np.array([0]*q),'mu':np.array([0])\
                                            if calibrate_mu else None,'exogen_additive':initial_params.get('exogen_additive',None)}
        arch_params[idx] = weighted_calibration(eps,initial_arch_params[idx],weights[idx],calibrate_mu)
        
    for key in {'P','pi_0','beta'}:
        initial_params[key] = hmm_params.get(key,None)    
    initial_params['omega'] = np.concatenate([np.array([arch_params[idx]['omega']]) for idx in range(d)])
    if q>0:
        initial_params['alpha'] = np.concatenate([np.array([arch_params[idx]['alpha']]) for idx in range(d)])
        initial_params['alpha'] = initial_params['alpha'].reshape(d,q).T
    if calibrate_mu:
        initial_params['mu'] = np.concatenate([np.array([arch_params[idx]['mu']]) for idx in range(d)]).reshape(-1,d)
    initial_params['predicted_states'] = states_sequence
    return initial_params