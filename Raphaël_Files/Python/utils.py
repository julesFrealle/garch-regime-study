import numpy as np
from numba import njit
from math import exp, log, erf, sqrt
import pandas as pd
import matplotlib.pyplot as plt

@njit
def get_sigma(eps, t_vals, omega, alpha=None, exogen_additive=None, mu=None, var=False):
    """
    Calcule la variance conditionnelle pour chaque état à plusieurs instants `t`.

    Parameters:
    - eps : Série des résidus (taille n)
    - t_vals : Integer ou liste des instants `t` pour lesquels on calcule les variances
    - omega : Paramètres de variance inconditionnelle pour chaque état (taille d)
    - alpha : Coefficients ARCH pour chaque lag (facultatif, taille qxd)
    
    Returns:
    - sigmas : Matrice (kxd) des écarts-types conditionnels pour chaque `t` et chaque état, ou variances.
    
    """
    d = len(omega)

    # Vérification si t_vals est un entier
    if t_vals.ndim == 0:  # Vérifie si t_vals est un entier
        vars = omega.copy()  # Initialisation avec omega

        t = t_vals  # Utiliser l'entier directement

        if t>0 :
            if alpha is not None:
                r = min(t, len(alpha))
                for j in range(r):
                    vars += alpha[j] * eps[t - 1 - j] ** 2
            if exogen_additive is not None:
                vars += manual_matrix_multiply(exogen_additive[t-1],mu) ### En fait la dépendance ne change rien au vu de la forme du produit matriciel, on veut toujours sigma[i] += X[t]*mu[i] (en dim 1)
        if var:
            return vars
        sigmas = np.sqrt(vars)
        return sigmas

    else:
        # Cas où t_vals est un tableau
        k = len(t_vals)
        sigmas = np.zeros((k, d))
        Vars = np.zeros((k, d))
        for idx in range(k):
            t = t_vals[idx]  # Récupérer l'instant t
            vars = omega.copy()  # Initialisation avec omega
            if t>0:
                if alpha is not None:
                    r = min(t, len(alpha))
                    for j in range(r):
                        vars += alpha[j] * eps[t - 1 - j] ** 2
                if exogen_additive is not None:
                    vars += manual_matrix_multiply(exogen_additive[t - 1],mu)  # Utiliser 'vars' correctement
            if var:
                Vars[idx, :] = vars
            else:
                sigmas[idx, :] = np.sqrt(vars)
        if var:
            return Vars
        return sigmas

@njit
def manual_matrix_multiply(A, B):
    """
    Perform matrix multiplication manually using Numba for efficiency.
    Supports vector-vector, matrix-vector, and matrix-matrix multiplication.
    Handles special broadcasting cases like NumPy's "@" operator.
    """
    # Case 1: Both inputs are 1D arrays (vector dot product)
    if A.ndim == 1 and B.ndim == 1:
        if A.shape[0] != B.shape[0]:
            raise ValueError(f"Incompatible shapes for dot product: {A.shape} and {B.shape}")
        result = 0.0
        for i in range(A.shape[0]):
            result += A[i] * B[i]
        return result

    # Case 2: A is 1D and B is 2D (broadcast A as a row vector)
    elif A.ndim == 1 and B.ndim == 2:
        n, m = B.shape
        if A.shape[0] != n:
            raise ValueError(f"Incompatible shapes for vector-matrix multiplication: {A.shape} and {B.shape}")
        result = np.zeros(m)
        for j in range(m):
            for i in range(n):
                result[j] += A[i] * B[i, j]
        return result

    # Case 3: A is 2D and B is 1D (matrix-vector multiplication)
    elif A.ndim == 2 and B.ndim == 1:
        n, m = A.shape
        if m != B.shape[0]:
            raise ValueError(f"Incompatible shapes for matrix-vector multiplication: {A.shape} and {B.shape}")
        result = np.zeros(n)
        for i in range(n):
            for j in range(m):
                result[i] += A[i, j] * B[j]
        return result

    # Case 4: A and B are both 2D matrices
    elif A.ndim == 2 and B.ndim == 2:
        n, m = A.shape
        m2, p = B.shape
        if m != m2:
            raise ValueError(f"Incompatible shapes for matrix multiplication: {A.shape} and {B.shape}")
        result = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    result[i, j] += A[i, k] * B[k, j]
        return result

    # Case 5: A is 1D and B is 2D (special broadcasting case)
    elif A.ndim == 1 and B.ndim == 2 and A.shape == (1,):
        result = np.zeros(B.shape[1])
        for j in range(B.shape[1]):
            result[j] = A[0] * B[0, j]
        return result

    # Unsupported case
    else:
        raise ValueError(f"Unsupported dimensions for A and B: {A.shape} and {B.shape}")

@njit
def normal_pdf(x, sigma):
    """
    Implémentation manuelle de la fonction de densité normale N(0, sigma^2).
    
    Parameters:
    - x : Valeur du résidu
    - sigma : Écart-type de la distribution
    
    Returns:
    - pdf : Valeur de la densité de probabilité pour N(0, sigma^2)
    """
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (x / sigma) ** 2)

@njit
def normal_cdf(x):
    """
    Numba-compatible implementation of the normal CDF.
    
    Parameters:
    - x : Value at which to evaluate the CDF.
    
    Returns:
    - CDF of the normal distribution at x.
    """
    return 0.5 * (1 + erf(x / sqrt(2)))

@njit
def normal_pdf_vectorized(x, sigma):
    """ Compute the normal probability density function for vectorized inputs. """
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (x / sigma) ** 2)

@njit
def get_transition_matrix(p, X, beta,kind='softmax'):
    """
    Calcule la matrice de transition basée sur la formule g_t(X_t, i, j).
    
    Arguments:
    - p : fonction ou matrice (n, n) représentant p(i, j), probabilité de transition de base.
    - X : vecteur ou scalaire des variables exogènes X_t.
    - beta : vecteur ou matrice (n, n) des coefficients beta_j.
    
    Retourne :
    - matrice de transition (n, n)
    """   
    d = len(beta)
    y = np.zeros((d,d))
    if kind == 'softmax': 
        for i in range(d):
            for j in range(d):
                y[i,j] = exp(manual_matrix_multiply(beta[i,j],X))
    elif kind == 'soft-exp':
        for i in range(d):
            for j in range(d):
                y[i,j] = log(1+exp(manual_matrix_multiply(beta[i,j],X)))
    else:
        raise ValueError('Wrong function kind, kind must be in ["softmax","soft-exp"]')
    P = p * y
    P = (P.T/P.sum(axis=1)).T
    return P

# Fonction pour convertir le dictionnaire en DataFrame
def convert_dict_to_dataframe(dico):
    df = pd.DataFrame()

    for iteration, params_ in dico.items():
        row = {}
        
        # Décomposer les arrays/matrices
        for key, value in params_.items():
            if isinstance(value, (np.ndarray,list)):  # Convertit liste ou matrice
                value = pd.DataFrame(value).values  # Assure un traitement uniforme
                
                if value.ndim == 1:  # Si vecteur 1D
                    for i, val in enumerate(value):
                        row[f"{key}_{i+1}"] = val
                elif value.ndim == 2:  # Si matrice 2D
                    for i in range(value.shape[0]):
                        for j in range(value.shape[1]):
                            row[f"{key}_{i+1}{j+1}"] = value[i, j]
            else:
                row[key] = value  # Valeurs scalaires
        
        # Ajouter l'itération comme index
        row['iteration'] = iteration
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    return df.set_index('iteration')



def generate_colors(d, avoid_range=(0.4, 0.55)):
    """
    Generate `d` distinct colors while avoiding a specific range of the spectrum.
    
    Parameters:
    d (int): Number of colors to generate.
    avoid_range (tuple): Range of the spectrum to avoid (in HSV space).
    
    Returns:
    List[str]: List of `d` color codes.
    """
    cmap = plt.get_cmap('hsv')  # Use the 'hsv' colormap for distinct colors
    avoid_min, avoid_max = avoid_range
    colors = []
    
    # Spread colors evenly but skip the avoid_range
    for i in range(d):
        hue = i / d
        if avoid_min <= hue <= avoid_max:
            hue += (avoid_max - avoid_min)  # Shift the hue to avoid turquoise range
        hue = hue % 1  # Ensure hue is within [0, 1] after shifting
        colors.append(cmap(hue))
    
    return colors
