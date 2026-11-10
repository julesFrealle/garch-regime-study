from scipy.stats import chi2
from collections import defaultdict
from copy import copy,deepcopy
from tqdm import tqdm
import statsmodels.api as sma
from utils import *
from functions import MS_hamilton_filter

def bissection(f, a, b, tol=1e-7, max_iter=1000):
    """
    Trouve le zéro d'une fonction continue croissante dans l'intervalle [a, b] en utilisant la méthode de la bissection.
    
    :param f: La fonction croissante dont on veut trouver le zéro.
    :param a: La borne inférieure de l'intervalle.
    :param b: La borne supérieure de l'intervalle.
    :param tol: La tolérance pour l'erreur (valeur par défaut = 1e-7).
    :param max_iter: Le nombre maximum d'itérations autorisées (valeur par défaut = 1000).
    
    :return: La valeur approchée de la racine (zéro de la fonction) si elle est trouvée, sinon None.
    """
    if f(a) * f(b) >= 0:
        raise ValueError(f"La fonction doit changer de signe dans l'intervalle [a, b], ici f({a})={f(a)} et f({b})={f(b)}")

    for i in range(max_iter):
        c = (a + b) / 2  # milieu de l'intervalle
        if abs(f(c)) < tol:  # si f(c) est suffisamment proche de 0, on a trouvé la racine
            return c
        elif f(a) * f(c) < 0:  # le zéro se trouve dans [a, c]
            b = c
        else:  # le zéro se trouve dans [c, b]
            a = c
    
    # Si on atteint ici, on n'a pas trouvé la racine avec la tolérance donnée
    return None

def get_var_gaussian(eps, params, T=None, i=None, pi=None, level=0.05,cheat=False):
    """
    Calcule la VaR sur une liste de moments T.

    Parameters:
    - eps : Série des résidus
    - T : Liste des instants où calculer la VaR
    - omega, alpha, P, pi_0 : Paramètres du modèle MS-ARCH
    - pi : Probabilités filtrées optionnelles (si déjà pré-calculées)
    - level : Niveau de confiance pour la VaR (ex: 0.05 pour VaR 5%)
    - cheat : Mode d'accélération (en utilisant pi[t, -1] au lieu de pi[t, t-1])

    Returns:
    - dict_var : Dictionnaire avec les VaR calculées à chaque instant t dans T
    """
    
    
    if i is None:
        params = { 0 : copy(params)}
        i = 0
        
    if T is None:
        T = np.arange(len(eps))
        
    omega = params[i].get('omega')
    alpha = params[i].get('alpha',None)
    mu = params[i].get('mu',None)
    exogen_additive = params[0].get('exogen_additive',None)
        
    # Si pi n'est pas pré-calculé, utiliser le filtre d'Hamilton pour l'obtenir
    if pi is None:
        pi, _ = MS_hamilton_filter(eps, params, i)
    
    # Calculer la VaR pour chaque valeur de T
    Vars = np.zeros((len(T)))
    d = len(omega)
    # Calcul pour chaque instant t dans T
    sigma = get_sigma(eps, T, omega, alpha,exogen_additive,mu,var=False)
    for idx,t in enumerate(T):
        f = lambda x: sum([normal_cdf(x / sigma[idx,i]) * pi[t+1, -1][i] if cheat else normal_cdf(x / sigma[idx,i]) * pi[t+1, t][i] for i in range(d)]) - level
        var_t = bissection(f, -200, 20)  # Utiliser bissection pour calculer la VaR
        Vars[idx] = var_t
    
    return Vars

def compute_vars(data,targets=None,level=0.05):
    Vars = defaultdict(lambda:{})
    if targets is None:
        targets = data.keys()
    for target in tqdm(targets):
        for indicator,result in data[target].items():
            dico = deepcopy(result)
            Vars[target][indicator] = {
                'data' : dico.get('data',None),\
                'Vars' : get_var_gaussian(dico['eps'],dico['result'],i=max(dico['result'].keys()),level=level),\
                'eps' : dico['eps'],\
            }
            Vars[target][indicator]['Violations'] = Vars[target][indicator]['eps']<Vars[target][indicator]['Vars']
    return Vars




def get_sigma_backtest(eps, params, T=None, i=None, pi=None, cheat=False):
    """
    Calcule la volatilité conditionnelle sigma_t^2 pour chaque instant t dans T,
    à des instants donnés, dans un cadre MS-ARCH.
    ATTENTION : il est important de récupérer la volatilité au carré par régime AVANT 
    de faire la somme pondérée par les pi (car sigma_t^2 = sum(pi[t+1, t] * sigma_t^2_i)).
    Sinon, le calcul serait faux.

    Parameters:
    - eps : np.array, série des résidus
    - params : dict, paramètres du modèle MS-ARCH
    - T : liste[int], instants où calculer la VaR
    - i : int, index du modèle dans params (par défaut 0)
    - pi : np.array, probabilités filtrées optionnelles (T x T x d)
    - cheat : bool, si True utilise pi[t+1, -1] au lieu de pi[t+1, t], cheat=True est utile pour 
    faire de l'inférence, tandis que cheat=False est nécessaire pour de la prédiction

    Returns:
    - np.array de shape (len(T), d) : contribution de chaque régime à sigma_t
    """

    # Si i n'est pas défini, on utilise les paramètres dans params[0]
    if i is None:
        params = {0: copy(params)}
        i = 0

    if T is None:
        T = np.arange(len(eps))

    omega = params[i].get('omega')
    alpha = params[i].get('alpha', None)
    mu = params[i].get('mu', None)
    exogen_additive = params[i].get('exogen_additive', None)

    d = len(omega)

    # Si les probabilités ne sont pas fournies, on les calcule
    if pi is None:
        pi, _ = MS_hamilton_filter(eps, params, i)  # suppose que cette fonction existe

    # Calcul de sigma (volatilité conditionnelle dans chaque régime)
    sigma = get_sigma(eps, T, omega, alpha, exogen_additive, mu, var=True)

    # Agrégation pondérée par les probabilités filtrées
    if cheat:
        # pi[t+1, -1] : dernière estimation disponible au temps t+1
        return np.array([
            sum([sigma[idx, j] * pi[t + 1, -1][j] for j in range(d)])
            for idx, t in enumerate(T)
        ])
    else:
        # pi[t+1, t] : probabilité filtrée au temps t, utilisée pour t+1
        return np.array([
            sum([sigma[idx, j] * pi[t + 1, t][j] for j in range(d)])
            for idx, t in enumerate(T)
        ])
        
        
def compute_sigma_backtest(data,targets=None,niter=1000):
    if targets is None:
        targets = data.keys()
    Sigma_backtest = defaultdict(lambda:{})
    for target in tqdm(targets):
        for indicator,dico in data[target].items():
            Sigma_backtest[target][indicator] = get_sigma_backtest(dico['eps'],dico['result'],i=max(dico['result'].keys()))
    return Sigma_backtest
        
        
       
def get_sigma_score(eps,sigma):
    """
    Compute the sigma score for one array of eps and one array of sigma
    
    Parameters
    ----------
    sigma : array
        sigma^2 forecast/inference for the target
    eps : array
        target returns    

    Returns
    -------
    float
        sigma score
    """
    return np.mean(eps**2/sigma**2 + np.log(sigma**2))
        
        
def compute_sigma_score(stored_data,sigmas,targets=None):
    """
    Compute the sigma score for each target and indicator.
    
    Parameters
    ----------
    Sigmas : dict
        Dictionary containing the sigma values for each target and indicator.
    stored_data : dict
        Dictionary containing the stored data for each target and indicator.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sigma scores.
    """
    scores = []
    if targets is None:
        targets = stored_data.keys()
    for target in targets:
        for indicator in stored_data[target].keys():
            scores.append(
                {
                    'target': target,
                    'indicator': indicator,
                    'sigma score': get_sigma_score(stored_data[target][indicator]['eps'],sigmas[target][indicator])
                }
            )
    return pd.DataFrame(scores)


def get_criteria_old(data,mean=False):
    # Initialisation de la table des critères avec une ligne 'No Exogeneous' (modèle sans exogène)
    # et une ligne par indicateur (modèle avec exogène)
    criteria = pd.DataFrame(index=['No Exogenous'] + list(data.keys()))
    
    ### Calcul des Degrees of Freedom (DF)
    # Hypothèse : modèle MS-ARCH(d, q) avec k exogènes
    # Nouveau calcul : df = d²(k+1) + d*q (structure corrigée pour séparer transition/additive)
    
    DF = []  # Initialisation des DF pour les modèles avec exogènes

    for dico in data.values():
        q, d = dico['final_result']['alpha'].shape  # q = ordre ARCH, d = nb régimes
        n = len(dico['eps'])
        # taille de l'échantillon
        k = dico['data'].shape[1]                  # nombre de variables exogènes
        df = d**2 * (k + 1) + d * q                  # formule corrigée pour df
        DF.append(df)

    # On ajoute en première ligne les df du modèle sans exogène :
    # Formule simplifiée : d² + d*q (cas k = 0)
    DF = [d**2 + d * q] + DF
    criteria['df'] = DF

    # Différence de complexité entre chaque modèle et le modèle de base
    criteria['excess df'] = criteria['df'] - criteria.loc['No Exogenous', 'df']

    # Log-vraisemblance du modèle de base (sans exogènes)
    initial_LL = data[list(data.keys())[0]]['initial_params']['log-likelihood']
    
    # Log-vraisemblances de tous les modèles (base + exogènes)
    LL = [initial_LL] + [dico['final_result']['log-likelihood'] for dico in data.values()]
    criteria['Log-Likelihood'] = LL

    # Calcul du critère AIC : 2k - 2ln(L)
    criteria['AIC'] = 2 * (criteria['df'] - criteria['Log-Likelihood'])

    # Calcul du critère BIC : k*ln(n) - 2ln(L)
    criteria['BIC'] = criteria['df'] * np.log(n) - 2 * criteria['Log-Likelihood']

    # Test du log-likelihood ratio (vs modèle sans exogènes)
    criteria['LR stat'] = 2 * (criteria['Log-Likelihood'] - criteria.loc['No Exogenous', 'Log-Likelihood'])

    # p-value associée au test LR sous chi² (ddl = df totale du modèle testé)
    criteria['LR p-value'] = 1 - chi2.cdf(criteria['LR stat'], criteria['excess df'])
    
    # if mean:
    #     for col in ['LR stat','p-value']:
    #         del criteria[col]
    #     for col in ['Log-Likelihood','AIC','BIC']:
    #         criteria[col] = criteria[col]/

    return criteria


def get_criteria(data,mean=True):
    # Initialisation de la table des critères avec une ligne 'No Exogeneous' (modèle sans exogène)
    # et une ligne par indicateur (modèle avec exogène)
    criteria = pd.DataFrame(index=list(data.keys()))
    
    ### Calcul des Degrees of Freedom (DF)
    # Hypothèse : modèle MS-ARCH(d, q) avec k exogènes
    # Nouveau calcul : df = d²(k+1) + d*q (structure corrigée pour séparer transition/additive)
    
    DF = []  # Initialisation des DF pour les modèles avec exogènes

    for dico in data.values():
        q, d = dico['final_result']['alpha'].shape  # q = ordre ARCH, d = nb régimes
        n = len(dico['eps'])
        # taille de l'échantillon
        k = dico.get('data',np.array([[]])).shape[1]                  # nombre de variables exogènes
        df = d**2 * (k + 1) + d * q                  # formule corrigée pour df
        DF.append(df)

    # Formule simplifiée : d² + d*q (cas k = 0)
    criteria['df'] = DF

    # Différence de complexité entre chaque modèle et le modèle de base
    criteria['excess df'] = criteria['df'] - criteria.loc['No Exogenous', 'df']
    
    
    
    # Log-vraisemblances de tous les modèles
    LL = [dico['final_result']['log-likelihood'] for dico in data.values()]
    criteria['Log-Likelihood'] = LL

    # Calcul du critère AIC : 2k - 2ln(L)
    criteria['AIC'] = 2 * (criteria['df'] - criteria['Log-Likelihood'])

    # Calcul du critère BIC : k*ln(n) - 2ln(L)
    criteria['BIC'] = criteria['df'] * np.log(n) - 2 * criteria['Log-Likelihood']

    # Test du log-likelihood ratio (vs modèle sans exogènes)
    criteria['LR stat'] = 2 * (criteria['Log-Likelihood'] - criteria.loc['No Exogenous', 'Log-Likelihood'])

    # p-value associée au test LR sous chi² (ddl = df totale du modèle testé)
    criteria['LR p-value'] = 1 - chi2.cdf(criteria['LR stat'], criteria['excess df'])
    
    if mean:
        mean_LL = [dico['final_result']['mean log-likelihood'] for dico in data.values()]
        criteria['mean Log-Likelihood'] = mean_LL
        sample_size = [len(dico['dates']) for dico in data.values()]
        criteria['sample size'] = sample_size
        criteria['mean AIC'] = criteria['AIC'] / criteria['sample size']
        criteria['mean BIC'] = criteria['BIC'] / criteria['sample size']
        


    return criteria

def get_criteria_all(data,targets=None):
    # Initialisation de la table des critères avec une ligne 'No Exogeneous' (modèle sans exogène)
    # et une ligne par indicateur (modèle avec exogène)
    DF = []
    if targets is None:
        targets = data.keys()
    for target in targets:
        result = data[target]
        to_add = get_criteria(result)
        to_add.insert(0,'target',target)
        DF.append(to_add)        
    return pd.concat(DF).reset_index(names='indicator').set_index('target')

def var_loss(eps, vars, level=0.05):
    loss = (level) * (eps - vars) * (eps > vars) + (1 - level) * (vars - eps) * (eps < vars)
    return np.mean(loss)


def kupiec_test(eps, vars, level=0.05):
    """
    Test de Kupiec stable (sans erreurs numériques).
    """
    violations = (eps < vars).astype(int)
    N = len(violations)
    x = violations.sum()

    # Éviter divisions par 0 ou log(0)
    pi_hat = np.clip(x / N, 1e-10, 1 - 1e-10)

    # Log-vraisemblances
    logL_unrestricted = x * np.log(pi_hat) + (N - x) * np.log(1 - pi_hat)
    logL_restricted = x * np.log(level) + (N - x) * np.log(1 - level)

    # Statistique du test
    LR = -2 * (logL_restricted - logL_unrestricted)
    p_value = 1 - chi2.cdf(LR, df=1)

    return p_value

def christoffersen_test(eps, vars, level=0.05):
    """
    Test de couverture conditionnelle (Christoffersen, 1998).
    Combine couverture inconditionnelle + indépendance des exceptions.

    Returns:
    - p_value du test global
    - statistiques : LR_indep, LR_uc, LR_cc
    """
    violations = (eps < vars).astype(int)
    T = len(violations)

    # Construire la matrice de transition des violations
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        if violations[t-1] == 0 and violations[t] == 0:
            n00 += 1
        elif violations[t-1] == 0 and violations[t] == 1:
            n01 += 1
        elif violations[t-1] == 1 and violations[t] == 0:
            n10 += 1
        elif violations[t-1] == 1 and violations[t] == 1:
            n11 += 1

    # Fréquences conditionnelles
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi_hat = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Statistique d’indépendance
    logL_indep = (n00 + n01) * np.log(1 - pi01) + n01 * np.log(pi01 + 1e-12) + \
                 (n10 + n11) * np.log(1 - pi11) + n11 * np.log(pi11 + 1e-12)

    logL_indep_null = (n00 + n01 + n10 + n11) * np.log(1 - pi_hat) + \
                      (n01 + n11) * np.log(pi_hat + 1e-12)

    LR_indep = -2 * (logL_indep_null - logL_indep)

    # LR_uc = test de Kupiec
    LR_uc = kupiec_test(eps, vars, level)

    # LR_cc = test global
    LR_cc = LR_uc + LR_indep
    p_value_cc = 1 - chi2.cdf(LR_cc, df=2)

    return p_value_cc

def dq_test(eps, vars, level=0.05, lags=4, X_exog=None):
    """
    Dynamic Quantile (DQ) Test - Engle & Manganelli (2004)
    
    Parameters:
        eps : array-like, shape (n,)
            Série des résidus (ou rendements réels).
        vars : array-like, shape (n,)
            Série des VaR estimées (à un niveau donné level).
        level : float
            Niveau de VaR (ex: 0.01 ou 0.05).
        lags : int
            Nombre de retards des violations à inclure dans la régression.
        X_exog : array-like, shape (n,) ou (n, k), optional
            Variables explicatives exogènes (mêmes dates que eps/vars).

    Returns:
        p_value : float
            La p-value du test.
        stat : float
            Statistique DQ (n * R² de la régression).
    """
    eps = np.asarray(eps)
    vars = np.asarray(vars)
    y = (eps < vars).astype(int)  # 1 si violation

    T = len(y)

    # Construction de la matrice X : [constante, lags de y - level, VaR, exogènes...]
    X_list = [np.ones(T)]  # constante

    # Ajout des lags des violations centrées : (y - level)
    for i in range(1, lags + 1):
        lagged = np.roll(y - level, i)
        lagged[:i] = 0  # padding initial
        X_list.append(lagged)

    # Ajout de la VaR
    X_list.append(vars)

    # Ajout des exogènes, si spécifiées
    if X_exog is not None:
        X_exog = np.asarray(X_exog)

        if X_exog.ndim == 1:
            X_list.append(X_exog)
        elif X_exog.ndim == 2:
            for i in range(X_exog.shape[1]):
                X_list.append(X_exog[:, i])
        else:
            raise ValueError("X_exog doit être de dimension 1 (n,) ou 2 (n, k)")

    # Matrice finale
    X = np.column_stack(X_list)

    # Supprimer les premières lignes affectées par les lags
    y_reg = y[lags:]
    X_reg = X[lags:]

    # Régression OLS
    model = sma.OLS(y_reg, X_reg)
    results = model.fit()

    # Statistique du test : n * R^2
    R2 = results.rsquared
    stat = len(y_reg) * R2
    df = X_reg.shape[1]
    p_value = 1 - chi2.cdf(stat, df)

    return p_value

def var_tests(data,targets=None,level=0.05):
    store = []
    if targets is None:
        targets = data.keys()
    for target in tqdm(targets):
        for indicator,result in data[target].items():
            dico = deepcopy(result)
            new_line = {
                        'target':target,\
                        'indicator':indicator,\
                        'Violation Rate' : np.mean(dico['Violations']),\
                        'Kupiec p-value' : kupiec_test(dico['eps'],dico['Vars'],level=level),\
                        'Christoffersen p-value' : christoffersen_test(dico['eps'],dico['Vars'],level=level),\
                        'DQ p-value' : dq_test(dico['eps'],dico['Vars'],level=level,X_exog=None),\
                        'VaR loss' : var_loss(dico['eps'],dico['Vars'],level=level),\
                        }
            store.append(new_line)

    return pd.DataFrame(store).set_index('target')
            
            
            
            
            
def get_rank(data, Columns=[None], method='median', Ascending=None):
    """
    Classe les modèles (indicateurs) en fonction de métriques données sur différentes cibles (targets).
    
    Args:
        data (pd.DataFrame): tableau des résultats avec au moins les colonnes ['indicator', 'target', ...]
        measure (str): 'median' ou 'mean' pour le type d'agrégation des rangs.
        Columns (list): liste des colonnes à utiliser pour le classement.
        Ascending (dict): pour chaque colonne, booléen indiquant si un plus petit score est meilleur.

    Returns:
        pd.DataFrame: rangs agrégés des indicateurs sur les colonnes spécifiées.
    """
    if Ascending is None:
        Ascending = {
            'Log-Likelihood': False,
            'AIC': True,
            'BIC': True,
            'LR p-value': True,
            'Kupiec p-value': False,
            'Christoffersen p-value': False,
            'DQ p-value': False,
            'Mean Violation Error': True,
            'VaR loss': True,
            'sigma score': True,
        }

    if method == 'median':
        return pd.concat([
            pd.DataFrame(
                data.reset_index()
                    .set_index('indicator')
                    .groupby('target')[column]
                    .rank(ascending=Ascending[column])
                    .groupby('indicator')
                    .median()
            ) for column in Columns
        ], axis=1)

    elif method == 'mean':
        return pd.concat([
            pd.DataFrame(
                data.reset_index()
                    .set_index('indicator')
                    .groupby('target')[column]
                    .rank(ascending=Ascending[column])
                    .groupby('indicator')
                    .mean()
            ) for column in Columns
        ], axis=1)
    
    else:
        raise ValueError('method should be either median or mean')


def apply_preprocessing(data, preprocessing_dict, tickers=None):
    """
    Modifie le DataFrame df in-place en appliquant le prétraitement spécifié dans preprocessing_dict.
    
    Parameters:
    - df: pandas DataFrame contenant toutes les séries.
    - preprocessing_dict: dict {ticker: [list de transformations]}.
    - tickers: liste optionnelle des tickers à traiter. Si None, tous les tickers du dictionnaire sont utilisés.
    
    Chaque transformation est sans look-forward.
    """
    
    df = deepcopy(data)
    
    if tickers is None:
        tickers = list(preprocessing_dict.keys())
    
    for ticker in tickers:
        if ticker not in df.columns:
            print(f"⚠️ Ticker {ticker} not found in DataFrame. Skipping.")
            continue
        if ticker not in preprocessing_dict:
            print(f"⚠️ No preprocessing steps found for {ticker}. Skipping.")
            continue

        series = df[ticker].copy()

        for step in preprocessing_dict[ticker]:
            if step == 'log':
                series = np.log(series)
            elif step == 'diff':
                series = series.diff()
            elif step == 'standardize':
                # mean = series.expanding().mean()
                # std = series.expanding().std(ddof=0)
                # mean = series.mean()
                # std = series.std(ddof=0)
                # series = (series - mean) / std
                series = (series.max() - series)/ (series.max() - series.min())
            elif step.startswith('adjust_with:'):
                fx_ticker = step.split(':')[1]
                if fx_ticker not in df.columns:
                    raise ValueError(f"Missing FX rate {fx_ticker} in DataFrame for adjustment of {ticker}")
                fx = df[fx_ticker]
                if 'EURUSD' in fx_ticker:
                    series = series * fx
                else:
                    series = series / fx
            else:
                raise ValueError(f"Unknown preprocessing step: {step} for ticker {ticker}")
        if 'log'in preprocessing_dict[ticker] and 'diff' in preprocessing_dict[ticker] and 'standardize' not in preprocessing_dict[ticker]:
            series *= 100
        
        df[ticker] = series  # overwrite original column in-place

    return df




def adf_with_ljungbox(y, max_lags=15, alpha=0.05, verbose=True):
    """
    Test de racine unitaire ADF avec sélection du nombre de lags via Ljung-Box.
    
    Paramètres :
    ------------
    y : array-like
        Série temporelle à tester (1D)
    max_lags : int
        Nombre maximum de retards testés dans Ljung-Box
    alpha : float
        Seuil de test (par défaut 5%)
    verbose : bool
        Affiche les résultats intermédiaires

    Retourne :
    ----------
    dict avec p-valeur, stat ADF, nombre optimal de lags retenus
    """
    y = pd.Series(y).dropna()
    best_lag = 0

    for lag in range(1,max_lags + 1):
        X = pd.concat(
            [y.shift(1)] + [y.diff().shift(i) for i in range(1, lag)],
            axis=1
        )[lag:]
        X.columns = ['y_lag1'] + [f'dy_lag{i}' for i in range(1, lag)]
        Y = y.diff()[lag:]
        res = sma.OLS(Y, sma.add_constant(X)).fit()        
        lb_test = acorr_ljungbox(res.resid, lags=[lag + 1], return_df=True)
        pval = lb_test.iloc[0]['lb_pvalue']

        if pval > alpha:
            best_lag = lag
            if verbose:
                print(f"Lag {lag} accepté : p-val = {pval:.4f}")
            break
        elif verbose:
            print(f"Lag {lag} rejeté : p-val = {pval:.4f}")

    # On applique maintenant le test ADF avec le lag choisi
    adf_result = adfuller(y, maxlag=best_lag, regression='c', autolag=None)
    stat, pval, usedlag, nobs, _ = adf_result

    if verbose:
        print("\n==== Résultat du test ADF ====")
        print(f"Statistique ADF     : {stat:.4f}")
        print(f"P-valeur            : {pval:.4f}")
        print(f"Retards utilisés    : {usedlag}")

    return alpha>pval
    
   

def get_integration_order(data,columns=None, max_lags=15, alpha=0.05, verbose=True):
    """
    Détermine l'ordre d'intégration de chaque colonne d'un DataFrame.
    
    Paramètres :
    ------------
    data : DataFrame
        Données à tester
    columns : list
        Liste des colonnes à tester (par défaut toutes)
    max_lags : int
        Nombre maximum de lags testés dans Ljung-Box
    alpha : float
        Seuil de test (par défaut 5%)
    verbose : bool
        Affiche les résultats intermédiaires

    Retourne :
    ----------
    dict avec l'ordre d'intégration pour chaque colonne
    """
    if columns is None:
        columns = data.columns

    data_ = deepcopy(data)

    results = {}
    for col in columns:
        print(col)
        k = 0
        y = data_[col].dropna()
        while k<20 and  adf_with_ljungbox(y, max_lags=max_lags, alpha=alpha, verbose=verbose)==0:
            y = data_[col].diff().dropna()
            k+=1
        results[col] = k
    return results
