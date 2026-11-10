#from classes import *
#from utils import *
from functions import *
from metrics import *

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.tsa as sm
import statsmodels.api as sma

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import plotly.express as px
import plotly.io as pio

import seaborn as sns
from IPython.display import display
import pickle
import os
from math import ceil
from collections import defaultdict
import pandas as pd
import numpy as np

from arch import arch_model


#%load_ext autoreload
#%autoreload 2
pd.set_option('display.max_columns', None)






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



def process_excel1(file) :
    data = pd.read_excel(file)
    new_header = data.iloc[1]
    description = data.iloc[0]
    data = data[2:]
    data.columns = new_header
    data.set_index('Dates', inplace=True)
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    data = data.astype(float)
    return data

def process_excel(file):
    """
    Loads and cleans Excel financial time series data.
    - Uses row 2 as column headers
    - Row 1 (index 0) is saved as metadata (optional)
    - Sets 'Dates' as index and parses to datetime
    - Returns clean DataFrame with unnamed columns
    """
    data = pd.read_excel(file)
    
    new_header = data.iloc[1]        # Row 1: new column names
    description = data.iloc[0]       # Row 0: just metadata
    data = data[2:]                  # Drop header rows
    data.columns = new_header
    data.columns.name = None         # ✅ REMOVE column name (important!)
    
    data.set_index('Dates', inplace=True)
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    # Convert columns to numeric, ignore errors
    data = data.apply(pd.to_numeric, errors='coerce')
    
    return data


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
        k = initial_params['exogen_additive'].shape[1]  # number of exos

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
        initial_arch_params[idx] = {'omega':np.array([hmm_params['omega'][idx]]),'alpha':np.array([0]*q),'mu': np.zeros(k)\
                                            if calibrate_mu else None,'exogen_additive':initial_params.get('exogen_additive',None)}
        arch_params[idx] = weighted_calibration(eps,initial_arch_params[idx],weights[idx],calibrate_mu)

        #Before it was 'mu':np.array([0]) 
        
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


def calibrate_and_store(
    data,                     # DataFrame contenant la cible et les variables exogènes
    targets,                  # Liste des noms de séries cibles (colonnes de data)
    indicators,               # Liste de listes de noms de colonnes exogènes
    niter_HMM=100,           # Nb d’itérations pour initialisation type HMM
    niter_MS=300,            # Nb d’itérations pour estimation sans exogènes
    niter_X=0,              # Nb d’itérations pour estimation avec exogènes
    d=2, q=1,                 # Ordres du MS-ARCH
    start=None, end=None,     # Dates pour restreindre les données exogènes
    additive_method='absolute', # Méthode de transformation des exogènes (|X| ou X²)
    folder='RealDataResults', # Dossier de sauvegarde
    preprocessing_dict = None   # Dictionnaire de prétraitement éventuel
):

    
    folder = 'Results/' + folder
    store = defaultdict(lambda: {})

    os.makedirs(folder, exist_ok=True)

    if preprocessing_dict is not None:
        data = apply_preprocessing(data, preprocessing_dict)
        with open(f"{folder}//preprocessing_dict.pkl", "wb") as f:
            pickle.dump(preprocessing_dict, f)
    with open(f"{folder}//preprocessed_data.pkl", "wb") as f:
        pickle.dump(data, f)

    
    #if np.ndim(indicators) != 2:
    #    raise ValueError('Indicators should be a list of lists')
    
    if not indicators:
        indicators = []
    elif np.ndim(indicators) != 2:
        raise ValueError('Indicators should be a list of lists')

    for target in tqdm(targets):
        # Étape 1 : prétraitement eps seul
        eps_series = data[target].dropna()
        valid_eps_dates = eps_series.index
        eps = np.array(eps_series, dtype='float64')

        # Conditions initiales (indépendantes des indicateurs)
        initial_params = get_initial_conditions(eps, {}, d, q, niter=niter_HMM)
        constrained_result = MS_EM_algorithm(eps, initial_params, niter=niter_MS)
        new_initial_params = deepcopy(constrained_result[niter_MS])

        store[target]['No Exogenous'] = {
            'eps': deepcopy(eps),
            'dates' : valid_eps_dates,
            'target': target,
            'indicator': 'No Exogenous',
            'initial_params': deepcopy(new_initial_params),
            'result': deepcopy(constrained_result),
            'final_result': deepcopy(new_initial_params),
            'states sequence': viterbi_continuous(eps, new_initial_params)
        }
        store[target]['No Exogenous']['final_result']['mean log-likelihood'] = new_initial_params['log-likelihood']/len(valid_eps_dates)
        
        with open(f"{folder}//{target}+No Exogenous_results"+".pkl", "wb") as f:
            pickle.dump(store[target]['No Exogenous'], f)

        for indicator in tqdm(indicators):
            path = '+'.join(indicator)
            
            # Étape 2 : préparation des indicateurs restreints aux dates valides d'eps
            indicator_df = data[indicator].loc[valid_eps_dates].copy()

            # Drop uniquement sur indicateurs
            indicator_df = indicator_df.dropna()
            joint_dates = indicator_df.index

            # Appliquer filtre start/end
            if start is not None:
                joint_dates = joint_dates[joint_dates >= pd.to_datetime(start)]
            if end is not None:
                joint_dates = joint_dates[joint_dates <= pd.to_datetime(end)]

            if joint_dates.empty:
                print(f"Skipped {target}+{path}: no valid joint dates.")
                continue

            # Final alignement
            eps_aligned = eps_series.loc[joint_dates]
            X = indicator_df.loc[joint_dates].values
            eps_arr = np.array(eps_aligned, dtype='float64')

            k = X.shape[1]
            
            # Préparation des paramètres et estimation
            params = deepcopy(new_initial_params)
            params['exogen_transition'] = deepcopy(X)
            if additive_method == 'absolute':
                additive = np.abs(X)
            elif additive_method == 'square':
                additive = np.square(X)
            else:
                raise ValueError("additive_method must be either 'absolute' or 'square'")
            params['exogen_additive'] = deepcopy(additive)

            params['mu'] = np.zeros((k, d))
            params['beta'] = np.zeros((d, d, k))

            result = MS_EM_algorithm(eps_arr, params, niter=niter_X)

            store[target][path] = {
                'data': deepcopy(X),
                'eps': deepcopy(eps_arr),
                'dates': joint_dates,
                'target': target,
                'indicator': path,
                'initial_params': deepcopy(params),
                'result': deepcopy(result),
                #'pi': deepcopy(pi),
                'final_result': result[niter_X],
                'states sequence': viterbi_continuous(eps_arr, result[niter_X])
            }

            store[target][path]['final_result']['exogen_transition'] = deepcopy(X)
            store[target][path]['final_result']['exogen_additive'] = deepcopy(additive)
            store[target][path]['final_result']['mean log-likelihood'] = result[niter_X]['log-likelihood']/len(joint_dates)

            #if '+' in path:
            #    path = '('+path+')'
            #with open(f"{folder}//{target}+{path}_results"+".pkl", "wb") as f:
            #    pickle.dump(store[target][path], f)

            if '+' in path:
                path_key = '(' + path + ')'
            else:
                path_key = path

            store[target][path_key] = store[target][path]  # duplicate under new key
            with open(f"{folder}//{target}+{path_key}_results.pkl", "wb") as f:
                pickle.dump(store[target][path_key], f)

    return store

def load_stored_results(folder='SPX_VIX_GOOD_RUN'):
    result_dir = f"Results/{folder}"
    stored_data = defaultdict(lambda: {})
    for fname in os.listdir(result_dir):
        if not fname.endswith('_results.pkl'):
            continue
        path = os.path.join(result_dir, fname)
        with open(path, 'rb') as f:
            result = pickle.load(f)
        parts = fname.replace('_results.pkl', '').split('+')
        target = parts[0]
        indicator = parts[1] if len(parts) > 1 else 'No Exogenous'
        stored_data[target][indicator] = result
    return stored_data



def plot_model_vs_vix(data, stored_data, sigma_dict, target='SPX Index', comparison_series='VIX Index'):
    """
    Plots:
    - 1 large VIX vs SPX Index + X=None plot
    - A row of model-only volatility plots (no VIX)
    - A row of full overlays: VIX vs each model
    
    All volatilities are annualized: sqrt(sigma²) * sqrt(252)
    """

    if target not in stored_data or target not in sigma_dict:
        print(f"❌ Target '{target}' not found in inputs.")
        return

    indicators = list(stored_data[target].keys())
    indicators = sorted(indicators, key=lambda x: (x != 'No Exogenous', x))  # put 'No Exogenous' first

    # Load VIX and No Exogenous model
    base_result = stored_data[target].get('No Exogenous')
    base_sigma2 = sigma_dict[target].get('No Exogenous')

    if base_result is None or base_sigma2 is None:
        print("❌ Missing 'No Exogenous' model.")
        return

    base_dates = base_result['dates']
    base_vol = np.sqrt(base_sigma2) * np.sqrt(252)
    try:
        base_vix = data.loc[base_dates, comparison_series].values
        #base_vix = data.reindex(base_dates)[comparison_series].values

    except Exception as e:
        print(f"❌ Could not extract VIX for 'No Exogenous': {e}")
        return

    # --- A. BIG TOP PLOT ---
    plt.figure(figsize=(14, 6))
    plt.plot(base_dates, base_vol, label='Model Volatility (X = None)', linewidth=1.5, color='royalblue')
    plt.plot(base_dates, base_vix, label='VIX Index', linewidth=1.5, color='darkorange')
    plt.title("VIX vs SPX Index + X = None", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- B. PLOTS OF MODEL VOLATILITIES ALONE (excluding No Exogenous) ---
    other_indicators = [ind for ind in indicators if ind != 'No Exogenous']
    n = len(other_indicators)
    if n > 0:
        ncols = 2
        nrows = ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
        axes = axes.flatten()

        for i, indicator in enumerate(other_indicators):
            dico = stored_data[target][indicator]
            sigma2 = sigma_dict[target].get(indicator)

            if sigma2 is None or np.isnan(sigma2).all():
                axes[i].set_title(f"{indicator} — Skipped (NaNs)")
                axes[i].axis('off')
                continue

            vol = np.sqrt(sigma2) * np.sqrt(252)
            dates = dico['dates']

            axes[i].plot(dates, vol, label=f'X = {indicator}', linewidth=1.2, color='steelblue')
            axes[i].set_title(f'Model Volatility: X = {indicator}', fontsize=10)
            axes[i].grid(True)

        # Clean up unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Model Volatility Alone (Annualized, No VIX)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # --- C. VIX vs Each Model (Full Overlays, BIG INDIVIDUAL PLOTS) ---
    for indicator in other_indicators:
        dico = stored_data[target][indicator]
        sigma2 = sigma_dict[target].get(indicator)

        if sigma2 is None or np.isnan(sigma2).all():
            print(f"⚠️ Skipping {indicator} — sigma² is invalid.")
            continue

        vol = np.sqrt(sigma2) * np.sqrt(252)
        dates = dico['dates']
        try:
            vix = data.loc[dates, comparison_series].values
        except Exception as e:
            print(f"❌ Could not get VIX for {indicator}: {e}")
            continue

        plt.figure(figsize=(14, 5))
        plt.plot(dates, vol, label='Model Volatility', linewidth=1.5, color='royalblue')
        plt.plot(dates, vix, label='VIX Index', linewidth=1.5, color='darkorange')
        plt.title(f"VIX vs {target} + X = {indicator}", fontsize=13)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_model_vs_vix_export_pdf(data, stored_data, sigma_dict, target='SPX Index', comparison_series='VIX Index', pdf_filename='volatility_vs_vix.pdf'):
    """
    Exports:
    - Big VIX vs SPX + X=None
    - Grid of model-only volatility plots
    - One-by-one large overlay plots (VIX vs each model)
    into a single multi-page PDF.
    """

    if target not in stored_data or target not in sigma_dict:
        print(f"❌ Target '{target}' not found in inputs.")
        return

    indicators = list(stored_data[target].keys())
    indicators = sorted(indicators, key=lambda x: (x != 'No Exogenous', x))  # 'No Exogenous' first

    with PdfPages(pdf_filename) as pdf:

        # --- A. Big Plot: VIX vs X=None ---
        base_result = stored_data[target].get('No Exogenous')
        base_sigma2 = sigma_dict[target].get('No Exogenous')

        if base_result is None or base_sigma2 is None:
            print("❌ Missing 'No Exogenous' model.")
            return

        base_dates = base_result['dates']
        base_vol = np.sqrt(base_sigma2) * np.sqrt(252)
        try:
            base_vix = data.loc[base_dates, comparison_series].values
        except Exception as e:
            print(f"❌ Could not extract VIX for 'No Exogenous': {e}")
            return

        plt.figure(figsize=(14, 6))
        plt.plot(base_dates, base_vol, label='Model Volatility (X = None)', linewidth=1.5, color='royalblue')
        plt.plot(base_dates, base_vix, label='VIX Index', linewidth=1.5, color='darkorange')
        plt.title("VIX vs SPX Index + X = None", fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # --- B. Model-only Volatility Grid (No VIX) ---
        other_indicators = [ind for ind in indicators if ind != 'No Exogenous']
        n = len(other_indicators)
        if n > 0:
            ncols = 2
            nrows = ceil(n / ncols)

            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
            axes = axes.flatten()

            for i, indicator in enumerate(other_indicators):
                dico = stored_data[target][indicator]
                sigma2 = sigma_dict[target].get(indicator)

                if sigma2 is None or np.isnan(sigma2).all():
                    axes[i].set_title(f"{indicator} — Skipped (NaNs)")
                    axes[i].axis('off')
                    continue

                vol = np.sqrt(sigma2) * np.sqrt(252)
                dates = dico['dates']

                axes[i].plot(dates, vol, label=f'X = {indicator}', linewidth=1.2, color='steelblue')
                axes[i].set_title(f'Model Volatility: X = {indicator}', fontsize=10)
                axes[i].grid(True)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle("Model Volatility Alone (Annualized, No VIX)", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

        # --- C. VIX vs Each Model (Big Individual Pages) ---
        for indicator in other_indicators:
            dico = stored_data[target][indicator]
            sigma2 = sigma_dict[target].get(indicator)

            if sigma2 is None or np.isnan(sigma2).all():
                print(f"⚠️ Skipping {indicator} — sigma² invalid.")
                continue

            vol = np.sqrt(sigma2) * np.sqrt(252)
            dates = dico['dates']
            try:
                vix = data.loc[dates, comparison_series].values
            except Exception as e:
                print(f"❌ Could not get VIX for {indicator}: {e}")
                continue

            plt.figure(figsize=(14, 5))
            plt.plot(dates, vol, label='Model Volatility', linewidth=1.5, color='royalblue')
            plt.plot(dates, vix, label='VIX Index', linewidth=1.5, color='darkorange')
            plt.title(f"VIX vs {target} + X = {indicator}", fontsize=13)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"✅ PDF saved to: {pdf_filename}")



def load_selected_columns(filepath, target, indicators):
    """
    Load Excel data and select only the specified target and indicator columns.
    Cleans headers and sets the datetime index.

    Parameters:
        filepath (str): Path to the Excel file.
        target (str): Name of the target column (e.g., 'SPX Index').
        indicators (list): List of additional indicator columns to include.

    Returns:
        pd.DataFrame: Cleaned DataFrame with datetime index and selected columns.
    """
    use_columns = ['Dates'] + [target] + indicators
    df = pd.read_excel(filepath, skiprows=2)
    df = df[use_columns]
    df.columns = ['Date'] + [target] + indicators
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df


# === Function 2 ===
def add_regime_labels(df, output_filename):
    """
    Assigns economic regime labels based on historical periods and saves the labeled DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'Date' column.
        output_filename (str): Path to the output Excel file.

    Returns:
        pd.DataFrame: The same DataFrame with a 'Regime' column added.
    """
    regime_periods = [
        ('1990-01-01', '2000-03-31', 'Stability'),
        ('2000-04-01', '2002-09-30', 'Crisis'),
        ('2002-10-01', '2004-03-31', 'Recovery'),
        ('2004-04-01', '2007-10-31', 'Stability'),
        ('2007-11-01', '2009-03-31', 'Crisis'),
        ('2009-04-01', '2010-12-31', 'Recovery'),
        ('2011-01-01', '2020-02-17', 'Stability'),
        ('2020-02-18', '2020-03-24', 'Crisis'),
        ('2020-03-25', '2021-09-15', 'Recovery'),
        ('2020-09-16', '2021-12-31', 'Stability'),
        ('2022-01-01', '2022-10-12', 'Crisis'),
        ('2022-10-13', '2023-07-31', 'Recovery'),
        ('2023-08-01', '2024-12-31', 'Stability'),
        ('2025-01-01', '2025-02-20', 'Stability'),
        ('2025-02-20', '2025-04-30', 'Crisis'),
    ]

    def assign_regime(date):
        for start, end, regime in regime_periods:
            if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                return regime
        return "Unclassified"

    df['Regime'] = df['Date'].apply(assign_regime)
    df.to_excel(output_filename, index=False)
    return df



def plot_returns_by_regime(filepath, index_name='SPX Index'):
    """
    Plots the (non-log) returns of a given index across time, segmented by economic regimes.
    
    Parameters:
        filepath (str): Path to Excel file with 'Date', the specified index column, and 'Regime'.
        index_name (str): Name of the index to use for returns (e.g., 'SPX Index').
    
    Returns:
        plotly.graph_objects.Figure: The interactive figure object (also shown).
    """
    df = pd.read_excel(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Use raw returns (can be adjusted to log returns if needed)
    df['Return'] = df[index_name]
    df = df.dropna(subset=['Return'])

    # Identify regime shifts
    df['Regime Shift'] = (df['Regime'] != df['Regime'].shift(1)).astype(int).cumsum()
    df['Return Clean'] = df['Return']
    df['Block ID'] = df['Regime'] + '_' + df['Regime Shift'].astype(str)

    fig = px.line(
        df,
        x='Date',
        y='Return Clean',
        color='Regime',
        line_group='Block ID',
        title=f'{index_name} Returns (by Regime)',
        labels={'Return Clean': 'Return'},
        template='plotly_white'
    )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Return',
        legend_title='Regime',
        hovermode='x unified',
        font=dict(size=14)
    )

    fig.show()


def plot_log_returns_by_regime(filepath, index_name='SPX Index'):
    """
    Plots the log returns of a given index over time, segmented by economic regimes.
    Log returns are additive over time and better for comparing volatility regimes.

    Parameters:
        filepath (str): Path to Excel file with 'Date', the specified index column, and 'Regime'.
        index_name (str): Name of the index to use for log return computation (e.g., 'SPX Index').

    Returns:
        plotly.graph_objects.Figure: The interactive figure object (also shown).
    """
    df = pd.read_excel(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Compute log returns: additive and suitable for volatility comparisons
    df['Log Return'] = np.log(df[index_name] / df[index_name].shift(1))
    df = df.dropna(subset=['Log Return'])

    # Identify regime shifts
    df['Regime Shift'] = (df['Regime'] != df['Regime'].shift(1)).astype(int).cumsum()
    df['Log Return Clean'] = df['Log Return']
    df['Block ID'] = df['Regime'] + '_' + df['Regime Shift'].astype(str)

    fig = px.line(
        df,
        x='Date',
        y='Log Return Clean',
        color='Regime',
        line_group='Block ID',
        title=f'{index_name} Log Returns (by Regime)',
        labels={'Log Return Clean': 'Log Return'},
        template='plotly_white'
    )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Log Return',
        legend_title='Regime',
        hovermode='x unified',
        font=dict(size=14)
    )

    fig.show()



def fit_arch1_per_regime(filepath, index_name='SPX Index', regime_periods=None, min_obs=100):
    """
    Fits an ARCH(1) model to each predefined regime period using log returns.

    Parameters:
        filepath (str): Path to the Excel file with 'Date', index column, and 'Regime'.
        index_name (str): Name of the index to use for returns (e.g., 'SPX Index').
        regime_periods (list): List of tuples (start_date, end_date, regime_label).
        min_obs (int): Minimum number of return observations required to fit a model.

    Returns:
        pd.DataFrame: DataFrame containing ARCH(1) parameters per regime.
    """
    df = pd.read_excel(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Return'] = np.log(df[index_name]).diff()

    if regime_periods is None:
        raise ValueError("You must provide regime_periods as a list of (start, end, label)")

    results = []

    for start, end, label in regime_periods:
        mask = (df['Date'] >= pd.Timestamp(start)) & (df['Date'] <= pd.Timestamp(end))
        returns = df.loc[mask, 'Return'].dropna() * 100  # Scale to percentage

        if len(returns) < min_obs:
            print(f"⚠️ Skipping {label} ({start}–{end}) — too few returns: {len(returns)}")
            continue

        am = arch_model(returns, vol='ARCH', p=1, mean='Constant', dist='normal', rescale=False)
        fit = am.fit(disp='off')

        params = fit.params
        print(f"✅ Fitted {label} — alpha0: {params.get('omega'):.4f}, alpha1: {params.get('alpha[1]'):.4f}")

        results.append({
            'Start': start,
            'End': end,
            'Regime': label,
            'mu': params.get('mu', None),
            'alpha0': params.get('omega'),
            'alpha1': params.get('alpha[1]')
        })

    return pd.DataFrame(results)

def analyze_arch_parameters_by_regime(results_df):
    """
    Displays grouped statistics (mean, std) of ARCH(1) parameters by regime,
    and visualizes distributions with boxplots for mu, alpha0, and alpha1.

    Parameters:
        results_df (pd.DataFrame): DataFrame with columns ['mu', 'alpha0', 'alpha1', 'Regime'].

    Returns:
        pd.DataFrame: Grouped statistics by regime.
    """
    filtered_df = results_df.dropna(subset=["mu", "alpha0", "alpha1"])

    # Group stats
    group_stats = filtered_df.groupby("Regime")[["mu", "alpha0", "alpha1"]].agg(["mean", "std"])
    display(group_stats)

    counts = filtered_df.groupby("Regime").size()
    print("Sample sizes used in boxplots per regime:")
    for regime, count in counts.items():
        print(f"   {regime}: {count} models")

    # Boxplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(data=filtered_df, x="Regime", y="mu", ax=axs[0])
    axs[0].set_title("Mean Return (mu) by Regime")
    axs[0].set_ylabel("mu")
    axs[0].grid(True)

    sns.boxplot(data=filtered_df, x="Regime", y="alpha0", ax=axs[1])
    axs[1].set_title("ARCH(1) alpha0 by Regime (Baseline Volatility)")
    axs[1].set_ylabel("alpha0")
    axs[1].grid(True)

    sns.boxplot(data=filtered_df, x="Regime", y="alpha1", ax=axs[2])
    axs[2].set_title("ARCH(1) alpha1 by Regime (Volatility Clustering)")
    axs[2].set_ylabel("alpha1")
    axs[2].grid(True)

    plt.suptitle("ARCH(1) Parameter Distributions by Regime", fontsize=16)
    plt.tight_layout()
    plt.show()

    return group_stats


def plot_arch_volatility_paths_by_regime(filepath, index_name='SPX Index', regime_periods=None, min_obs=100):
    """
    Fits ARCH(1) models per regime using percentage returns and plots the estimated volatility paths.

    Parameters:
        filepath (str): Path to the Excel file containing 'Date', index_name, and 'Regime'.
        index_name (str): Name of the index column (e.g., 'SPX Index').
        regime_periods (list): List of (start_date, end_date, regime_label).
        min_obs (int): Minimum number of observations required to fit a model.

    Returns:
        dict: Dictionary {regime_label: list of (start, end, df)} with volatility columns added.
    """
    df = pd.read_excel(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Return"] = df[index_name].pct_change() * 100  # Percentage return
    df = df.dropna(subset=["Return"])

    if regime_periods is None:
        raise ValueError("regime_periods must be provided.")

    regimes = sorted(set(label for _, _, label in regime_periods))
    regime_groups = {r: [] for r in regimes}

    for start, end, label in regime_periods:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        returns = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()

        if len(returns) < min_obs:
            print(f"⚠️ Skipping {label} {start} to {end} — too few returns: {len(returns)}")
            continue

        am = arch_model(returns["Return"], vol='ARCH', p=1, mean='Constant', dist='normal', rescale=False)
        fit = am.fit(disp='off')
        returns["Volatility"] = fit.conditional_volatility
        regime_groups[label].append((start, end, returns))

    # === Plotting ===
    ncols = 2
    total_plots = sum(len(v) for v in regime_groups.values())
    nrows = (total_plots + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axs = axs.flatten()
    plot_idx = 0

    for regime in regimes:
        for (start, end, returns) in regime_groups[regime]:
            ax = axs[plot_idx]
            ax.plot(returns["Date"], returns["Volatility"], color="darkblue")
            ax.set_title(f"{regime}: {start[:10]} → {end[:10]}")
            ax.set_ylabel("Volatility")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True)
            plot_idx += 1

    # Turn off unused axes
    for i in range(plot_idx, len(axs)):
        axs[i].axis("off")

    plt.suptitle("ARCH(1) Volatility Paths by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return regime_groups


def fit_garch_by_regime(filepath, index_name='SPX Index', regime_periods=None, min_obs=100):
    """
    Fits a GARCH(1,1) model per regime and visualizes parameter distributions.

    Parameters:
        filepath (str): Path to Excel file containing 'Date', index_name, and 'Regime'.
        index_name (str): Column to use for return calculations.
        regime_periods (list): List of (start, end, label) regime tuples.
        min_obs (int): Minimum number of return points required per regime.

    Returns:
        pd.DataFrame: DataFrame with GARCH parameter estimates per regime.
    """
    df = pd.read_excel(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Return'] = df[index_name].pct_change() * 100
    df = df.dropna(subset=['Return'])

    if regime_periods is None:
        raise ValueError("You must provide a list of regime_periods.")

    results = []

    for start, end, label in regime_periods:
        mask = (df['Date'] >= pd.Timestamp(start)) & (df['Date'] <= pd.Timestamp(end))
        returns = df.loc[mask, 'Return'].dropna()

        if len(returns) < min_obs:
            print(f"⚠️ Skipping {label} ({start}–{end}) — too few returns: {len(returns)}")
            continue

        am = arch_model(returns, vol='GARCH', p=1, q=1, mean='Constant', dist='normal', rescale=False)
        fit = am.fit(disp='off')
        params = fit.params

        results.append({
            'Start': start,
            'End': end,
            'Regime': label,
            'mu': params.get('mu', None),
            'omega': params.get('omega', None),
            'alpha1': params.get('alpha[1]', None),
            'beta1': params.get('beta[1]', None),
            'persistence': params.get('alpha[1]', 0) + params.get('beta[1]', 0)
        })

    results_df = pd.DataFrame(results)
    return results_df

def plot_garch_parameter_distributions(results_df):
    """
    Creates boxplots for GARCH parameters (mu, omega, alpha1, persistence) grouped by regime.

    Parameters:
        results_df (pd.DataFrame): DataFrame with GARCH parameters and a 'Regime' column.

    Returns:
        pd.DataFrame: Group statistics (mean/std) per regime.
    """
    results_df = results_df.dropna(subset=['mu', 'omega', 'alpha1', 'persistence'])
    group_stats = results_df.groupby("Regime")[["mu", "omega", "alpha1", "beta1", "persistence"]].agg(["mean", "std"])
    display(group_stats)

    fig, axs = plt.subplots(1, 4, figsize=(22, 6))


    counts = results_df.groupby("Regime").size()
    print("Sample sizes used in boxplots per regime:")
    for regime, count in counts.items():
        print(f"   {regime}: {count} models")


    sns.boxplot(data=results_df, x="Regime", y="mu", ax=axs[0])
    axs[0].set_title("Mean Return (mu) by Regime")
    axs[0].grid(True)

    sns.boxplot(data=results_df, x="Regime", y="omega", ax=axs[1])
    axs[1].set_title("GARCH omega by Regime (Baseline Variance)")
    axs[1].grid(True)

    sns.boxplot(data=results_df, x="Regime", y="alpha1", ax=axs[2])
    axs[2].set_title("GARCH alpha1 by Regime (Shock Impact)")
    axs[2].grid(True)

    sns.boxplot(data=results_df, x="Regime", y="persistence", ax=axs[3])
    axs[3].set_title("GARCH Persistence (α₁ + β₁) by Regime")
    axs[3].grid(True)

    plt.suptitle("GARCH(1,1) Parameter Distributions by Regime", fontsize=16)
    plt.tight_layout()
    plt.show()

    return group_stats


import matplotlib.dates as mdates

def plot_garch_volatility_paths_by_regime(filepath, index_name='SPX Index', regime_periods=None, min_obs=100):
    """
    Fits GARCH(1,1) models per regime using percentage returns and plots volatility paths.

    Parameters:
        filepath (str): Path to Excel file with 'Date', index_name, and 'Regime'.
        index_name (str): Name of the column for return calculation (e.g. 'SPX Index').
        regime_periods (list): List of tuples (start_date, end_date, regime_label).
        min_obs (int): Minimum number of data points per regime to fit GARCH.

    Returns:
        dict: Dictionary {regime: [(start, end, df_with_volatility)]}
    """
    df = pd.read_excel(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Return"] = df[index_name].pct_change() * 100
    df = df.dropna(subset=["Return"])

    if regime_periods is None:
        raise ValueError("You must provide regime_periods.")

    regimes = sorted(set(label for _, _, label in regime_periods))
    regime_groups = {r: [] for r in regimes}

    for start, end, label in regime_periods:
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        returns = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()

        if len(returns) < min_obs:
            print(f"⚠️ Skipping {label} {start} to {end} — too few returns: {len(returns)}")
            continue

        am = arch_model(returns["Return"], vol='GARCH', p=1, q=1, mean='Constant', dist='normal', rescale=False)
        fit = am.fit(disp='off')
        returns["Volatility"] = fit.conditional_volatility
        regime_groups[label].append((start, end, returns))

    # Plot setup
    ncols = 2
    total_plots = sum(len(v) for v in regime_groups.values())
    nrows = (total_plots + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axs = axs.flatten()
    plot_idx = 0

    for regime in regimes:
        for (start, end, returns) in regime_groups[regime]:
            ax = axs[plot_idx]
            ax.plot(returns["Date"], returns["Volatility"], color="darkorange")
            ax.set_title(f"{regime}: {start[:10]} → {end[:10]}")
            ax.set_ylabel("Volatility")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True)
            plot_idx += 1

    # Turn off unused subplots
    for i in range(plot_idx, len(axs)):
        axs[i].axis("off")

    plt.suptitle("GARCH(1,1) Volatility Paths by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return regime_groups



def fit_global_garch_and_add_volatility(filepath, index_name='SPX Index'):
    """
    Loads SPX data, computes returns, fits a global GARCH(1,1) model to the entire sample,
    and appends conditional volatility to the DataFrame.

    Parameters:
        filepath (str): Path to Excel file with 'Date', index_name, and 'Regime'.
        index_name (str): The index used to compute returns (default = 'SPX Index').

    Returns:
        pd.DataFrame: DataFrame with 'Return', 'Volatility', and standardized residuals added.
        arch_model_fit: Fitted GARCH(1,1) model object for diagnostics.
    """
    df = pd.read_excel(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df["Return"] = df[index_name].pct_change() * 100
    df = df.dropna(subset=["Return"])

    am = arch_model(df["Return"], vol="GARCH", p=1, q=1, mean="Constant", dist="normal", rescale=False)
    fit = am.fit(disp="off")

    df["Volatility"] = fit.conditional_volatility
    df["z_t"] = fit.std_resid  # Standardized residuals

    return df, fit


from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

def analyze_standardized_residuals_by_regime(df, residual_column='z_t', regime_column='Regime'):
    """
    For each regime, prints the moments (mean, std, skewness, kurtosis) of the standardized residuals
    and plots a histogram.

    Parameters:
        df (pd.DataFrame): DataFrame containing standardized residuals and regime labels.
        residual_column (str): Name of the standardized residual column (default = 'z_t').
        regime_column (str): Name of the regime label column.

    Returns:
        dict: A dictionary of moment statistics for each regime.
    """
    regimes = sorted(df[regime_column].dropna().unique())
    results = {}

    ncols = 2
    nrows = (len(regimes) + 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axs = axs.flatten()

    print("=== Standardized Residual Moments by Regime ===\n")

    for i, regime in enumerate(regimes):
        subset = df[df[regime_column] == regime][residual_column].dropna()

        mean = np.mean(subset)
        std = np.std(subset)
        skewness = skew(subset)
        kurt = kurtosis(subset, fisher=False)

        print(f"Regime: {regime}")
        print(f"  Mean     : {mean:.4f}")
        print(f"  Std Dev  : {std:.4f}")
        print(f"  Skewness : {skewness:.4f}")
        print(f"  Kurtosis : {kurt:.4f}\n")

        results[regime] = {
            "mean": mean,
            "std": std,
            "skewness": skewness,
            "kurtosis": kurt
        }

        ax = axs[i]
        ax.hist(subset, bins=60, density=True, alpha=0.6, color="steelblue", edgecolor="black")
        ax.set_title(f"Histogram of $z_t$ — {regime}")
        ax.set_xlabel("z_t")
        ax.set_ylabel("Density")
        ax.grid(True)

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.suptitle("Histograms of Standardized GARCH(1,1) Residuals by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return results


#GARCH - regime aware Model

#utlise cette vol pour corrélation entre rendements standardisés (divisé par cette vol)
#regarder aussi la corrélation entre rendements non standardisés

import numpy as np
from scipy.optimize import minimize

# === Log-likelihood for regime-aware GARCH(1,1) ===
def regime_garch_loglik(params, y, regimes_num):
    """
    Computes the negative log-likelihood of a GARCH(1,1) model with regime-dependent parameters.
    """
    n = len(y)
    K = 3  # We hardcode K=3 for Stability, Crisis, Recovery
    omega = params[0:K]
    alpha = params[K:2*K]
    beta = params[2*K:3*K]

    sigma2 = np.zeros_like(y)
    sigma2[0] = np.var(y)

    for t in range(1, n):
        k = regimes_num[t]
        sigma2[t] = omega[k] + alpha[k] * y[t-1]**2 + beta[k] * sigma2[t-1]

    loglik = -0.5 * np.sum(np.log(sigma2) + y**2 / sigma2)
    return -loglik

def extract_garch_params(params, regime_labels):
    K = len(regime_labels)
    omega = params[0:K]
    alpha = params[K:2*K]
    beta = params[2*K:3*K]
    


    result = {}
    for i, k in enumerate(regime_labels):
        result[k] = {
            'omega': omega[i],
            'alpha': alpha[i],
            'beta': beta[i]
        }
    return result

# === Main estimation function ===
def fit_regime_garch(y, regimes, expected_labels=['Stability', 'Crisis', 'Recovery']):
    """
    Fits a GARCH(1,1) model with different parameters per regime.

    Parameters:
        y (array-like): Return series.
        regimes (array-like): Regime labels per observation.
        expected_labels (list): Fixed regime names, ordered [Stability, Crisis, Recovery].

    Returns:
        dict: containing fitted parameters, optimization result, mappings.
    """
    y = np.asarray(y)
    regimes = np.asarray(regimes)

    # Ensure only expected regimes are used
    if not set(np.unique(regimes)).issubset(set(expected_labels)):
        raise ValueError(f"Unexpected regime label(s) found. Only {expected_labels} are allowed.")

    # Fixed mapping to ensure regime consistency
    mapping = {label: i for i, label in enumerate(expected_labels)}
    reverse_mapping = {i: label for label, i in mapping.items()}
    regimes_num = np.array([mapping[r] for r in regimes])

    K = len(expected_labels)
    init_params = np.array([0.1]*K + [0.05]*K + [0.9]*K)
    bounds = [(1e-6, None)] * (3*K)

    def constraint_sum(params):
        alpha = params[K:2*K]
        beta = params[2*K:3*K]
        return 0.999 - (alpha + beta)

    constraints = [{'type': 'ineq', 'fun': lambda p, k=k: constraint_sum(p)[k]} for k in range(K)]

    result = minimize(
        regime_garch_loglik,
        init_params,
        args=(y, regimes_num),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    structured_params = extract_garch_params(result.x, expected_labels)

    return {
        'params': structured_params,
        'opt_result': result,
        'mapping': mapping,
        'reverse_mapping': reverse_mapping
    }


def compute_sigma2_regime(y, regimes_num, params, reverse_mapping):
    """
    Computes conditional variances σ²ₜ using a regime-specific GARCH(1,1) model.

    Parameters:
        y (array-like): Return series.
        regimes_num (array-like): Integer-encoded regime indices (0,1,2).
        params (dict): Dictionary of GARCH parameters by regime name.
        reverse_mapping (dict): Maps regime index to regime name.

    Returns:
        np.ndarray: Array of conditional variances σ²ₜ.
    """
    y = np.asarray(y)
    regimes_num = np.asarray(regimes_num)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.var(y)  # Initialize with unconditional variance

    for t in range(1, len(y)):
        k = regimes_num[t]
        regime = reverse_mapping[k]
        p = params[regime]
        sigma2[t] = p['omega'] + p['alpha'] * y[t-1]**2 + p.get('beta',0) * sigma2[t-1]

    return sigma2


def analyze_regime_garch_residuals(filepath, index_name='SPX Index', expected_labels=['Stability', 'Crisis', 'Recovery']):
    """
    Fits a regime-aware GARCH(1,1) model using 3 fixed regimes, computes standardized residuals,
    then analyzes their distribution (mean, std, skew, kurt) within each regime and plots histograms.

    Parameters:
        filepath (str): Path to Excel file with 'Date', index column, and 'Regime'.
        index_name (str): Name of the price column used to compute returns.
        expected_labels (list): Fixed regime labels (default: Stability, Crisis, Recovery).

    Returns:
        dict: {
            'df': full DataFrame with z_t_regime,
            'params': fitted GARCH parameters per regime,
            'moments': {regime: {mean, std, skewness, kurtosis}}
        }
    """
    # === Load and prepare data ===
    df = pd.read_excel(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["Return"] = df[index_name].pct_change() * 100
    df = df.dropna(subset=["Return"])
    
    # Filter for expected regimes only
    df = df[df["Regime"].isin(expected_labels)].copy()

    # === Fit regime-aware GARCH ===
    fit_result = fit_regime_garch(df["Return"], df["Regime"], expected_labels=expected_labels)
    params = fit_result["params"]
    mapping = fit_result["mapping"]
    reverse_mapping = fit_result["reverse_mapping"]

    y = df["Return"].values
    regimes_num = np.array([mapping[r] for r in df["Regime"]])
    sigma2 = compute_sigma2_regime(y, regimes_num, params, reverse_mapping)
    df["z_t_regime"] = y / np.sqrt(sigma2)

    # === Histogram and moment analysis ===
    regimes = expected_labels  # Force display of all 3 even if sample is small
    ncols = 2
    nrows = (len(regimes) + 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axs = axs.flatten()

    moments = {}
    print("=== Standardized Residual Moments by Regime ===\n")

    for i, regime in enumerate(regimes):
        subset = df[df["Regime"] == regime]["z_t_regime"].dropna()
        if subset.empty:
            print(f"Regime: {regime} — ⚠️ No data available.\n")
            moments[regime] = None
            axs[i].set_title(f"{regime} (no data)")
            axs[i].axis("off")
            continue

        mean = np.mean(subset)
        std = np.std(subset)
        skewness = skew(subset)
        kurt = kurtosis(subset, fisher=False)

        print(f"Regime: {regime}")
        print(f"  Mean     : {mean:.4f}")
        print(f"  Std Dev  : {std:.4f}")
        print(f"  Skewness : {skewness:.4f}")
        print(f"  Kurtosis : {kurt:.4f}\n")

        moments[regime] = {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurt
        }

        axs[i].hist(subset, bins=60, density=True, alpha=0.6, color="steelblue", edgecolor="black")
        axs[i].set_title(f"Histogram of $z_t$ — {regime}")
        axs[i].set_xlabel("z_t")
        axs[i].set_ylabel("Density")
        axs[i].grid(True)

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.suptitle("Histograms of Standardized Residuals (Regime-Aware GARCH)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return {
        "df": df,
        "params": params,
        "moments": moments
    }


def regime_correlation_analysis(df, col1, col2, regime_col="Regime", K=3):
    """
    Computes correlation between two return series (raw and standardized) by regime using regime-aware GARCH volatility.

    Parameters:
    - df: DataFrame with return columns and regime label
    - col1, col2: str, names of return columns (e.g., "SPX Return", "CAC Return")
    - regime_col: name of the regime column (default "Regime")
    - K: number of regimes (default 3)

    Returns:
    - DataFrame with raw and standardized correlations by regime
    """
    
    y1 = df[col1].values
    y2 = df[col2].values
    regimes = df[regime_col].values

    # Fit GARCH to each return series
    fit1 = fit_regime_garch(y1, regimes)
    fit2 = fit_regime_garch(y2, regimes)

    map1 = fit1["mapping"]
    map2 = fit2["mapping"]
    rev1 = fit1["reverse_mapping"]
    rev2 = fit2["reverse_mapping"]

    reg1_num = np.array([map1[r] for r in regimes])
    reg2_num = np.array([map2[r] for r in regimes])

    sigma2_1 = compute_sigma2_regime(y1, reg1_num, fit1["params"], rev1)
    sigma2_2 = compute_sigma2_regime(y2, reg2_num, fit2["params"], rev2)

    df = df.copy()
    df["z1"] = y1 / np.sqrt(sigma2_1)
    df["z2"] = y2 / np.sqrt(sigma2_2)

    results = []
    for regime in sorted(np.unique(regimes)):
        sub = df[df[regime_col] == regime]
        raw_corr = sub[col1].corr(sub[col2])
        std_corr = sub["z1"].corr(sub["z2"])
        results.append((regime, raw_corr, std_corr))

    return pd.DataFrame(results, columns=["Regime", f"{col1} vs {col2} Raw Corr", f"{col1} vs {col2} Std Corr"])


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "notebook"

def plot_rolling_correlation_plotly_regime_aware(df, target="SPX", index="S5MATR",
                                                 rolling_window=40, log_returns=True, standardize=True):
    """
    Plot rolling correlation between two series using regime-colored Plotly chart,
    optionally standardizing returns via regime-aware GARCH(1,1).

    Parameters:
        df (pd.DataFrame): Must contain 'Date' and 'Regime' columns + raw prices for target and index.
        target (str): Ticker of the anchor index (e.g. 'SPX').
        index (str): Ticker of the compared index/sector (e.g. 'S5MATR').
        rolling_window (int): Size of rolling correlation window (in days).
        log_returns (bool): If True, use log returns; else, use percentage returns.
        standardize (bool): If True, standardize returns using regime-aware GARCH volatility.
    """
    df_plot = df.copy()

    # Compute returns
    if log_returns:
        df_plot[f"{target}_ret"] = np.log(df_plot[target]).diff() * 100
        df_plot[f"{index}_ret"] = np.log(df_plot[index]).diff() * 100
    else:
        df_plot[f"{target}_ret"] = df_plot[target].pct_change() * 100
        df_plot[f"{index}_ret"] = df_plot[index].pct_change() * 100

    df_plot = df_plot.dropna(subset=[f"{target}_ret", f"{index}_ret", "Regime", "Date"]).copy()

    # === Standardize using regime-aware GARCH volatility ===
    if standardize:
        y1 = df_plot[f"{target}_ret"].values
        y2 = df_plot[f"{index}_ret"].values
        regimes = df_plot["Regime"].values

        # Fit and compute volatility
        fit1 = fit_regime_garch(y1, regimes)
        fit2 = fit_regime_garch(y2, regimes)

        reg1 = np.array([fit1["mapping"][r] for r in regimes])
        reg2 = np.array([fit2["mapping"][r] for r in regimes])

        sigma2_1 = compute_sigma2_regime(y1, reg1, fit1["params"], fit1["reverse_mapping"])
        sigma2_2 = compute_sigma2_regime(y2, reg2, fit2["params"], fit2["reverse_mapping"])

        df_plot[f"{target}_ret"] = y1 / np.sqrt(sigma2_1)
        df_plot[f"{index}_ret"] = y2 / np.sqrt(sigma2_2)

    # === Compute rolling correlation ===
    df_plot["RollingCorr"] = (
        df_plot[f"{target}_ret"]
        .rolling(window=rolling_window, min_periods=5)
        .corr(df_plot[f"{index}_ret"])
    )

    # Insert NaNs at regime changes
    df_plot["Regime Shift"] = (df_plot["Regime"] != df_plot["Regime"].shift(1)).astype(int).cumsum()
    df_plot["RollingCorrClean"] = df_plot["RollingCorr"]
    df_plot["Block ID"] = df_plot["Regime"] + "_" + df_plot["Regime Shift"].astype(str)

    # === Plot ===
    fig = px.line(
        df_plot,
        x="Date",
        y="RollingCorrClean",
        color="Regime",
        line_group="Block ID",
        title=f"{rolling_window}-Day Rolling Correlation: {target} vs {index}",
        labels={"RollingCorrClean": "Rolling Correlation"},
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Correlation",
        legend_title="Regime",
        hovermode="x unified",
        font=dict(size=14)
    )

    fig.show()



import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns

def fit_arch1_all_indices(filepath, regime_periods, min_obs=100, date_col='Dates', sheet_name='Data'):
    """
    Fits ARCH(1) models per regime for all indices in the Excel file and pools results.

    Parameters:
        filepath (str): Path to the Excel file.
        regime_periods (list): List of tuples (start_date, end_date, regime_label).
        min_obs (int): Minimum number of returns per regime to fit ARCH(1).
        date_col (str): Name of the date column.
        sheet_name (str): Sheet containing the data.

    Returns:
        pd.DataFrame: Combined DataFrame of ARCH(1) parameters with regime and index labels.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    all_results = []
    index_columns = [col for col in df.columns if col != date_col]

    for index in index_columns:
        df['Return'] = np.log(df[index]).diff()

        for start, end, label in regime_periods:
            mask = (df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))
            returns = df.loc[mask, 'Return'].dropna() * 100  # scale to percentage

            if len(returns) < min_obs:
                print(f"⚠️ Skipping {index} – {label} ({start}–{end}): too few returns ({len(returns)})")
                continue

            try:
                am = arch_model(returns, vol='ARCH', p=1, mean='Constant', dist='normal', rescale=False)
                fit = am.fit(disp='off')
                params = fit.params

                print(f"✅ {index} — {label} — alpha0: {params.get('omega'):.4f}, alpha1: {params.get('alpha[1]'):.4f}")

                all_results.append({
                    'Index': index,
                    'Start': start,
                    'End': end,
                    'Regime': label,
                    'mu': params.get('mu', None),
                    'alpha0': params.get('omega'),
                    'alpha1': params.get('alpha[1]')
                })
            except Exception as e:
                print(f"❌ Failed to fit {index} — {label}: {e}")
                continue

    return pd.DataFrame(all_results)

def analyze_arch_parameters_by_regime(results_df):
    """
    Displays grouped statistics and boxplots of ARCH(1) parameters by regime.

    Parameters:
        results_df (pd.DataFrame): DataFrame with ['Index', 'mu', 'alpha0', 'alpha1', 'Regime'].

    Returns:
        pd.DataFrame: Grouped statistics by regime.
    """

    

    filtered_df = results_df.dropna(subset=["mu", "alpha0", "alpha1"])

    # Group stats
    group_stats = filtered_df.groupby("Regime")[["mu", "alpha0", "alpha1"]].agg(["mean", "std"])
    display(group_stats)


    #Non outsiders
    counts = filtered_df.groupby("Regime").size()
    print("Sample sizes used in boxplots per regime:")
    for regime, count in counts.items():
        print(f"   {regime}: {count} models")

    # Boxplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(data=filtered_df, x="Regime", y="mu", ax=axs[0])
    axs[0].set_title("Mean Return (mu) by Regime")
    axs[0].set_ylabel("mu")
    axs[0].grid(True)

    sns.boxplot(data=filtered_df, x="Regime", y="alpha0", ax=axs[1])
    axs[1].set_title("ARCH(1) alpha0 by Regime (Baseline Volatility)")
    axs[1].set_ylabel("alpha0")
    axs[1].grid(True)

    sns.boxplot(data=filtered_df, x="Regime", y="alpha1", ax=axs[2])
    axs[2].set_title("ARCH(1) alpha1 by Regime (Volatility Clustering)")
    axs[2].set_ylabel("alpha1")
    axs[2].grid(True)

    plt.suptitle("ARCH(1) Parameter Distributions by Regime (All Indices)", fontsize=16)
    plt.tight_layout()
    plt.show()

    return group_stats

def analyze_arch_parameters_by_regime_swarm(results_df):
    """
    Creates swarmplots and summary statistics for ARCH(1) parameters grouped by regime.

    Parameters:
        results_df (pd.DataFrame): Must include ['Index', 'mu', 'alpha0', 'alpha1', 'Regime'].

    Returns:
        pd.DataFrame: Grouped statistics (mean, std) per regime.
    """
    filtered_df = results_df.dropna(subset=["mu", "alpha0", "alpha1"])

    # Sample count per regime
    counts = filtered_df.groupby("Regime").size()
    print("📊 Sample sizes used in ARCH swarmplots:")
    for regime, count in counts.items():
        print(f"   {regime}: {count} models")

    # Group stats
    group_stats = filtered_df.groupby("Regime")[["mu", "alpha0", "alpha1"]].agg(["mean", "std"])
    display(group_stats)

    # Swarmplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    sns.swarmplot(data=filtered_df, x="Regime", y="mu", ax=axs[0])
    axs[0].set_title("Mean Return (mu) by Regime")
    axs[0].grid(True)

    sns.swarmplot(data=filtered_df, x="Regime", y="alpha0", ax=axs[1])
    axs[1].set_title("ARCH alpha0 by Regime (Baseline Volatility)")
    axs[1].grid(True)

    sns.swarmplot(data=filtered_df, x="Regime", y="alpha1", ax=axs[2])
    axs[2].set_title("ARCH alpha1 by Regime (Volatility Clustering)")
    axs[2].grid(True)

    plt.suptitle("ARCH(1) Parameter Swarmplots by Regime (All Indices)", fontsize=16)
    plt.tight_layout()
    plt.show()

    return group_stats



def fit_garch_all_indices(filepath, regime_periods, min_obs=100, date_col='Dates', sheet_name='Data'):
    """
    Fits GARCH(1,1) models per regime for all indices in Excel file, returns pooled DataFrame.

    Parameters:
        filepath (str): Path to Excel file.
        regime_periods (list): List of (start, end, label) regime tuples.
        min_obs (int): Minimum number of returns required.
        date_col (str): Column name for dates.
        sheet_name (str): Excel sheet containing the data.

    Returns:
        pd.DataFrame: Parameters per regime and index.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    index_columns = [col for col in df.columns if col != date_col]
    results = []

    for index in index_columns:
        df['Return'] = df[index].pct_change() * 100
        df = df.dropna(subset=['Return'])

        for start, end, label in regime_periods:
            mask = (df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))
            returns = df.loc[mask, 'Return'].dropna()

            if len(returns) < min_obs:
                print(f"⚠️ Skipping {index} – {label} ({start}–{end}): too few returns ({len(returns)})")
                continue

            try:
                am = arch_model(returns, vol='GARCH', p=1, q=1, mean='Constant', dist='normal', rescale=False)
                fit = am.fit(disp='off')
                params = fit.params

                results.append({
                    'Index': index,
                    'Start': start,
                    'End': end,
                    'Regime': label,
                    'mu': params.get('mu'),
                    'omega': params.get('omega'),
                    'alpha1': params.get('alpha[1]'),
                    'beta1': params.get('beta[1]'),
                    'persistence': params.get('alpha[1]', 0) + params.get('beta[1]', 0)
                })

                print(f"✅ {index} — {label} — α₁: {params.get('alpha[1]'):.4f}, β₁: {params.get('beta[1]'):.4f}")
            except Exception as e:
                print(f"❌ Failed to fit {index} — {label}: {e}")
                continue

    return pd.DataFrame(results)

def plot_garch_parameter_distributions(results_df):
    """
    Creates boxplots for GARCH(1,1) parameters by regime.

    Parameters:
        results_df (pd.DataFrame): Must include ['Regime', 'mu', 'omega', 'alpha1', 'beta1', 'persistence'].
    """
    filtered_df = results_df.dropna(subset=['mu', 'omega', 'alpha1', 'beta1', 'persistence'])

    # Sample count per regime
    counts = filtered_df.groupby("Regime").size()
    print("Sample sizes used in GARCH boxplots:")
    for regime, count in counts.items():
        print(f"   {regime}: {count} models")

    # Group stats
    group_stats = filtered_df.groupby("Regime")[["mu", "omega", "alpha1", "beta1", "persistence"]].agg(["mean", "std"])
    display(group_stats)

    # Plot
    fig, axs = plt.subplots(1, 4, figsize=(22, 6))

    sns.boxplot(data=filtered_df, x="Regime", y="mu", ax=axs[0])
    axs[0].set_title("Mean Return (mu) by Regime")
    axs[0].grid(True)

    sns.boxplot(data=filtered_df, x="Regime", y="omega", ax=axs[1])
    axs[1].set_title("GARCH omega by Regime (Baseline Variance)")
    axs[1].grid(True)

    sns.boxplot(data=filtered_df, x="Regime", y="alpha1", ax=axs[2])
    axs[2].set_title("GARCH alpha1 by Regime (Shock Impact)")
    axs[2].grid(True)

    sns.boxplot(data=filtered_df, x="Regime", y="persistence", ax=axs[3])
    axs[3].set_title("GARCH Persistence (α₁ + β₁) by Regime")
    axs[3].grid(True)

    plt.suptitle("GARCH(1,1) Parameter Distributions by Regime (All Indices)", fontsize=16)
    plt.tight_layout()
    plt.show()

    return group_stats

def plot_garch_parameter_distributions_swarm(results_df):
    """
    Creates swarmplots for GARCH(1,1) parameters by regime.

    Parameters:
        results_df (pd.DataFrame): Must include ['Regime', 'mu', 'omega', 'alpha1', 'beta1', 'persistence'].
    """
    filtered_df = results_df.dropna(subset=['mu', 'omega', 'alpha1', 'beta1', 'persistence'])

    # Sample count per regime
    counts = filtered_df.groupby("Regime").size()
    print("📊 Sample sizes used in GARCH swarmplots:")
    for regime, count in counts.items():
        print(f"   {regime}: {count} models")

    # Group stats
    group_stats = filtered_df.groupby("Regime")[["mu", "omega", "alpha1", "beta1", "persistence"]].agg(["mean", "std"])
    display(group_stats)

    # Plot
    fig, axs = plt.subplots(1, 5, figsize=(22, 6))

    sns.swarmplot(data=filtered_df, x="Regime", y="mu", ax=axs[0])
    axs[0].set_title("Mean Return (mu) by Regime")
    axs[0].grid(True)

    sns.swarmplot(data=filtered_df, x="Regime", y="omega", ax=axs[1])
    axs[1].set_title("GARCH omega by Regime (Baseline Variance)")
    axs[1].grid(True)

    sns.swarmplot(data=filtered_df, x="Regime", y="alpha1", ax=axs[2])
    axs[2].set_title("GARCH alpha1 by Regime (Shock Impact)")
    axs[2].grid(True)

    sns.swarmplot(data=filtered_df, x="Regime", y="beta1", ax=axs[3])
    axs[3].set_title("GARCH β₁ by Regime")
    axs[3].grid(True)

    sns.swarmplot(data=filtered_df, x="Regime", y="persistence", ax=axs[4])
    axs[4].set_title("GARCH Persistence (α₁ + β₁) by Regime")
    axs[4].grid(True)

    plt.suptitle("GARCH(1,1) Parameter Swarmplots by Regime (All Indices)", fontsize=16)
    plt.tight_layout()
    plt.show()

    return group_stats


def compute_per_period_garch_vols(df, sectors, regime_periods, date_col="Date"):
    """
    For each sector and regime period, fits a separate GARCH(1,1) model on log returns
    and reconstructs the conditional volatility (σ). These volatilities are used to 
    standardize the returns within each period, yielding a 'z-score' series.

    Parameters:
        df (pd.DataFrame): Original DataFrame with price and regime info.
        sectors (list): List of sector column names (e.g. ['SPX', 'S5CONS', ...]).
        regime_periods (list): List of (start_date, end_date, regime_label) tuples.
        date_col (str): Name of the date column in df.

    Returns:
        pd.DataFrame: Modified DataFrame with added columns:
                      - {sector}_ret: log return
                      - {sector}_sigma: conditional volatility
                      - {sector}_z: standardized return
    """
    from arch import arch_model

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Compute log returns for all sectors
    for col in sectors:
        df[f"{col}_ret"] = np.log(df[col]).diff() * 100

    df = df.dropna().copy()

    # Initialize empty σ containers
    for col in sectors:
        df[f"{col}_sigma"] = np.nan

    # Fit GARCH(1,1) for each regime period and sector
    for start, end, label in regime_periods:
        mask = (df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))

        for col in sectors:
            returns = df.loc[mask, f"{col}_ret"].dropna()
            if len(returns) < 100:
                print(f"⚠️ Skipping {col} — {label} ({start}–{end}): too few returns ({len(returns)})")
                continue

            try:
                am = arch_model(returns, vol="GARCH", p=1, q=1, mean="Constant", dist="normal", rescale=False)
                fit = am.fit(disp="off")
                cond_vol = fit.conditional_volatility

                # Align back to full df by index
                df.loc[cond_vol.index, f"{col}_sigma"] = cond_vol.values
                print(f"✅ Fitted {col} — {label}: α₁={fit.params['alpha[1]']:.4f}, β₁={fit.params['beta[1]']:.4f}")
            except Exception as e:
                print(f"❌ GARCH failed for {col} — {label}: {e}")

    # Standardize returns
    for col in sectors:
        df[f"{col}_z"] = df[f"{col}_ret"] / df[f"{col}_sigma"]

    return df

def compute_correlations_by_regime(df, sectors, regimes):
    """
    Computes raw and GARCH-standardized return correlation matrices for each regime.
    Assumes standardized returns are already computed in columns {sector}_z.

    Parameters:
        df (pd.DataFrame): DataFrame containing {sector}_ret, {sector}_z, and 'Regime'.
        sectors (list): List of sector tickers (e.g. ['SPX', 'S5CONS', ...]).
        regimes (list): List of regime labels (e.g. ['Crisis', 'Recovery', 'Stability']).

    Returns:
        dict: raw_corr[regime] = raw return correlation matrix
        dict: std_corr[regime] = standardized return correlation matrix
    """
    raw_corr = {}
    std_corr = {}

    for regime in regimes:
        print(f"\n==== {regime.upper()} ====\n")
        sub = df[df["Regime"] == regime].copy()

        # Raw return correlation matrix
        raw = sub[[f"{col}_ret" for col in sectors]].corr()
        raw.index = raw.columns = sectors
        raw_corr[regime] = raw

        # Standardized return correlation matrix (per-period GARCH-based)
        std = sub[[f"{col}_z" for col in sectors]].corr()
        std.index = std.columns = sectors
        std_corr[regime] = std

        # Plot raw correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(raw, square=True, annot=True, cmap='crest', fmt=".2f")
        plt.title(f"{regime} — Raw Correlation")
        plt.tight_layout()
        plt.show()

        # Plot standardized correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(std, square=True, annot=True, cmap='crest', fmt=".2f")
        plt.title(f"{regime} — GARCH Standardized Correlation")
        plt.tight_layout()
        plt.show()

    return raw_corr, std_corr


from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def single_lasso_trial(T=5000, k=10, true_nonzero_idxs=[2, 6], gamma_vals=[0.8, 0.5], 
                       phi=0.8, omega=0.1, alpha=0.2, random_seed=None):
    """
    Run one simulation and LASSO trial on an ARCH-X model.
    
    Parameters:
    - T (int): Number of time steps.
    - k (int): Total number of exogenous variables.
    - true_nonzero_idxs (list): Indices of exogenous variables that actually affect volatility.
    - gamma_vals (list): Coefficients for the true active exogenous variables.
    - phi (float): AR(1) parameter for exogenous variables.
    - omega, alpha (float): ARCH parameters.
    - random_seed (int): Random seed for reproducibility.
    
    Returns:
    - dict containing selected variables, true variables, success flag, coefficients, and lambda.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Step 1: Define true gamma vector
    gamma_values = np.zeros(k)
    gamma_values[true_nonzero_idxs[0]] = gamma_vals[0]
    gamma_values[true_nonzero_idxs[1]] = gamma_vals[1]

    # Step 2: Simulate AR(1) exogenous variables
    X = np.zeros((T, k))
    for j in range(k):
        for t in range(1, T):
            X[t, j] = phi * X[t - 1, j] + np.random.normal(0, 1)

    # Step 3: Generate ARCH-X returns
    sigma_squared = np.zeros(T)
    returns = np.zeros(T)
    returns[0] = 0
    
    for t in range(1, T):
        sigma_squared[t] = omega + alpha * returns[t - 1]**2 + np.dot(gamma_values, X[t, :])
        sigma_squared[t] = np.abs(sigma_squared[t])
        returns[t] = np.random.normal(0, np.sqrt(sigma_squared[t]))

    # Step 4: Standardize exogenous variables
    X_exo = X[1:, :]  # Drop first row to align with y_lasso
    scaler = StandardScaler()
    X_exo_scaled = scaler.fit_transform(X_exo)

    # Step 5: Build LASSO design matrix and target
    r_squared_lag = returns[:-1]**2
    X_lasso = np.column_stack([r_squared_lag, X_exo_scaled])
    y_lasso = sigma_squared[1:]

    # Step 6: Run LASSO path
    alphas, coefs, _ = lasso_path(X_lasso, y_lasso)
    exogenous_coefs = coefs[1:, :]  # Exclude r_squared_lag from selection

    selected_lambda, selected_variables = None, None
    for i in range(len(alphas)):
        nonzero_indices = np.where(exogenous_coefs[:, i] != 0)[0]
        if len(nonzero_indices) == 2:
            selected_lambda = alphas[i]
            selected_variables = nonzero_indices
            break

    success = set(selected_variables) == set(true_nonzero_idxs) if selected_variables is not None else False
    selected_coefs = (
        exogenous_coefs[:, np.where(alphas == selected_lambda)[0][0]][selected_variables]
        if selected_variables is not None else None
    )

    return {
        'selected_lambda': selected_lambda,
        'selected_variables': selected_variables,
        'selected_coefficients': selected_coefs,
        'true_active_variables': true_nonzero_idxs,
        'success': success
    }


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