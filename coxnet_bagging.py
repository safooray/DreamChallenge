"""Fits an elasticnet regularizd proportional hazards model to data."""
import pandas as pd
import numpy as np
import statsmodels.duration.hazard_regression as hreg
import evaluation
import scipy.stats as stats


SPLIT = 888
N_SHUFFLES = 5
N_MODELS = 10
N_FEATS = 100

ma_gene_df = pd.read_pickle(path='Integrated_MA_Gene.df')
ma_clinical_df = pd.read_pickle(path='MA_Gene_Clinical.df')

ma_gene = stats.zscore(ma_gene_df.transpose().as_matrix())
survival = ma_clinical_df.D_OS.values
event = ma_clinical_df.D_OS_FLAG.values


# Sets random generator seed for reproducability.
np.random.seed(111)

for _ in range(N_MODELS):
    feature_ids = np.random.permutation(ma_gene.shape[1])
    x = ma_gene[:, feature_ids[:N_FEATS]]
    assert x.shape == (ma_gene.shape[0], N_FEATS)

    result = 0
    for _ in range(N_SHUFFLES):
        order = np.random.permutation(x.shape[0])
        x = x[order]
        event = event[order]
        survival = survival[order]

        ma_gene_train, ma_gene_val = np.split(x, [SPLIT])
        survival_train, survival_val = np.split(survival, [SPLIT])
        event_train, event_val = np.split(event, [SPLIT])

        # Fits elastic-net regularized Cox regression to training set.
        Phreg = hreg.PHReg(survival_train, ma_gene_train, event_train)
        PhregResults = Phreg.fit_regularized(alpha=0.1, L1_wt=0.5)

        # Calulates risk for validation set.
        risk_val = np.exp(np.dot(ma_gene_val, PhregResults.params))

        # Calculates c-index.
        c_index = evaluation.c_index(risk_val, survival_val, 1 - event_val)

        result += c_index
        print("c_index: {}".format(c_index))

    # Calculates average c_index over N_SHUFFLES randomization.
    print(result / N_SHUFFLES)
