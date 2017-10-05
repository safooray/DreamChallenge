"""Fits a proportional hazards model to data after PCA dimentionality reduction.
PH model is regularized by elastic net.
"""
import pandas as pd
import numpy as np
import statsmodels.duration.hazard_regression as hreg
import sklearn.decomposition as decomposition
import evaluation
import scipy.stats as stats


def fit_cox_elasticnet(ma_gene, event, survival, alpha, l1_wt):

    # Fits elastic-net regularized Cox regression to training set.
    Phreg = hreg.PHReg(survival, ma_gene, event)
    PhregResults = Phreg.fit_regularized(alpha=alpha, L1_wt=l1_wt)
    np.save('params.npy', PhregResults.params)

    risk_train = np.exp(np.dot(ma_gene, PhregResults.params))
    # Calculates c-index.
    c_index = evaluation.c_index(risk_train, survival, 1 - event)
    return c_index


if __name__ == '__main__':
    ma_gene_df = pd.read_pickle(path='Integrated_MA_Gene.df')
    ma_clinical_df = pd.read_pickle(path='MA_Gene_Clinical.df')

    ma_gene_highdim = stats.zscore(ma_gene_df.transpose().as_matrix())
    survival = ma_clinical_df.D_OS.values
    event = ma_clinical_df.D_OS_FLAG.values

    pca = decomposition.PCA(n_components=80)
    ma_gene = pca.fit_transform(ma_gene_highdim)

    result = fit_cox_elasticnet(ma_gene, event, survival, 0.1, 0.5)

    print result
