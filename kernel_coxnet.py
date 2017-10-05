"""Fits an elasticnet regularizd proportional hazards model to data."""
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.decomposition as decomposition
from sklearn.kernel_approximation import RBFSampler
import statsmodels.duration.hazard_regression as hreg
import survivalnet
import random


SPLIT = 888
N_SHUFFLES = 10


def verify_transformation(lifted_x, x, gamma):
    test_inds = [random.randrange(0, lifted_x.shape[0]) for _ in range(10)]
    gaussian_kernel = lambda x,y,gamma: np.exp(-np.sum((x-y)**2)*gamma)

    for ind in test_inds:
        dot_prod = np.dot(lifted_x[ind], lifted_x[ind+1])
        kernel_value = gaussian_kernel(x[ind], x[ind+1], gamma)
        np.testing.assert_almost_equal(dot_prod, kernel_value, 1)


def run(gamma, n_comp, rbf_feature_size=240):
    ma_gene_df = pd.read_pickle(path='Integrated_MA_Gene.df')
    ma_clinical_df = pd.read_pickle(path='MA_Gene_Clinical.df')
    
    ma_gene_data = ma_gene_df.transpose().as_matrix()
    survival = ma_clinical_df.D_OS.values
    event = ma_clinical_df.D_OS_FLAG.values

    result = 0

    # Apply PCA.
    pca = decomposition.PCA(n_components=n_comp, whiten=True)
    ma_gene_normal = pca.fit_transform(ma_gene_data)
    
    # Extract random Fourier features.
    rbf_feature = RBFSampler(gamma=gamma, n_components=rbf_feature_size, 
                             random_state=1)
    ma_gene = rbf_feature.fit_transform(ma_gene_normal)
    verify_transformation(ma_gene, ma_gene_normal, gamma)

    # Sets random generator seed for reproducability.
    np.random.seed(111)

    for _ in range(N_SHUFFLES):
        order = np.random.permutation(ma_gene.shape[0])
        ma_gene = ma_gene[order]
        event = event[order]
        survival = survival[order]

        ma_gene_train, ma_gene_val = np.split(ma_gene, [SPLIT])
        survival_train, survival_val = np.split(survival, [SPLIT])
        event_train, event_val = np.split(event, [SPLIT])

        # Fits elastic-net regularized Cox regression to training set.
        Phreg = hreg.PHReg(survival_train, ma_gene_train, event_train)
        #PhregResults = Phreg.fit_regularized(alpha=0.1, L1_wt=0.5)
        PhregResults = Phreg.fit()
        # Calulates risk for validation set.
        risk_val = np.exp(np.dot(ma_gene_val, PhregResults.params))
        # Calculates c-index.
        SA = survivalnet.optimization.SurvivalAnalysis()
        c_index = SA.c_index(risk_val, survival_val, 1 - event_val)

        result += c_index
        print "c_index: {}".format(c_index)

    # Calculates average c_index over N_SHUFFLES randomization.
    result /= N_SHUFFLES
    print result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--n_comp', dest='n_comp', default=80, type=int,
                        help='Number of components to keep after PCA.')
    parser.add_argument('-g', '--gamma', dest='gamma', default=1, type=float,
                        help='Gamma for the Gaussian kernel.')
    args = parser.parse_args()
    run(args.gamma, args.n_comp)
