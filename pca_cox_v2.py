"""Fits a proportional hazards model to data after PCA dimentionality reduction.
PCA transformation is learned from all data including testing data.
"""
import pandas as pd
import numpy as np
import statsmodels.duration.hazard_regression as hreg
import sklearn.decomposition as decomposition  
import survivalnet
import scipy.stats as stats


SPLIT = 888
N_COMPS = [5, 10, 15, 20, 30, 60, 120, 250]
N_SHUFFLES = 5

ma_gene_df = pd.read_pickle(path='Integrated_MA_Gene.df')
ma_clinical_df = pd.read_pickle(path='MA_Gene_Clinical.df')

ma_gene = stats.zscore(ma_gene_df.transpose().as_matrix())
survival = ma_clinical_df.D_OS.values
event = ma_clinical_df.D_OS_FLAG.values

results = np.zeros(len(N_COMPS))

for j, n_comp in enumerate(N_COMPS):
	# Sets random generator seed for reproducability.
	np.random.seed(111)
	# Reduces dimentionality to n_comp using PCA on all samples.
	pca = decomposition.PCA(n_components=n_comp)
	ma_gene_lowdim = pca.fit_transform(ma_gene)

	for _ in range(N_SHUFFLES):
		order = np.random.permutation(event.shape[0])
		ma_gene_lowdim = ma_gene_lowdim[order]
		event = event[order]
		survival = survival[order]

		ma_gene_lowdim_train, ma_gene_lowdim_val = np.split(ma_gene_lowdim,
															[SPLIT])
		survival_train, survival_val = np.split(survival, [SPLIT])
		event_train, event_val = np.split(event, [SPLIT])

		# Fits elastic-net regularized Cox regression to training set.
		Phreg = hreg.PHReg(survival_train, ma_gene_lowdim_train, event_train) 
		PhregResults = Phreg.fit_regularized(alpha=0.5)
		# Calulates risk for validation set.
		risk_val = np.exp(np.dot(ma_gene_lowdim_val, PhregResults.params))

		# Calculates c-index.
		SA = survivalnet.optimization.SurvivalAnalysis()
		c_index = SA.c_index(risk_val, survival_val, 1 - event_val)

		results[j] += c_index
		print "n_comp: {}, c_index: {}".format(n_comp, c_index) 

# Calculates average c_index over N_SHUFFLES randomization.
results /= N_SHUFFLES
print results
