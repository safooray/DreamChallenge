"""Fits a proportional hazards model to data after PCA dimentionality reduction.
PCA transformation is learned from training data and applied to testing data.
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

# Sets random generator seed for reproducability.
np.random.seed(111)
results = np.zeros(len(N_COMPS))

for _ in range(N_SHUFFLES):
	order = np.random.permutation(event.shape[0])
	ma_gene = ma_gene[order]
	event = event[order]
	survival = survival[order]

	ma_gene_train, ma_gene_val = np.split(ma_gene, [SPLIT])
	survival_train, survival_val = np.split(survival, [SPLIT])
	event_train, event_val = np.split(event, [SPLIT])

	for j, n_comp in enumerate(N_COMPS):
		
		# Reduces dimentionality to n_comp using PCA.
		# Learns components from training set and applies the trasformation to 
		# the validation set.
		pca = decomposition.PCA(n_components=n_comp)
		ma_gene_lowdim_train = pca.fit_transform(ma_gene_train)
		ma_gene_lowdim_val = np.dot(ma_gene_val, pca.components_.transpose())

		# Fits elastic-net regularized Cox regression to training set.
		Phreg = hreg.PHReg(survival[:SPLIT], ma_gene_lowdim_train, event[:SPLIT]) 
		PhregResults = Phreg.fit_regularized(alpha=0.5)
		# Calulates risk for validation set.
		risk_val = np.exp(np.dot(ma_gene_lowdim_val, PhregResults.params))

		# Calculates c-index.
		SA = survivalnet.optimization.SurvivalAnalysis()
		c_index = SA.c_index(risk_val, survival[SPLIT:], 1 - event[SPLIT:])

		results[j] += c_index
		print "n_comp: {}, c_index: {}".format(n_comp, c_index) 

# Calculates average c_index over N_SHUFFLES randomization.
results /= N_SHUFFLES
print results
