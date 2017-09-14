import pandas as pd
import os
import math

root_dir = '/home/syouse3/DreamMM/'

clinical_df = pd.read_csv(os.path.join(root_dir, 'Clinical/globalClinTraining.csv')) 
data_files = set(clinical_df['MA_geneLevelExpFile'])
#data_files = set(clinical_df['MA_probeLevelExpFile'])

df_list = []
expression_dir = os.path.join(root_dir, 'Expression/Microarray Data')
for ef in data_files:
	if not str(ef) == 'nan':
		df_list.append(pd.read_csv(os.path.join(expression_dir, ef)))
		print(ef)

#First column of the data frame contains gene names in Entrez encoding.
ma_genes = set(df_list[0].ix[:,0])
for df in df_list:
	print df.columns
	ma_genes = ma_genes.intersection(df.ix[:,0])

for i in range(len(df_list)):  
    df_list[i] = df_list[i].set_index(df_list[i].columns[0])

integ_df = pd.concat(df_list, axis=1, join='inner')
