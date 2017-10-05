"""Fits a proportional hazards model to data after PCA dimentionality reduction.
PH model is regularized by elastic net.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = '/test-data/'
#DATA_DIR = 'Expression/'
ROOT_DIR = '/'
#ROOT_DIR = ''
WRITE_DIR = '/output/predictions.tsv'
#WRITE_DIR = 'output/predictions.tsv'
CLINICAL_FILE = '/test-data/sc2_Validation_ClinAnnotations.csv'
#CLINICAL_FILE = 'Clinical/globalClinTraining.csv'


def risk(ma_gene, params):
    return np.exp(np.dot(ma_gene, params))


def zscore(x, mean, std):
    return (x - mean) / std


def write_tsv(studies, patients, risk_score, high_risk_flag, path):
    with open(path, 'a') as output:
        output.write('study\tpatient\tpredictionscore\thighriskflag\n')
        print('study\tpatient\tpredictionscore\thighriskflag\n')
        for s, p, r, f in zip(studies, patients, risk_score, high_risk_flag):
            output.write('{}\t{}\t{}\t{}\n'.format(s, p, r, str(f).upper()))
            print('{}\t{}\t{}\t{}\n'.format(s, p, r, str(f).upper()))


def integrate_dfs(data_type, features=None):
    print(data_type)
    data_files = set(clinical_df[data_type])

    df_list = []
    for ef in data_files:
        print(ef)
        if not str(ef) == 'nan':
            if str(ef).endswith('.csv'):
                df_list.append(pd.read_csv(os.path.join(DATA_DIR, ef)))
            else:
                df_list.append(pd.read_table(os.path.join(DATA_DIR, ef), '\t'))

    for i in range(len(df_list)):
        df_list[i] = df_list[i].set_index(df_list[i].columns[0])
        if features:
            df_list[i] = df_list[i].loc[features]
        print(df_list[i].shape)

    return pd.concat(df_list, axis=1, join='inner')


if __name__ == '__main__':
    clinical_df = pd.read_csv(CLINICAL_FILE)

    data_types = ['MA_geneLevelExpFile', 'MA_probeLevelExpFile',
                  'RNASeq_geneLevelExpFile', 'RNASeq_transLevelExpFile']

    for data_type in data_types:
        integ_df = integrate_dfs(data_type)
        print(integ_df.shape)
        print(np.sum(np.sum(np.isnan(integ_df))))

    features = np.load(os.path.join(ROOT_DIR, 'features.npy'))
    integ_df = integrate_dfs(data_types[0], features)
    # Column 1 of clinical df contains patient IDs.
    clinical_df = clinical_df.set_index(clinical_df.columns[1])
    ma_clinical_df = clinical_df.loc[integ_df.columns]

    mean = np.load(os.path.join(ROOT_DIR, 'mean.npy'))
    std = np.load(os.path.join(ROOT_DIR, 'std.npy'))
    ma_gene_highdim = zscore(integ_df.transpose().as_matrix(), mean, std)

    pca_projection = np.load(os.path.join(ROOT_DIR, 'pca.npy'))
    ma_gene = np.dot(ma_gene_highdim, pca_projection.transpose())

    params = np.load(os.path.join(ROOT_DIR, 'params.npy'))

    risk_score = risk(ma_gene, params)
    high_risk_flag = risk_score > np.median(risk_score)

    patients = clinical_df.index.values
    patients_2 = integ_df.columns
    studies = clinical_df.loc[patients_2].Study.values

    print(params.shape, pca_projection.shape, features.shape, patients.shape,
          patients_2.shape, studies.shape)

    write_path = os.path.join(WRITE_DIR)
    write_tsv(studies, patients_2, risk_score, high_risk_flag, write_path)
