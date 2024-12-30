from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd
from glob import glob
import warnings

warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv('/tsd/p1504/data/durable/preliminary-pre-ctad-data/ai-development-data/ai_scd_crf_corr_apoe_ptau217_ptau181_data_2024-09-16.csv', sep=';', encoding='mac_roman')
df = df[df.ptau217 != df.ptau217.max()]
df = df[df['apoe'] != 0]
df = df[['participant_id', 'age', 'sex', 'education', 'ptau181_num', 'ptau217', 'apoe', 'cog_scale_score', 'dep_scale_score', 'alc_scale_score']]
df.dropna(inplace=True)

cantab = pd.read_csv('/ess/p1504/data/durable/preliminary-pre-ctad-data/ai-development-data/ai-mind_cantab_data_ai_2024-04-24.csv', sep=';', encoding='mac_roman')
cantab = cantab[['participant_id', 'PALFAMS28', 'PALTEA28']]

main = pd.merge(df, cantab, on='participant_id', how='inner')
df = main.copy()

# Find the best cluster for each group
for k, feature in enumerate(['demo', 'tau217_apoe', 'tau181_apoe', 'blood', 'clinical', 'cognitive', 'bc', 'db', 'dc', 'dbc', 'd_cl', 'cl_cb', 'cl_c_bd', 'ptau181_num', 'ptau217']):
    features = []
    feature_name = None

    if feature == 'demo':
        features.append('age')
        features.append('sex')
        features.append('education')
        features.append('alc_scale_score')
        feature_name = 'Demographics'
    elif feature == 'clinical':
        features.append('cog_scale_score')
        features.append('dep_scale_score')
        feature_name = 'Clinical'
    elif feature == 'tau217_apoe':
        features.append('ptau217')
        features.append('apoe')
        feature_name = f'{feature}'
    elif feature == 'tau181_apoe':
        features.append('ptau181_num')
        features.append('apoe')
        feature_name = f'{feature}'
    elif feature == 'cognitive':
        features.append('PALFAMS28')
        features.append('PALTEA28')
        feature_name = 'Cognitive'
    elif feature == 'bc':
        features.append('PALFAMS28')
        features.append('PALTEA28')
        features.append('ptau181_num')
        features.append('ptau217')
        features.append('apoe')
        feature_name = 'Blood and Cognitive'
    elif feature == 'db':
        features.append('age')
        features.append('sex')
        features.append('education')
        features.append('ptau181_num')
        features.append('ptau217')
        features.append('apoe')
        feature_name = 'Demographics and Blood'
    elif feature == 'dc':
        features.append('age')
        features.append('sex')
        features.append('education')
        features.append('PALFAMS28')
        features.append('PALTEA28')
        feature_name = 'Demographics and Cognitive'
    elif feature == 'dbc':
        features.append('age')
        features.append('sex')
        features.append('education')
        features.append('PALFAMS28')
        features.append('PALTEA28')
        features.append('ptau181_num')
        features.append('ptau217')
        features.append('apoe')
        feature_name = 'Demographics, Blood and Cognitive'
    elif feature == 'd_cl':
        features.append('age')
        features.append('sex')
        features.append('education')
        features.append('alc_scale_score')
        features.append('cog_scale_score')
        features.append('dep_scale_score')
        feature_name = 'Demographics and Clinical'
    elif feature == 'cl_cb':
        features.append('cog_scale_score')
        features.append('dep_scale_score')
        features.append('PALFAMS28')
        features.append('PALTEA28')
        features.append('ptau181_num')
        features.append('ptau217')
        features.append('apoe')
        feature_name = 'Clinical, Cognitive, and Blood'
    elif feature == 'cl_c_bd':
        features.append('cog_scale_score')
        features.append('dep_scale_score')
        features.append('PALFAMS28')
        features.append('PALTEA28')
        features.append('ptau181_num')
        features.append('ptau217')
        features.append('apoe')
        features.append('age')
        features.append('sex')
        features.append('education')
        features.append('alc_scale_score')
        feature_name = 'Clinical, Cognitive, Blood, and Demographics'
    else:
        features.append(feature)
        feature_name = f'{feature}'

    df2 = df[features]

    if 'sex' in features:
        df2 = pd.get_dummies(df2, columns=['sex'], dtype=float, drop_first=False)
    if 'education' in features:
        df2 = pd.get_dummies(df2, columns=['education'], dtype=float, drop_first=False)
    if 'apoe' in features:
        df2 = pd.get_dummies(df2, columns=['apoe'], dtype=float, drop_first=False)
    if 'ptau181_num' in features:
        df2['ptau181_num'] = df2['ptau181_num'].clip(upper=5)
    if 'ptau217' in features:
        df2['ptau217'] = df2['ptau217'].clip(upper=10)

    df2.dropna(inplace=True)
