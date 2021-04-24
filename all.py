import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import functools
from multiprocessing import Pool
from tqdm import tqdm

from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from scipy import linalg

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib import gridspec

import umap
import umap.plot

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC

################################################
#                  Figure 1.1
################################################

scaled_arr = StandardScaler().fit_transform(profiles)
reducer = umap.UMAP()
embedding = reducer.fit_transform(profiles)

moa_map = {'AD Relevant': 0,
 'Auto/Endo': 1,
 'Inflammation': 2,
 'Membrane Channels': 3,
 'ROS': 4,
 'Transport': 5}
well_to_moa = dict(reagent_map.groupby('Metadata_Well')['MOA'].first())
legend_elements = []
for i in range(6):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=list(moa_map.keys())[i], markerfacecolor=sns.color_palette()[i],markersize=15))
    
fig, ax = plt.subplots()
ax.scatter(embedding[:, 0], embedding[:, 1], s=5, c=[sns.color_palette()[moa_map[well_to_moa[x]]] for x in well_labels])
ax.set_xticks([])
ax.set_yticks([])
plt.title('Means')
fig.legend(handles=legend_elements, bbox_to_anchor=(0.55, 0.425))
plt.show()

################################################
#                  Figure 1.2
################################################

def robust_linear_normalise(df, control_wells, plates):
    normalised_plates = []
    feature_columns = df.columns[5:]
    for plate in plates:
        plate_data = df.loc[df.Plate.eq(plate)].copy()
        
        # Calculate percentiles from controls
        control_data = plate_data.loc[plate_data.Well.isin(control_wells)].values[:, 5:].astype(float)
        low_percentiles = np.percentile(control_data, 1, axis=0)
        high_percentiles = np.percentile(control_data, 99, axis=0)
        
        divisor = high_percentiles - low_percentiles
        divisor[divisor == 0] = 1 # When there is no variation in negative controls, don't normalise
        plate_data[feature_columns] = plate_data[feature_columns].subtract(low_percentiles, axis=1).divide(divisor, axis=1)
        normalised_plates.append(plate_data)
        
    return pd.concat(normalised_plates, ignore_index=True)

def means_profiles(df, wells, plates):
    reagent_profiles_dict = {}
    for well in tqdm(wells):
        plate_means = []
        for plate in plates:
            plate_mean = np.mean(df.loc[df.Well.eq(well) & df.Plate.eq(plate)].values[:, 5:].astype(float), axis=0)
            plate_means.append(plate_mean)
        well_profile = np.median(np.array(plate_means), axis=0)
        reagent_profiles_dict[well] = well_profile
        
    return pd.DataFrame.from_dict(reagent_profiles_dict, orient='index', columns=df.columns[5:]).reset_index().rename(columns={'index': 'Well'})

def ks_profiles(df, control_wells, wells, plates):
    # Set up negative control samples
    control_df = df.loc[df.Well.isin(control_wells)]
    pfunc = functools.partial(well_ks_profiles, control_df=control_df, plates=plates)
    # Set up starmap args
    starmap_args = []
    for well in wells:
        starmap_args.append((well, df.loc[df.Well.eq(well)]))
        
    with Pool(32) as pool:
        res = list(pool.starmap(pfunc, tqdm(starmap_args, total=len(starmap_args)), chunksize=1))
        
    reagent_profiles_dict = dict(zip(wells, res))
    return pd.DataFrame.from_dict(reagent_profiles_dict, orient='index', columns=df.columns[5:]).reset_index().rename(columns={'index': 'Well'})
        
def well_ks_profiles(well, reagent_df, control_df, plates):
    plate_profiles = []
    for plate in plates:
        plate_profile = []
        plate_arr = reagent_df.loc[reagent_df.Plate.eq(plate)].values[:, 5:].astype(float).T
        control_arr = control_df.loc[control_df.Plate.eq(plate)].values[:, 5:].astype(float).T
        for i in range(len(plate_arr)):
            plate_profile.append(ks_2samp(plate_arr[i], control_arr[i], mode='asymp')[0])
        plate_profiles.append(plate_profile)
        
    well_profile = np.median(np.array(plate_profiles), axis=0)
    return well_profile

def svm_profiles(df, control_wells, wells, plates):
    # Set up negative control samples
    control_df = df.loc[df.Well.isin(control_wells)]
    pfunc = functools.partial(well_svm_profiles, control_df=control_df, plates=plates)
    # Set up starmap args
    starmap_args = []
    for well in wells:
        starmap_args.append((well, df.loc[df.Well.eq(well)]))
        
    with Pool(32) as pool:
        res = list(pool.starmap(pfunc, tqdm(starmap_args, total=len(starmap_args)), chunksize=1))
        
    reagent_profiles_dict = dict(zip(wells, res))
    return pd.DataFrame.from_dict(reagent_profiles_dict, orient='index', columns=df.columns[5:]).reset_index().rename(columns={'index': 'Well'})
        
def well_svm_profiles(well, reagent_df, control_df, plates):
    plate_profiles = []
    for plate in plates:
        reagent_X = reagent_df.loc[reagent_df.Plate.eq(plate)].values[:, 5:].astype(float)
        control_X = control_df.loc[control_df.Plate.eq(plate)].values[:, 5:].astype(float)
        X = np.concatenate([control_X, reagent_X], axis=0)
        y = [0]*len(control_X) + [1]*len(reagent_X)
        clf = LinearSVC()
        m = clf.fit(X, y)
        plate_profiles.append(m.coef_[0])
        
    well_profile = np.median(np.array(plate_profiles), axis=0)
    return well_profile

def fa_profiles(df, control_wells, wells, plates, components=60):
    control_X = df.loc[df.Well.isin(control_wells)].values[:, 5:].astype(float)
    fa = FactorAnalysis(n_components=components)
    control_X_reduc = fa.fit_transform(control_X)
    # Compute hat matrix for the MAP estimate
    A = fa.components_.T
    mu = fa.mean_
    sigma = np.diag(fa.noise_variance_)
    A_hat = A.T @ np.linalg.inv((A @ A.T) + sigma)
    
    reagent_profiles_dict = {}
    for well in wells:
        plate_profiles = []
        for plate in plates:
            reagent_X = df.loc[df.Well.eq(well) & df.Plate.eq(plate)].values[:, 5:].astype(float)
            v = np.mean(reagent_X, axis=0) - mu
            plate_profiles.append(A_hat @ v)
            
        well_profile = np.median(np.array(plate_profiles), axis=0)
        reagent_profiles_dict[well] = well_profile
    
    return pd.DataFrame.from_dict(reagent_profiles_dict, orient='index', columns=list(range(components))).reset_index().rename(columns={'index': 'Well'})

treated_wells = reagent_map.loc[reagent_map.ReagentClass.eq('SAMPLE')].Metadata_Well.unique()
treated_wells = treated_wells[treated_wells != 'N22']
negative_control_wells = reagent_map.loc[reagent_map.ReagentClass.eq('NEGATIVE')].Metadata_Well.unique()
reagent_profiles = means_profiles(all_features, treated_wells, [1, 2, 3, 4, 5])

def knn_nsc(profiles, reagent_map, k, metric='cosine'):
    dists_arr = profiles.values[:, 1:].astype(float)
    dists = cdist(dists_arr, dists_arr, metric=metric)
    dists = pd.DataFrame(data=dists, index=profiles.Well.values, columns=profiles.Well.values)
    well_to_moa = dict(reagent_map.groupby('Metadata_Well')['MOA'].first())
    well_to_rid = dict(reagent_map.groupby('Metadata_Well')['Reagent ID'].first())
    well_to_conc = dict(reagent_map.groupby('Metadata_Well')['Conc (uM)'].first())
    preds = []
    for well in profiles.Well.values:
        curr_rid = well_to_rid[well]
        curr_conc = well_to_conc[well]
        same_rid_wells = [k for k, v in well_to_rid.items() if v == curr_rid]
        well_dists = dists.loc[well]
        well_dists = well_dists.loc[~well_dists.index.isin(same_rid_wells)].sort_values()
        nearest_moas = [well_to_moa[w] for w in well_dists.index[:k]]
        pred = Counter(nearest_moas).most_common(1)[0][0]
        preds.append(pred)
    
    return preds

well_to_moa = dict(reagent_map.groupby('Metadata_Well')['MOA'].first())
true_labels = [well_to_moa[w] for w in reagent_profiles.Well]
preds = knn_nsc(reagent_profiles, reagent_map, 1, metric='cosine')
accuracy_score(true_labels, preds)

moas = ['AD Relevant', 'Inflammation', 'Membrane Channels', 'Auto/Endo', 'Transport', 'ROS']
fig, axs = plt.subplots(2, 2)
cm = confusion_matrix(true_labels, means_preds, labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=True, cbar=False, vmin=0, vmax=0.55, ax=axs[0,0])
axs[0,0].set_xticks([])
axs[0,0].set_title('Means: accuracy = 54.4%', fontsize=12, fontweight='bold', color='#464a47')

cm = confusion_matrix(true_labels, ks_preds, labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=True, cbar=False, vmin=0, vmax=0.55, ax=axs[0,1])
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,1].set_title('KS-statistic: accuracy = 50.2%', fontsize=12, fontweight='bold', color='#464a47')

cm = confusion_matrix(true_labels, svm_preds, labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=True, cbar=False, vmin=0, vmax=0.55, ax=axs[1,0])
axs[1,0].set_title('SVM: accuracy = 56.7%', fontsize=12, fontweight='bold', color='#464a47')

cm = confusion_matrix(true_labels, fa_preds, labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=True, cbar=False, vmin=0, vmax=0.55, ax=axs[1,1])
axs[1,1].set_yticks([])
axs[1,1].set_title('Factor analysis: accuracy = 55.8%', fontsize=12, fontweight='bold', color='#464a47')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

################################################
#                  Figure 1.3
################################################

moas = ['AD', 'Infl', 'M Ch', 'A/E', 'Tran', 'ROS']
well_to_conc = dict(reagent_map.groupby('Metadata_Well')['Conc (uM)'].first())
fig, axs = plt.subplots(1, 4)
cm = confusion_matrix(list(map(moa_abbrv_map, true_labels)), list(map(moa_abbrv_map, CL_803_preds)), labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=False, cbar=False, vmin=0, vmax=0.75, ax=axs[0])
axs[0].set_title('AD 1: accuracy = 54.4%', fontsize=8, fontweight='bold', color='#464a47')
axs[0].set_aspect('equal')
axs[0].tick_params(axis='y', rotation=0)

cm = confusion_matrix(list(map(moa_abbrv_map, true_labels)), list(map(moa_abbrv_map, CL_808_preds)), labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=False, cbar=False, vmin=0, vmax=0.75, ax=axs[1])
axs[1].set_yticks([])
axs[1].set_title('AD 2: accuracy = 45.6%', fontsize=8, fontweight='bold', color='#464a47')
axs[1].set_aspect('equal')

cm = confusion_matrix(list(map(moa_abbrv_map, true_labels)), list(map(moa_abbrv_map, CL_840_preds)), labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=False, cbar=False, vmin=0, vmax=0.75, ax=axs[2])
axs[2].set_title('Health 1: accuracy = 48.4%', fontsize=8, fontweight='bold', color='#464a47')
axs[2].set_yticks([])
axs[2].set_aspect('equal')

cm = confusion_matrix(list(map(moa_abbrv_map, true_labels)), list(map(moa_abbrv_map, CL_856_preds)), labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=False, cbar=False, vmin=0, vmax=0.75, ax=axs[3])
axs[3].set_yticks([])
axs[3].set_title('Health 2: accuracy = 61.9%', fontsize=8, fontweight='bold', color='#464a47')
axs[3].set_aspect('equal')

plt.subplots_adjust(wspace=0.1)
plt.show()

################################################
#                  Figure 1.4
################################################

well_to_conc = dict(reagent_map.groupby('Metadata_Well')['Conc (uM)'].first())
well_preds = pd.DataFrame({'Well': reagent_profiles.Well.values, 'pred': preds, 'moa': [well_to_moa[w] for w in reagent_profiles.Well.values], 
                           'conc': [well_to_conc[w] for w in reagent_profiles.Well.values]})
concs = [0.3, 1, 3, 10, 30]
moas = ['AD', 'Infl', 'M Ch', 'A/E', 'Tran', 'ROS']
accs = ['65.1%', '55.8%', '65.1%', '44.2%', '41.9%']
fig, axs = plt.subplots(1, 5, sharex=True, sharey=True)
for i, ax in enumerate(axs):
    true_labels = well_preds.loc[well_preds.conc.eq(concs[i])].moa.values
    conc_preds = well_preds.loc[well_preds.conc.eq(concs[i])].pred.values
    cm = confusion_matrix(list(map(moa_abbrv_map, true_labels)), list(map(moa_abbrv_map, conc_preds)), labels=moas, normalize='true')
    sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=False, cbar=False, vmin=0, vmax=0.75, ax=ax)
    ax.set_title(str(concs[i]) + ' uM: Acc = ' + accs[i], fontsize=8, fontweight='bold', color='#464a47')
    ax.set_aspect('equal')
    ax.tick_params(axis='y', rotation=0)
plt.show()

################################################
#                  Figure 1.5
################################################

negative_control_wells = reagent_map.loc[reagent_map.ReagentClass.eq('NEGATIVE')].Metadata_Well.unique()
normalised_features = robust_linear_normalise(all_features, negative_control_wells, [1, 2, 3, 4, 5])
treated_wells = reagent_map.loc[reagent_map.ReagentClass.eq('SAMPLE')].Metadata_Well.unique()
treated_wells = treated_wells[treated_wells != 'N22']
normalised_features_treated = normalised_features.loc[normalised_features.Well.isin(treated_wells)]

well_to_moa = dict(reagent_map.groupby('Metadata_Well')['MOA'].first())
X = normalised_features_treated.values[:, 5:].astype(float)
y = normalised_features_treated.Well.map(well_to_moa).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_clf = RandomForestClassifier(n_estimators=250, n_jobs=32)
rf_clf.fit(X_train, y_train)

lr_clf = LogisticRegression(solver='saga', multi_class='ovr', n_jobs=16, max_iter=1000)
lr_clf.fit(X_train, y_train)

moas = ['AD Relevant', 'Inflammation', 'Membrane Channels', 'Auto/Endo', 'Transport', 'ROS']
fig, axs = plt.subplots(1, 2)
cm = confusion_matrix(y_test, rf_preds, labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=True, cbar=False, vmin=0, vmax=1, ax=axs[0])
axs[0].set_title('Random Forest: acc = 74.4%', fontsize=14, fontweight='bold', color='#464a47')

cm = confusion_matrix(y_test, lr_clf.predict(X_test), labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=True, cbar=False, vmin=0, vmax=1, ax=axs[1])
axs[1].set_yticks([])
axs[1].set_title('Logistic Regression: acc = 72.7%', fontsize=14, fontweight='bold', color='#464a47')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

################################################
#                  Figure 1.6
################################################

bf_features, cona_features, dapi_features, mito_features, phal_features = [], [], [], [], []
feature_names = all_features.columns[5:]
for f in feature_names:
    if 'BF' in f:
        if all([ch not in f for ch in ['ConA', 'DAPI', 'Mito', 'Phal']]):
            bf_features.append(f)
    if 'ConA' in f:
        if all([ch not in f for ch in ['BF', 'DAPI', 'Mito', 'Phal']]):
            cona_features.append(f)
    if 'DAPI' in f:
        if all([ch not in f for ch in ['BF', 'ConA', 'Mito', 'Phal']]):
            dapi_features.append(f)
    if 'Mito' in f:
        if all([ch not in f for ch in ['BF', 'ConA', 'DAPI', 'Phal']]):
            mito_features.append(f)
    if 'Phal' in f:
        if all([ch not in f for ch in ['BF', 'ConA', 'DAPI', 'Mito']]):
            phal_features.append(f)
            
feature_partition = {'BF': bf_features, 
                     'ConA': cona_features, 
                     'DAPI': dapi_features, 
                     'Mito': mito_features, 
                     'Phal': phal_features}

moas = ['AD', 'Infl', 'M Ch', 'A/E', 'Tran', 'ROS']
moa_abbrv = dict(zip(['AD Relevant', 'Inflammation', 'Membrane Channels', 'Auto/Endo', 'Transport', 'ROS'], ['AD', 'Infl', 'M Ch', 'A/E', 'Tran', 'ROS']))
moa_abbrv_map = lambda x: moa_abbrv[x]
fig, axs = plt.subplots(1, 5)
cm = confusion_matrix(list(map(moa_abbrv_map, y_test)), list(map(moa_abbrv_map, bf_preds)), labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=False, cbar=False, vmin=0, vmax=0.82, ax=axs[0])
axs[0].set_title('BF: acc = 51.1%', fontsize=10, fontweight='bold', color='#464a47')
axs[0].set_aspect('equal')
axs[0].tick_params(axis='y', rotation=0)

cm = confusion_matrix(list(map(moa_abbrv_map, y_test)), list(map(moa_abbrv_map, cona_preds)), labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=False, cbar=False, vmin=0, vmax=0.82, ax=axs[1])
axs[1].set_yticks([])
axs[1].set_title('Con A: acc = 51.1%', fontsize=10, fontweight='bold', color='#464a47')
axs[1].set_aspect('equal')

cm = confusion_matrix(list(map(moa_abbrv_map, y_test)), list(map(moa_abbrv_map, dapi_preds)), labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=False, cbar=False, vmin=0, vmax=0.82, ax=axs[2])
axs[2].set_title('DAPI: acc = 55.2%', fontsize=10, fontweight='bold', color='#464a47')
axs[2].set_yticks([])
axs[2].set_aspect('equal')

cm = confusion_matrix(list(map(moa_abbrv_map, y_test)), list(map(moa_abbrv_map, mito_preds)), labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=False, cbar=False, vmin=0, vmax=0.82, ax=axs[3])
axs[3].set_yticks([])
axs[3].set_title('Mito: acc = 52.3%', fontsize=10, fontweight='bold', color='#464a47')
axs[3].set_aspect('equal')


cm = confusion_matrix(list(map(moa_abbrv_map, y_test)), list(map(moa_abbrv_map, phal_preds)), labels=moas, normalize='true')
sns.heatmap(cm, xticklabels=moas, yticklabels=moas, cmap='Blues', annot=False, cbar=False, vmin=0, vmax=0.82, ax=axs[4])
axs[4].set_yticks([])
axs[4].set_title('Phal: acc = 52.3%', fontsize=10, fontweight='bold', color='#464a47')
axs[4].set_aspect('equal')
plt.subplots_adjust(wspace=0.1)
plt.show()

################################################
#                  Table 1.1
################################################

rf_importances = pd.Series(index=all_features.columns[5:], data=rf_clf.feature_importances_).sort_values(ascending=False)
rf_importances[:5]

################################################
#                  Figure 1.7
################################################

rf_importances = pd.Series(index=all_features.columns[5:], data=rf_clf.feature_importances_).sort_values(ascending=False)
groups = ['Correlation', 'Granularity', 'Intensity', 'LocRad', 'Texture']
channels = ['BF', 'ConA', 'DAPI', 'Mito', 'Phal']
nuclei_imp = np.zeros((5, 5))
for i, group in enumerate(groups):
    for j, channel in enumerate(channels):
        if group == 'LocRad':
            nuclei_imp[i, j] = rf_importances.loc[rf_importances.index.str.contains('Nuclei') & 
                                                 rf_importances.index.str.contains(channel) &
                                                 (rf_importances.index.str.contains('RadialDistribution') | rf_importances.index.str.contains('Location'))].mean()
        else:
            nuclei_imp[i, j] = rf_importances.loc[rf_importances.index.str.contains('Nuclei') & 
                                                 rf_importances.index.str.contains(channel) &
                                                 rf_importances.index.str.contains(group)].mean()
            
feature_groups = ['Correlation', 'Granularity', 'Intensity', 'Location', 'Texture']
channels = ['BF', 'Con A', 'DAPI', 'Mito', 'Phal']
fig, axs = plt.subplots(1, 3)
sns.heatmap(nuclei_imp*1e4, xticklabels=channels, yticklabels=feature_groups, cmap='Blues', annot=True, cbar=False, vmin=0, vmax=3, ax=axs[0])
axs[0].set_title('Nuclei', fontsize=14, fontweight='bold', color='#464a47')
axs[0].set_aspect('equal')

g2 = sns.heatmap(cytoplasm_imp*1e4, xticklabels=channels, yticklabels=feature_groups, cmap='Blues', annot=True, cbar=False, vmin=0, vmax=3, ax=axs[1],                 
                 mask=np.isnan(cytoplasm_imp))
g2.set_facecolor('#d7d8db')
axs[1].set_yticks([])
axs[1].set_title('Cytoplasms', fontsize=14, fontweight='bold', color='#464a47')
axs[1].set_aspect('equal')

g1 = sns.heatmap(cells_imp*1e4, xticklabels=channels, yticklabels=feature_groups, cmap='Blues', annot=True, cbar=False, vmin=0, vmax=3, ax=axs[2], mask=np.isnan(cells_imp))
g1.set_facecolor('#d7d8db')
axs[2].set_yticks([])
axs[2].set_title('Cells', fontsize=14, fontweight='bold', color='#464a47')
axs[2].set_aspect('equal')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

################################################
#                  Table 1.2
################################################

lr_importances = dict()
for i, moa in enumerate(lr_clf.classes_):
    most_pos_neg = []
    labelled_coefs = pd.Series(index=all_features.columns[5:], data=lr_clf.coef_[i]).sort_values(ascending=False)
    most_pos_neg.append((labelled_coefs.index[0], labelled_coefs[0]))
    most_pos_neg.append((labelled_coefs.index[-1], labelled_coefs[-1]))
    lr_importances[moa] = most_pos_neg
    
################################################
#                  Figure 1.8
################################################

plate_means = []
for plate in range(1, 6):
    plate_mean = np.mean(normalised_features.loc[normalised_features.Well.isin(negative_control_wells) & normalised_features.Plate.eq(plate)].values[:, 5:].astype(float), 
                         axis=0)
    plate_means.append(plate_mean)
negative_control_profile = np.median(np.array(plate_means), axis=0)

neg_to_reagent_dists = cdist(np.expand_dims(negative_control_profile, 0), reagent_profiles.values[:, 1:].astype(float), metric='euclidean')[0]
neg_to_reagent = pd.Series(index=reagent_profiles.Well.values, data=neg_to_reagent_dists)
neg_to_reagent.sort_values()

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
moas = ['AD Relevant', 'Inflammation', 'Membrane Channels', 'Auto/Endo', 'Transport', 'ROS']
for i, ax in enumerate(axs.reshape(-1)):
    moa = moas[i]
    ad_rel_reagents = reagent_map.loc[reagent_map.MOA.eq(moa)]['Synonym'].unique()
    conc_dists_map = {}
    for r in ad_rel_reagents:
        conc_dists = []
        for conc in [0.3, 1.0, 3.0, 10.0, 30.0]:
            well = reagent_map.loc[(reagent_map['Conc (uM)'] == conc) & (reagent_map['Synonym'] == r)].Metadata_Well.unique()[0]
            conc_dists.append(neg_to_reagent[well])
        conc_dists_map[r] = conc_dists
    x = [0.3, 1.0, 3.0, 10.0, 30.0]
    for k, v in conc_dists_map.items():
        line = ax.plot(x, v, 'o--', label=k)
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 7.375})
    ax.legend(loc='upper left', prop={'size': 4.5})
    #ax.set_xlabel('Conc (uM)')
    #ax.set_ylabel('log-Euclidean dist. to DMSO')
    ax.set_ylim(top=max(neg_to_reagent))
    ax.set_title(moa, fontsize=8, fontweight='bold', color='#464a47')
plt.subplots_adjust(wspace=0.1, hspace=0.15)
fig.text(0.5, 0.04, 'Conc (uM)', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Euclidean dist. to DMSO', va='center', rotation='vertical', fontsize=14)
plt.show()

################################################
#    Figure 1.9, 1.10, 1.11, 1.12 Table 1.4
################################################
# These are results on the microglia data derived in the same way as the astrocyte data in the relevant snippet above