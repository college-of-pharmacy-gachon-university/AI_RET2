# REINVENT Compatible Multi-Algorithm RET Mutant QSAR Model with Scikit-Learn Random Search
# ============================================================
# Multi-fingerprint and multi-algorithm evaluation for REINVENT generative modeling
# Using scikit-learn RandomizedSearchCV for hyperparameter optimization

import os
import numpy as np
import pandas as pd
import pickle
import warnings
import argparse
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
SEED = 56789
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Cheminformatics libraries
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.ML.Cluster import Butina
from rdkit.Chem.Scaffolds import MurckoScaffold

# Machine learning libraries
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor)
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import (cross_val_score, KFold, train_test_split,
                                     GroupShuffleSplit, RandomizedSearchCV)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
from scipy.stats import uniform, randint

# UMAP for clustering-based split
try:
    import umap
    UMAP_AVAILABLE = True
except (ImportError, Exception) as e:
    UMAP_AVAILABLE = False
    print(f"UMAP not available: {e}. Using random split as fallback.")

# =======================================================================
# 1. FINGERPRINT GENERATION CLASSES
# =======================================================================
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
SEED = 56789
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Cheminformatics libraries
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.ML.Cluster import Butina
from rdkit.Chem.Scaffolds import MurckoScaffold

# Machine learning libraries
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, IsolationForest)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import (cross_val_score, KFold, train_test_split,
                                     GroupShuffleSplit)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist

# =======================================================================
# 1. FINGERPRINT GENERATION CLASSES
# =======================================================================

class REINVENTDescriptors:
    """Generate REINVENT-compatible molecular descriptors (one type at a time)"""

    def __init__(self):
        self.descriptor_type = None
        self.parameters = {}

    def smiles_to_mols(self, smiles_list):
        """Convert SMILES to RDKit molecules"""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
        valid_mols = [mols[i] for i in valid_indices]
        return valid_mols, valid_indices

    def get_ecfp_fingerprints(self, smiles_list, radius=3, n_bits=2048, use_counts=False):
        """Generate ECFP fingerprints (binary or counts)"""
        mols, valid_idx = self.smiles_to_mols(smiles_list)

        if use_counts:
            fps = [AllChem.GetMorganFingerprint(mol, radius, useCounts=True)
                   for mol in mols]
            nfp = np.zeros((len(fps), n_bits), np.int32)
            for i, fp in enumerate(fps):
                for idx, v in fp.GetNonzeroElements().items():
                    nidx = idx % n_bits
                    nfp[i, nidx] += int(v)
        else:
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
                   for mol in mols]
            nfp = np.zeros((len(fps), n_bits), np.int32)
            for i, fp in enumerate(fps):
                fp_array = np.zeros((1, n_bits), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(fp, fp_array)
                nfp[i] = fp_array

        self.descriptor_type = 'ecfp_counts' if use_counts else 'ecfp'
        self.parameters = {'radius': radius, 'size': n_bits, 'use_counts': use_counts}
        return nfp

    def get_avalon_fingerprints(self, smiles_list, n_bits=2048):
        """Generate Avalon fingerprints"""
        mols, valid_idx = self.smiles_to_mols(smiles_list)
        fps = [pyAvalonTools.GetAvalonFP(mol, nBits=n_bits) for mol in mols]

        nfp = np.zeros((len(fps), n_bits), dtype=np.int32)
        for i, fp in enumerate(fps):
            fp_array = np.zeros((1, n_bits), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            nfp[i] = fp_array

        self.descriptor_type = 'avalon'
        self.parameters = {'size': n_bits}
        return nfp

    def get_maccs_fingerprints(self, smiles_list):
        """Generate MACCS keys"""
        mols, valid_idx = self.smiles_to_mols(smiles_list)
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]

        nfp = np.zeros((len(fps), 167), dtype=np.int32)
        for i, fp in enumerate(fps):
            fp_array = np.zeros((1, 167), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            nfp[i] = fp_array

        self.descriptor_type = 'maccs_keys'
        self.parameters = {}
        return nfp

    def get_scaffold_groups(self, smiles_list):
        """Get Murcko scaffold groups for group-based splitting"""
        mols, _ = self.smiles_to_mols(smiles_list)
        scaffolds = []

        for mol in mols:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smiles)
            except:
                scaffolds.append("unknown")

        # Convert to numeric groups
        unique_scaffolds = list(set(scaffolds))
        scaffold_groups = [unique_scaffolds.index(s) for s in scaffolds]

        return scaffold_groups

# =======================================================================
# 1.5 SPLITTING FUNCTIONS
# =======================================================================

def random_split(X, y, test_size=0.2, random_state=SEED):
    """Random train/test split"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test

def scaffold_split(smiles_list, y, test_size=0.2, random_state=SEED):
    """Scaffold-based split using Murcko scaffolds"""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    scaffolds = []

    for mol in mols:
        if mol is not None:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smiles)
            except:
                scaffolds.append("unknown")
        else:
            scaffolds.append("invalid")

    # Get unique scaffolds and assign groups
    unique_scaffolds = list(set(scaffolds))
    scaffold_groups = [unique_scaffolds.index(s) for s in scaffolds]

    # Use GroupShuffleSplit to ensure scaffolds are not split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X=np.arange(len(smiles_list)), y=y, groups=scaffold_groups))

    return train_idx, test_idx

def butina_split(smiles_list, y, test_size=0.2, random_state=SEED):
    """Butina clustering-based split"""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]
    valid_indices = [i for i, smi in enumerate(smiles_list) if Chem.MolFromSmiles(smi) is not None]

    if len(mols) < 2:
        # Fallback to random split if not enough molecules
        return random_split(np.arange(len(smiles_list)), y, test_size, random_state)

    # Generate ECFP fingerprints for clustering
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048) for mol in mols]

    # Calculate Tanimoto distances in the format expected by Butina
    n = len(fps)
    dist_list = []
    for i in range(n):
        for j in range(i+1, n):
            dist = 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dist_list.append(dist)

    # Butina clustering - expects a list of distances, not a matrix
    try:
        clusters = Butina.ClusterData(dist_list, n, 0.35, isDistData=True)
    except Exception as e:
        print(f"Butina clustering failed: {e}. Falling back to random split.")
        return random_split(np.arange(len(smiles_list)), y, test_size, random_state)

    # Assign cluster labels
    cluster_labels = np.zeros(n, dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_labels[idx] = cluster_id

    # Split clusters into train/test
    unique_clusters = np.unique(cluster_labels)
    if len(unique_clusters) < 2:
        # If all points are in one cluster, fallback to random split
        return random_split(np.arange(len(smiles_list)), y, test_size, random_state)

    np.random.seed(random_state)
    test_clusters = np.random.choice(unique_clusters, size=max(1, int(len(unique_clusters) * test_size)), replace=False)

    train_idx = []
    test_idx = []
    for i, cluster in enumerate(cluster_labels):
        original_idx = valid_indices[i]
        if cluster in test_clusters:
            test_idx.append(original_idx)
        else:
            train_idx.append(original_idx)

    # Handle invalid molecules
    invalid_indices = [i for i, smi in enumerate(smiles_list) if Chem.MolFromSmiles(smi) is None]
    if invalid_indices:
        train_idx.extend(invalid_indices[:len(invalid_indices)//2])
        test_idx.extend(invalid_indices[len(invalid_indices)//2:])

    return np.array(train_idx), np.array(test_idx)

def umap_clustering_split(smiles_list, y, test_size=0.2, random_state=SEED, n_components=2):
    """UMAP-based clustering split"""
    if not UMAP_AVAILABLE:
        print("UMAP not available, falling back to random split")
        return random_split(np.arange(len(smiles_list)), y, test_size, random_state)

    # Generate ECFP fingerprints
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
    valid_mols = [mols[i] for i in valid_indices]

    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048) for mol in valid_mols]
    fp_array = np.zeros((len(fps), 2048))
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, fp_array[i])

    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(fp_array)

    # K-means clustering
    n_clusters = max(2, int(len(valid_indices) * 0.1))  # Roughly 10% of data per cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embedding)

    # Split clusters
    unique_clusters = np.unique(cluster_labels)
    np.random.seed(random_state)
    test_clusters = np.random.choice(unique_clusters, size=int(len(unique_clusters) * test_size), replace=False)

    train_idx = []
    test_idx = []
    for i, cluster in enumerate(cluster_labels):
        original_idx = valid_indices[i]
        if cluster in test_clusters:
            test_idx.append(original_idx)
        else:
            train_idx.append(original_idx)

    # Handle invalid molecules
    invalid_indices = [i for i, smi in enumerate(smiles_list) if Chem.MolFromSmiles(smi) is None]
    if invalid_indices:
        train_idx.extend(invalid_indices[:len(invalid_indices)//2])
        test_idx.extend(invalid_indices[len(invalid_indices)//2:])

    return np.array(train_idx), np.array(test_idx)

# =======================================================================
# 2. SCIKIT-LEARN RANDOM SEARCH-BASED MULTI-ALGORITHM EVALUATION FUNCTIONS
# =======================================================================

def get_param_distributions():
    """Define parameter distributions for RandomizedSearchCV"""
    return {
        'RandomForest': {
            'n_estimators': randint(200, 401),
            'max_depth': randint(10, 21),
            'min_samples_split': randint(5, 16),
            'min_samples_leaf': randint(2, 7)
        },
        'ExtraTrees': {
            'n_estimators': randint(200, 401),
            'max_depth': randint(10, 21),
            'min_samples_split': randint(5, 16),
            'min_samples_leaf': randint(2, 7)
        },
        'ElasticNet': {
            'alpha': uniform(0.001, 0.999),
            'l1_ratio': uniform(0.1, 0.8)
        }
    }

def evaluate_multiple_algorithms_randomized(X_train, y_train, X_test, y_test):
    """Evaluate multiple algorithms with RandomizedSearchCV optimization"""

    algorithms = {
        'RandomForest': {
            'model_class': RandomForestRegressor,
            'fixed_params': {'random_state': SEED, 'n_jobs': -1},
            'needs_scaling': False
        },
        'ExtraTrees': {
            'model_class': ExtraTreesRegressor,
            'fixed_params': {'random_state': SEED, 'n_jobs': -1},
            'needs_scaling': False
        },
        'ElasticNet': {
            'model_class': ElasticNet,
            'fixed_params': {'random_state': SEED, 'max_iter': 2000},
            'needs_scaling': True
        }
    }

    param_distributions = get_param_distributions()
    results = {}

    for name, config in algorithms.items():
        print(f"  Optimizing {name} with RandomizedSearchCV...")

        # Scale data if needed
        if config['needs_scaling']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_use, X_test_use = X_train_scaled, X_test_scaled
        else:
            X_train_use, X_test_use = X_train, X_test
            scaler = None

        # Create base model
        base_model = config['model_class'](**config['fixed_params'])

        # Setup RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions[name],
            n_iter=20,
            cv=5,
            scoring='r2',
            random_state=SEED,
            n_jobs=-1
        )

        # Fit and find best parameters
        random_search.fit(X_train_use, y_train)
        best_model = random_search.best_estimator_

        # Evaluate
        results[name] = comprehensive_model_evaluation(
            best_model, X_train_use, y_train, X_test_use, y_test, cv_splits=10
        )
        results[name]['best_params'] = random_search.best_params_
        results[name]['scaler'] = scaler
        results[name]['algorithm'] = name
        results[name]['random_search'] = random_search

        print(f"    {name} R²: {results[name]['test_r2']:.3f} (Best params: {random_search.best_params_})")

    return results

def comprehensive_model_evaluation(model, X_train, y_train, X_test, y_test, cv_splits=10):
    """Comprehensive evaluation of a single model"""

    # Cross-validation on training set
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    cv_rmse_scores = -cross_val_score(model, X_train, y_train, cv=kfold,
                                     scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(cv_rmse_scores)
    cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=kfold,
                                    scoring='neg_mean_absolute_error')

    # Train model and evaluate
    model.fit(X_train, y_train)

    # Training set predictions
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_pearson, _ = pearsonr(y_train, y_train_pred)

    # Test set predictions
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_pearson, _ = pearsonr(y_test, y_test_pred)
    test_spearman, _ = spearmanr(y_test, y_test_pred)

    return {
        'model': model,
        'cv_r2_mean': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std(),
        'cv_r2_scores': cv_r2_scores,
        'cv_rmse_mean': cv_rmse_scores.mean(),
        'cv_rmse_std': cv_rmse_scores.std(),
        'cv_rmse_scores': cv_rmse_scores,
        'cv_mae_mean': cv_mae_scores.mean(),
        'cv_mae_std': cv_mae_scores.std(),
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_pearson': train_pearson,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_pearson': test_pearson,
        'test_spearman': test_spearman
    }

# =======================================================================
# 3. MAIN EXECUTION PIPELINE
# =======================================================================

def main(input_file, models_dir, results_file):
    """Main execution pipeline with different splitting strategies

    Parameters:
    -----------
    input_file : str
        Path to input CSV file containing SMILES and pIC50 data
    models_dir : str
        Directory path for saving trained models
    results_file : str
        Path for saving the comprehensive results CSV file
    """

    print("REINVENT Compatible Multi-Algorithm RET Mutant QSAR Model with Randomized Search")
    print("=" * 86)
    print(f"Random seed: {SEED}")
    print(f"Input file: {input_file}")
    print(f"Models directory: {models_dir}")
    print(f"Results file: {results_file}")
    print("Evaluating multiple fingerprints, algorithms, and splitting strategies")

    # Create directory for saving models
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models will be saved to: {models_dir}/")

    print("\n" + "="*86)

    # 1. Load full dataset
    print("\n1. LOADING FULL DATASET")
    print("-" * 25)

    try:
        full_df = pd.read_csv(input_file)

        smiles_list = full_df['RDKIT_SMILES'].values
        y_full = full_df['pIC50'].values

        print(f"Full dataset: {len(full_df)} compounds")

        # Dataset statistics
        print(f"\nDataset Statistics:")
        print(f"  pIC50 range: {y_full.min():.3f} - {y_full.max():.3f}")
        print(f"  Activity span: {y_full.max() - y_full.min():.3f} log units")
        print(f"  Mean: {y_full.mean():.3f}, Std: {y_full.std():.3f}")

    except FileNotFoundError:
        print(f"ERROR: Input dataset file '{input_file}' not found.")
        return None

    # 2. Define splitting strategies and multiple random seeds for Random split only
    split_strategies = {
        'Scaffold': scaffold_split,
        'Butina': butina_split,
        'UMAP_Clustering': umap_clustering_split
    }
    
    # Multiple random seeds for Random split type only
    random_seeds = [10, 1234, 123, 42, 5678]

    # 3. Initialize descriptor generator
    descriptor_gen = REINVENTDescriptors()

    # 4. Evaluate each split type
    print("\n2. EVALUATING SPLIT STRATEGIES")
    print("-" * 35)

    all_split_results = {}

    # Add Random split separately since it works differently with multiple seeds
    split_strategies_with_random = {
        'Scaffold': scaffold_split,
        'Butina': butina_split,
        'UMAP_Clustering': umap_clustering_split
    }
    
    # Handle Random split with multiple seeds
    for seed in random_seeds:
        split_name = f'Random_Seed_{seed}'
        print(f"\nEvaluating {split_name} split (seed={seed})...")
        
        # Perform Random split with specific seed
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(
            np.arange(len(smiles_list)), y_full, test_size=0.2, random_state=seed
        )
        X_train_smiles = smiles_list[X_train_idx]
        X_test_smiles = smiles_list[X_test_idx]

        print(f"  Train set: {len(X_train_smiles)} compounds")
        print(f"  Test set: {len(X_test_smiles)} compounds")

        # Evaluate fingerprints for this split
        fingerprint_configs = {
            'ECFP': {'method': 'ecfp', 'params': {'radius': 3, 'n_bits': 2048, 'use_counts': False}},
            'ECFP_Counts': {'method': 'ecfp_counts', 'params': {'radius': 3, 'n_bits': 2048, 'use_counts': True}},
            'Avalon': {'method': 'avalon', 'params': {'n_bits': 2048}},
            'MACCS': {'method': 'maccs', 'params': {}}
        }

        split_results = {}

        for fp_name, config in fingerprint_configs.items():
            print(f"  Evaluating {fp_name} fingerprints...")

            # Generate fingerprints
            if config['method'] == 'ecfp':
                X_train_fps = descriptor_gen.get_ecfp_fingerprints(X_train_smiles, **config['params'])
                X_test_fps = descriptor_gen.get_ecfp_fingerprints(X_test_smiles, **config['params'])
            elif config['method'] == 'ecfp_counts':
                X_train_fps = descriptor_gen.get_ecfp_fingerprints(X_train_smiles, **config['params'])
                X_test_fps = descriptor_gen.get_ecfp_fingerprints(X_test_smiles, **config['params'])
            elif config['method'] == 'avalon':
                X_train_fps = descriptor_gen.get_avalon_fingerprints(X_train_smiles, **config['params'])
                X_test_fps = descriptor_gen.get_avalon_fingerprints(X_test_smiles, **config['params'])
            elif config['method'] == 'maccs':
                X_train_fps = descriptor_gen.get_maccs_fingerprints(X_train_smiles)
                X_test_fps = descriptor_gen.get_maccs_fingerprints(X_test_smiles)

            print(f"    Features generated: {X_train_fps.shape[1]}")

            # Keep all features
            X_train_var = X_train_fps
            X_test_var = X_test_fps
            print(f"    Features after variance threshold: {X_train_var.shape[1]} (all features kept)")

            # Model & algorithm comparison: Train on train, test on test
            print("    Model & algorithm comparison...")
            comparison_results = evaluate_multiple_algorithms_randomized(X_train_var, y_train, X_test_var, y_test)

            # Combine results (only comparison results)
            fp_results = {}
            for algo in comparison_results:
                fp_results[f"{algo}_Comparison"] = comparison_results[algo]
                fp_results[f"{algo}_Comparison"]['evaluation_type'] = 'comparison'

            # Add metadata
            for algo_name in fp_results:
                fp_results[algo_name]['fingerprint_type'] = fp_name
                fp_results[algo_name]['n_features'] = X_train_var.shape[1]
                fp_results[algo_name]['split_type'] = split_name
                fp_results[algo_name]['random_seed'] = seed

            split_results[fp_name] = fp_results

            # Save models for comparison evaluation (trained with hyperparameter tuning)
            print("    Saving models...")
            for algo in comparison_results:
                model_info = {
                    'model': comparison_results[algo]['model'],
                    'scaler': comparison_results[algo]['scaler'],
                    'best_params': comparison_results[algo]['best_params'],
                    'fingerprint_type': fp_name,
                    'fingerprint_params': config['params'],
                    'split_type': split_name,
                    'evaluation_type': 'comparison',
                    'test_r2': comparison_results[algo]['test_r2'],
                    'test_rmse': comparison_results[algo]['test_rmse'],
                    'cv_r2_mean': comparison_results[algo]['cv_r2_mean'],
                    'n_features': X_train_var.shape[1],
                    'random_seed': seed
                }
                
                model_filename = f"{models_dir}/{split_name}_{fp_name}_{algo}_comparison.pkl"
                with open(model_filename, 'wb') as f:
                    pickle.dump(model_info, f)

            # Print summary for this fingerprint
            comparison_keys = [k for k in fp_results if 'Comparison' in k]
            if comparison_keys:
                best_algo_for_fp = max(comparison_keys, key=lambda x: fp_results[x]['test_r2'])
                best_r2_for_fp = fp_results[best_algo_for_fp]['test_r2']
                print(f"    Best algorithm: {best_algo_for_fp} (R² = {best_r2_for_fp:.3f})")
                
                # Save the best model for this split-fingerprint combination
                best_algo_name = best_algo_for_fp.replace('_Comparison', '')
                best_model_info = {
                    'model': fp_results[best_algo_for_fp]['model'],
                    'scaler': fp_results[best_algo_for_fp]['scaler'],
                    'best_params': fp_results[best_algo_for_fp]['best_params'],
                    'fingerprint_type': fp_name,
                    'fingerprint_params': config['params'],
                    'split_type': split_name,
                    'evaluation_type': 'comparison',
                    'test_r2': fp_results[best_algo_for_fp]['test_r2'],
                    'test_rmse': fp_results[best_algo_for_fp]['test_rmse'],
                    'cv_r2_mean': fp_results[best_algo_for_fp]['cv_r2_mean'],
                    'n_features': X_train_var.shape[1],
                    'random_seed': seed
                }
                best_model_filename = f"{models_dir}/{split_name}_{fp_name}_BEST_MODEL.pkl"
                with open(best_model_filename, 'wb') as f:
                    pickle.dump(best_model_info, f)
                print(f"    Best model saved: {best_model_filename}")

        all_split_results[split_name] = split_results

    # Handle other split types (Scaffold, Butina, UMAP_Clustering)
    for split_name, split_func in split_strategies_with_random.items():
        print(f"\nEvaluating {split_name} split...")

        # Perform split
        X_train_idx, X_test_idx = split_func(smiles_list, y_full, test_size=0.2)
        X_train_smiles = smiles_list[X_train_idx]
        y_train = y_full[X_train_idx]
        X_test_smiles = smiles_list[X_test_idx]
        y_test = y_full[X_test_idx]

        print(f"  Train set: {len(X_train_smiles)} compounds")
        print(f"  Test set: {len(X_test_smiles)} compounds")

        # Evaluate fingerprints for this split
        fingerprint_configs = {
            'ECFP': {'method': 'ecfp', 'params': {'radius': 3, 'n_bits': 2048, 'use_counts': False}},
            'ECFP_Counts': {'method': 'ecfp_counts', 'params': {'radius': 3, 'n_bits': 2048, 'use_counts': True}},
            'Avalon': {'method': 'avalon', 'params': {'n_bits': 2048}},
            'MACCS': {'method': 'maccs', 'params': {}}
        }

        split_results = {}

        for fp_name, config in fingerprint_configs.items():
            print(f"  Evaluating {fp_name} fingerprints...")

            # Generate fingerprints
            if config['method'] == 'ecfp':
                X_train_fps = descriptor_gen.get_ecfp_fingerprints(X_train_smiles, **config['params'])
                X_test_fps = descriptor_gen.get_ecfp_fingerprints(X_test_smiles, **config['params'])
            elif config['method'] == 'ecfp_counts':
                X_train_fps = descriptor_gen.get_ecfp_fingerprints(X_train_smiles, **config['params'])
                X_test_fps = descriptor_gen.get_ecfp_fingerprints(X_test_smiles, **config['params'])
            elif config['method'] == 'avalon':
                X_train_fps = descriptor_gen.get_avalon_fingerprints(X_train_smiles, **config['params'])
                X_test_fps = descriptor_gen.get_avalon_fingerprints(X_test_smiles, **config['params'])
            elif config['method'] == 'maccs':
                X_train_fps = descriptor_gen.get_maccs_fingerprints(X_train_smiles)
                X_test_fps = descriptor_gen.get_maccs_fingerprints(X_test_smiles)

            print(f"    Features generated: {X_train_fps.shape[1]}")

            # Keep all features
            X_train_var = X_train_fps
            X_test_var = X_test_fps
            print(f"    Features after variance threshold: {X_train_var.shape[1]} (all features kept)")

            # Model & algorithm comparison: Train on train, test on test
            print("    Model & algorithm comparison...")
            comparison_results = evaluate_multiple_algorithms_randomized(X_train_var, y_train, X_test_var, y_test)

            # Combine results (only comparison results)
            fp_results = {}
            for algo in comparison_results:
                fp_results[f"{algo}_Comparison"] = comparison_results[algo]
                fp_results[f"{algo}_Comparison"]['evaluation_type'] = 'comparison'

            # Add metadata
            for algo_name in fp_results:
                fp_results[algo_name]['fingerprint_type'] = fp_name
                fp_results[algo_name]['n_features'] = X_train_var.shape[1]
                fp_results[algo_name]['split_type'] = split_name
                fp_results[algo_name]['random_seed'] = None  # Other split types don't use random seeds

            split_results[fp_name] = fp_results

            # Save models for comparison evaluation (trained with hyperparameter tuning)
            print("    Saving models...")
            for algo in comparison_results:
                model_info = {
                    'model': comparison_results[algo]['model'],
                    'scaler': comparison_results[algo]['scaler'],
                    'best_params': comparison_results[algo]['best_params'],
                    'fingerprint_type': fp_name,
                    'fingerprint_params': config['params'],
                    'split_type': split_name,
                    'evaluation_type': 'comparison',
                    'test_r2': comparison_results[algo]['test_r2'],
                    'test_rmse': comparison_results[algo]['test_rmse'],
                    'cv_r2_mean': comparison_results[algo]['cv_r2_mean'],
                    'n_features': X_train_var.shape[1],
                    'random_seed': None
                }
                
                model_filename = f"{models_dir}/{split_name}_{fp_name}_{algo}_comparison.pkl"
                with open(model_filename, 'wb') as f:
                    pickle.dump(model_info, f)

            # Print summary for this fingerprint
            comparison_keys = [k for k in fp_results if 'Comparison' in k]
            if comparison_keys:
                best_algo_for_fp = max(comparison_keys, key=lambda x: fp_results[x]['test_r2'])
                best_r2_for_fp = fp_results[best_algo_for_fp]['test_r2']
                print(f"    Best algorithm: {best_algo_for_fp} (R² = {best_r2_for_fp:.3f})")
                
                # Save the best model for this split-fingerprint combination
                best_algo_name = best_algo_for_fp.replace('_Comparison', '')
                best_model_info = {
                    'model': fp_results[best_algo_for_fp]['model'],
                    'scaler': fp_results[best_algo_for_fp]['scaler'],
                    'best_params': fp_results[best_algo_for_fp]['best_params'],
                    'fingerprint_type': fp_name,
                    'fingerprint_params': config['params'],
                    'split_type': split_name,
                    'evaluation_type': 'comparison',
                    'test_r2': fp_results[best_algo_for_fp]['test_r2'],
                    'test_rmse': fp_results[best_algo_for_fp]['test_rmse'],
                    'cv_r2_mean': fp_results[best_algo_for_fp]['cv_r2_mean'],
                    'n_features': X_train_var.shape[1],
                    'random_seed': None
                }
                best_model_filename = f"{models_dir}/{split_name}_{fp_name}_BEST_MODEL.pkl"
                with open(best_model_filename, 'wb') as f:
                    pickle.dump(best_model_info, f)
                print(f"    Best model saved: {best_model_filename}")

        all_split_results[split_name] = split_results

    results_summary = []
    for split_name, split_results in all_split_results.items():
        for fp_name, fp_results in split_results.items():
            for algo_name, results in fp_results.items():
                summary_row = {
                    'Split_Type': split_name,
                    'Fingerprint': fp_name,
                    'Algorithm': algo_name.split('_')[0],
                    'Evaluation_Type': results['evaluation_type'],
                    'RandomSeed': results.get('random_seed', None),
                    'Test_R2': results.get('test_r2', results.get('cv_r2_mean', 0)),
                    'Test_RMSE': results.get('test_rmse', results.get('cv_rmse_mean', 0)),
                    'Test_MAE': results.get('test_mae', results.get('cv_mae_mean', 0)),
                    'CV_R2_Mean': results.get('cv_r2_mean', 0),
                    'CV_R2_Std': results.get('cv_r2_std', 0),
                    'Train_R2': results.get('train_r2', 0),
                    'Test_Pearson': results.get('test_pearson', 0),
                    'N_Features': results['n_features'],
                    'Needs_Scaling': results.get('scaler') is not None,
                    'Best_Params': str(results.get('best_params', {}))
                }
                results_summary.append(summary_row)

    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(results_file, index=False)

    print(f"Comprehensive results saved as: {results_file}")

    # Find and report the overall best model
    comparison_results = results_df[results_df['Evaluation_Type'] == 'comparison']
    if not comparison_results.empty:
        best_idx = comparison_results['Test_R2'].idxmax()
        best_result = comparison_results.loc[best_idx]
        print(f"\nOverall Best Model:")
        print(f"  Split: {best_result['Split_Type']}")
        print(f"  Fingerprint: {best_result['Fingerprint']}")
        print(f"  Algorithm: {best_result['Algorithm']}")
        print(f"  Test R²: {best_result['Test_R2']:.4f}")
        print(f"  Test RMSE: {best_result['Test_RMSE']:.4f}")
        print(f"  CV R² (mean ± std): {best_result['CV_R2_Mean']:.4f} ± {best_result['CV_R2_Std']:.4f}")
        print(f"  Saved as: {models_dir}/{best_result['Split_Type']}_{best_result['Fingerprint']}_BEST_MODEL.pkl")

    print(f"\nTotal models saved: {len([f for f in os.listdir(models_dir) if f.endswith('.pkl')])}")
    print(f"Models directory: {models_dir}/")

    print("\n" + "="*86)
    print("MULTI-SPLIT RANDOMIZED SEARCH QSAR MODELING COMPLETED!")
    print("="*86)

    return {'results': all_split_results, 'results_df': results_df, 'models_dir': models_dir}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Multi-Algorithm RET Mutant QSAR Models with Multiple Random Seeds"
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input CSV file containing SMILES and pIC50 data'
    )

    parser.add_argument(
        '--models-dir', '-m',
        default='saved_models',
        help='Directory path for saving trained models (default: saved_models)'
    )

    parser.add_argument(
        '--output', '-o',
        default='qsar_results.csv',
        help='Path for saving the comprehensive results CSV file (default: qsar_results.csv)'
    )

    args = parser.parse_args()

    # Run the main training pipeline
    results = main(args.input, args.models_dir, args.output)
