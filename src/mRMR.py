import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer

def mRMR_feature_selection(df:pd.DataFrame, target_column:str, n_features_to_select:int=10, n_bins:int=5, random_state:int=42):
    if target_column not in df.columns:
        raise ValueError(f"Objetivo '{target_column}' no encontrado en el DataFrame.")

    # Solo trabajamos con datos numéricos
    df_numeric = df.select_dtypes(include=[np.number])
    
    y = df_numeric[target_column].values
    X = df_numeric.drop(columns=[target_column])
    feature_names = X.columns.tolist()

    if len(feature_names) == 0:
        raise ValueError("No hay columnas numéricas suficientes para realizar la selección.")

    import warnings
    # Discretización (Mejora el cálculo de Información Mutua)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        X_disc = discretizer.fit_transform(X.values)

    # Relevancia (MI con el objetivo)
    mi_target = mutual_info_regression(X_disc, y, random_state=random_state)
    mi_series = pd.Series(mi_target, index=feature_names)

    selected = []
    remaining = feature_names.copy()

    # Selección iterativa
    while len(selected) < min(n_features_to_select, len(feature_names)):
        if not selected:
            # La primera es la más relevante
            best_feat = mi_series.idxmax()
        else:
            scores = {}
            for feat in remaining:
                relevance = mi_series[feat]
                
                # Redundancia (promedio MI con las ya seleccionadas)
                redundancy_vals = []
                idx_feat = feature_names.index(feat)
                for sel_feat in selected:
                    idx_sel = feature_names.index(sel_feat)
                    mi_redundancy = mutual_info_regression(
                        X_disc[:, [idx_feat]], X_disc[:, idx_sel], random_state=random_state
                    )[0]
                    redundancy_vals.append(mi_redundancy)
                
                redundancy = np.mean(redundancy_vals)
                scores[feat] = relevance - redundancy
            
            best_feat = max(scores, key=scores.get)

        selected.append(best_feat)
        remaining.remove(best_feat)
        
    return selected
