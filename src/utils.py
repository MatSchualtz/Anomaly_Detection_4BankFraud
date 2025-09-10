import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def df_summary_with_fraud(df, target_col, id_cols_slice):
    """
    Retorna um resumo do dataframe, incluindo tipos, nulos e, para IDs, teste de associação com fraude.
    
    Parâmetros:
    df : pd.DataFrame
        Dataframe a ser analisado
    target_col : str
        Nome da coluna target (ex: 'isFraud')
    id_cols_slice : slice
        Slice das colunas de ID para fazer teste de associação
    """
    
    data = {
        'Data Name': [],
        'Data Type': [],
        'Data Null (A)': [],
        'Data Null (P%)': [],
        'Fraud Rate Null': [],
        'Fraud Rate Not Null': [],
        'P-value': []
    }
    

    id_vars = list(df.columns)[id_cols_slice]
    
    for col in id_vars:
        data['Data Name'].append(col)
        data['Data Type'].append(df[col].dtype)
        null_count = df[col].isnull().sum()
        data['Data Null (A)'].append(null_count)
        data['Data Null (P%)'].append(null_count / len(df) * 100)
        
        null_flag = df[col].isna().astype(int)
        fraud_rate_by_null = df.groupby(null_flag)[target_col].mean()
        contingency = pd.crosstab(null_flag, df[target_col])
        chi2, p, dof, ex = chi2_contingency(contingency)
        data['Fraud Rate Null'].append(fraud_rate_by_null.get(1, 0))
        data['Fraud Rate Not Null'].append(fraud_rate_by_null.get(0, 0))
        data['P-value'].append(p)
    
    summary = pd.DataFrame(data)
    summary = summary.sort_values(by='Data Null (A)', ascending=False)
    
    return summary


def low_importance_numerical(df, target_col, threshold=0.001, random_state=42):
    """
    Retorna lista de colunas numéricas de baixa importância usando RandomForest.

    Parâmetros:
    df : pd.DataFrame
        Dataframe completo (contendo target)
    target_col : str
        Nome da coluna target
    threshold : float
        Limite mínimo de importância para manter a coluna
    random_state : int
        Semente para reprodutibilidade
    """
    
    X = df.select_dtypes(include=['int64', 'float64']).drop(columns=[target_col, 'TransactionID'], errors='ignore')
    y = df[target_col]
    X_filled = X.fillna(-999)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_filled, y)
    
    # Importância das features
    importances = pd.Series(model.feature_importances_, index=X.columns)
    low_importance_cols = importances[importances < threshold].index.to_list()
    importance_df = importances.sort_values(ascending=False).reset_index()
    importance_df.columns = ['Feature', 'Importance']
    importance_df['cumulative_importance'] = importance_df['Importance'].cumsum()
    importance_df['contribution'] = importance_df['Importance']  # incremental contribution

    # Gráfico da importância acumulada
    plt.figure(figsize=(12,6))
    plt.plot(importance_df['Feature'], importance_df['cumulative_importance'], marker='o', color='steelblue')
    plt.axhline(y=0.9, color='red', linestyle='--', label='90% acumulado')
    plt.xticks(rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importância Acumulada")
    plt.title("Importância Acumulada das Features Numéricas")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return low_importance_cols, importance_df
