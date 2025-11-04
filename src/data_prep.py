import pandas as pd


class DataPreparationPipeline:
    """
    Classe com os preprocessamentos escolhidos para os dados.
    Atualmente inclui apenas a seleção de variáveis relevantes.
    """

    def __init__(self,df_transaction,df_identity):
        """
        Inicializa a classe com as listas de variáveis
        que devem ser mantidas em cada dataset.
        """
        self.transaction_features = [
            'TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt',
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
            'C1', 'C5', 'D2', 'D3', 'D8', 'D9', 'D10', 'D14', 'M1', 'M2', 'M3',
            'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V4', 'V12', 'V19', 'V23', 'V37',
            'V44', 'V46', 'V53', 'V56', 'V61', 'V66', 'V77', 'V86', 'V148',
            'V167', 'V170', 'V188', 'V194', 'V221', 'V242', 'V247', 'V250'
        ]

        self.identity_features = [
            'TransactionID', 'id_02', 'id_05', 'id_06', 'id_13', 'id_14', 'id_15',
            'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33',
            'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo'
        ]
        
        self.df_transaction = df_transaction
        
        self.df_identity = df_identity

    def feature_selection(self):
        """
        Seleciona apenas as colunas relevantes dos dataframes de transações e identidade.

        Args:
            df_transaction (pd.DataFrame): DataFrame original com dados de transações.
            df_identity (pd.DataFrame): DataFrame original com dados de identidade.

        Returns:
            df_transaction (pd.DataFrame): DataFrame de transações filtrado.
            df_identity (pd.DataFrame): DataFrame de identidade filtrado.
        """
        self.df_transaction = self.df_transaction[self.transaction_features].copy()
        self.df_identity = self.df_identity[self.identity_features].copy()

        return self
    
    def df_merge(self):
        merged_df = pd.merge(self.df_transaction, self.df_identity, on='TransactionID', how='left')
        
        merged_df.drop(columns='TransactionID', inplace= True)
        
        merged_df = merged_df[['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD',
            'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain','C1', 'C5', 'D2', 'D3', 'D8', 'D9', 'D10', 'D14', 'M1', 'M2', 'M3',
            'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V4', 'V12', 'V19', 'V23', 'V37','V44', 'V46', 'V53', 'V56', 'V61', 'V66', 'V77', 'V86', 'V148',
            'V167', 'V170', 'V188', 'V194', 'V221', 'V242', 'V247', 'V250', 'id_02', 'id_05', 'id_06', 'id_13', 'id_14', 'id_15',
            'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33','id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']]
        
        return merged_df


## FUNCAO FUTURA PARA CATEGORIZAR AS HORAS

def categorize_risk(hour):
    if 0 <= hour < 6:
        return 'Alto risco'  
    elif 6 <= hour < 12:
        return 'Médio risco'
    elif 12 <= hour < 18:
        return 'Baixo risco'
    else:
        return 'Médio/Alto risco'
    
    
# FUNCAO FUTURA PARA CALCULAR JANELAS MOVEIS

def compute_windows_one_group(times, values, windows):
    """
    times: np.array (sorted) de tempos em segundos para 1 card1
    values: np.array de TransactionAmt (float) correspondente
    windows: dict label->seconds
    retorna: dict[label] -> dict com arrays 'count','mean','std' (mesmo comprimento de times)
    """
    n = len(times)
    out = {label: {'count': np.zeros(n, dtype=int),
                   'mean' : np.zeros(n, dtype=float),
                   'std'  : np.zeros(n, dtype=float)}
           for label in windows}
    if n == 0:
        return out

    psum = np.concatenate(([0.0], np.cumsum(values, dtype=float)))    # len n+1
    psum2 = np.concatenate(([0.0], np.cumsum(values * values, dtype=float)))

    idx = np.arange(n)

    for label, w in windows.items():
        left_idx = np.searchsorted(times, times - w, side='left')

        counts = idx - left_idx + 1  
        sums = psum[idx + 1] - psum[left_idx]
        sums2 = psum2[idx + 1] - psum2[left_idx]

        means = sums / counts
        vars_ = (sums2 / counts) - (means * means)
        vars_[vars_ < 0] = 0.0
        stds = np.sqrt(vars_)

        out[label]['count'] = counts
        out[label]['mean'] = means
        out[label]['std'] = stds

    return out
