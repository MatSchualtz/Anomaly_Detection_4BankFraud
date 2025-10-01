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
        return merged_df
