import pandas as pd
import numpy as np
from tqdm import tqdm

class DataPreparationPipeline:
    """
    Pipeline completo para preparação de dados em detecção de fraude
    Complete data preparation pipeline for fraud detection
    """
    
    def __init__(self, df_transaction: pd.DataFrame, df_identity: pd.DataFrame, 
                 is_training: bool = True, fitted_params: dict = None):
        """
        Inicializa o pipeline com dados de transação e identidade
        Initializes pipeline with transaction and identity data
        
        Args:
            df_transaction: DataFrame com dados transacionais / Transaction data
            df_identity: DataFrame com dados de identidade / Identity data  
            is_training: Se é dados de treino / Whether it's training data
            fitted_params: Parâmetros aprendidos no treino / Parameters learned during training
        """
        # Variáveis base para processamento
        # Base variables for processing
        self.transaction_features = [
            'TransactionID', 'TransactionDT', 'TransactionAmt',
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
            'M7', 'M8', 'M9'
        ]

        # Inclui target apenas no treino
        # Include target only in training
        if is_training:
            self.transaction_features.append('isFraud')

        self.identity_features = ['TransactionID', 'DeviceType', 'DeviceInfo']

        # DataFrames de trabalho
        # Working DataFrames
        self.df_transaction = df_transaction
        self.df_identity = df_identity
        self.merged_df = pd.DataFrame()
        self.df_final = pd.DataFrame()
        
        # Configuração do modo
        # Mode configuration
        self.is_training = is_training
        self.fitted_params = fitted_params if fitted_params else {}
        
        # Parâmetros aprendidos durante treinamento
        # Parameters learned during training
        self.fraud_rates_dict_ = self.fitted_params.get('fraud_rates_dict', {})
        self.email_popular_domains_ = self.fitted_params.get('email_popular_domains', {'P': set(), 'R': set()})
        self.email_fraud_rates_ = self.fitted_params.get('email_fraud_rates', {'P_high_risk': set(), 'R_high_risk': set()})
        self.global_fraud_rate_ = self.fitted_params.get('global_fraud_rate', 0)

    def feature_selection(self):
        """
        Seleciona features relevantes baseado na lista definida
        Selects relevant features based on defined list
        """
        # Filtra colunas existentes nos dados
        # Filters existing columns in data
        tx_cols = [c for c in self.transaction_features if c in self.df_transaction.columns]
        id_cols = [c for c in self.identity_features if c in self.df_identity.columns]

        self.df_transaction = self.df_transaction[tx_cols].copy()
        self.df_identity = self.df_identity[id_cols].copy()

        return self

    def df_merge(self):
        """
        Combina dados de transação e identidade
        Combines transaction and identity data
        """
        self.merged_df = pd.merge(self.df_transaction, self.df_identity, on='TransactionID', how='left')

        # Define ordem das colunas
        # Defines column order
        cols_order = [
            'TransactionID', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
            'TransactionDT', 'TransactionAmt', 'ProductCD', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'DeviceType', 'DeviceInfo'
        ]
        
        # Insere isFraud na posição correta se existir
        # Inserts isFraud in correct position if exists
        if 'isFraud' in self.merged_df.columns:
            cols_order.insert(2, 'isFraud')
            
        cols_order = [c for c in cols_order if c in self.merged_df.columns]
        self.merged_df = self.merged_df[cols_order].copy()

        return self

    def _fill_missing_categorical(self, df):
        """Preenche valores categóricos faltantes / Fills missing categorical values"""
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            df.loc[:, cat_cols] = df.loc[:, cat_cols].fillna('missing')
        return df

    def _card_features(self, df_local):
        """Transforma colunas de cartão em indicadores / Transforms card columns into indicators"""
        CARD_COLUMNS = ['card2', 'card3', 'card4', 'card5', 'card6']
        CARD_COLUMNS = [c for c in CARD_COLUMNS if c in df_local.columns]
        for card_column in CARD_COLUMNS:
            df_local[f'{card_column}_missing'] = df_local[card_column].isna().astype(int)
            df_local.drop(columns=card_column, inplace=True)
        return df_local

    def _prob_cond_features(self, df_local):
        """
        Substitui categorias por taxas de fraude históricas
        Replaces categories with historical fraud rates
        """
        CATEGORICAL_COLUMNS = ['ProductCD', 'M1', 'M2', 'M3', 'M4', 'M6', 'M7', 'M8', 'M9']
        CATEGORICAL_COLUMNS = [c for c in CATEGORICAL_COLUMNS if c in df_local.columns]
        
        if self.is_training and 'isFraud' in df_local.columns:
            # Calcula taxas no treino
            # Calculates rates in training
            for cat_col in CATEGORICAL_COLUMNS:
                fraud_rate = df_local.groupby(cat_col)['isFraud'].mean()
                self.fraud_rates_dict_[cat_col] = fraud_rate.to_dict()
                df_local[f'fraud_{cat_col}_rate'] = df_local[cat_col].map(self.fraud_rates_dict_[cat_col])
                
            self.global_fraud_rate_ = df_local['isFraud'].mean()
        else:
            # Aplica taxas no teste
            # Applies rates in test
            for cat_col in CATEGORICAL_COLUMNS:
                if cat_col in self.fraud_rates_dict_:
                    df_local[f'fraud_{cat_col}_rate'] = df_local[cat_col].map(self.fraud_rates_dict_[cat_col])
                else:
                    df_local[f'fraud_{cat_col}_rate'] = self.global_fraud_rate_
        
        # Remove colunas originais
        # Removes original columns
        drop_cols = CATEGORICAL_COLUMNS + ['M5']
        drop_cols = [c for c in drop_cols if c in df_local.columns]
        df_local.drop(columns=drop_cols, inplace=True)
        
        return df_local

    def _create_email_features(self, df_local, p_col='P_emaildomain', r_col='R_emaildomain', 
                             popular_threshold=0.01, high_risk_margin=0.02):
        """
        Engenharia de features para domínios de email
        Feature engineering for email domains
        """
        df_local = df_local.copy()
        
        # Limpeza de dados
        # Data cleaning
        df_local[p_col] = df_local[p_col].astype(str).str.lower().str.strip().replace({'missing': np.nan})
        df_local[r_col] = df_local[r_col].astype(str).str.lower().str.strip().replace({'missing': np.nan})

        # Indicadores básicos
        # Basic indicators
        df_local['missing_p_domain'] = df_local[p_col].isna().astype(int)
        df_local['missing_r_domain'] = df_local[r_col].isna().astype(int)
        df_local['Same_Email'] = (df_local[p_col] == df_local[r_col]) & df_local[p_col].notna()

        if self.is_training and 'isFraud' in df_local.columns:
            # Aprendizado durante treino
            # Learning during training
            p_counts = df_local[p_col].value_counts(normalize=True, dropna=True)
            r_counts = df_local[r_col].value_counts(normalize=True, dropna=True)

            self.email_popular_domains_['P'] = set(p_counts[p_counts >= popular_threshold].index)
            self.email_popular_domains_['R'] = set(r_counts[r_counts >= popular_threshold].index)

            # Cálculo de risco
            # Risk calculation
            tmp = df_local.copy()
            tmp[p_col] = tmp[p_col].fillna('missing')
            tmp[r_col] = tmp[r_col].fillna('missing')

            p_domain_fraud = tmp.groupby(p_col)['isFraud'].mean()
            r_domain_fraud = tmp.groupby(r_col)['isFraud'].mean()

            global_rate = df_local['isFraud'].mean()
            self.global_fraud_rate_ = global_rate
            self.email_fraud_rates_['P_high_risk'] = set(p_domain_fraud[p_domain_fraud > (global_rate + high_risk_margin)].index)
            self.email_fraud_rates_['R_high_risk'] = set(r_domain_fraud[r_domain_fraud > (global_rate + high_risk_margin)].index)

        # Aplicação das features
        # Feature application
        df_local['P_email_popular'] = df_local[p_col].isin(self.email_popular_domains_['P']).astype(int)
        df_local['R_email_popular'] = df_local[r_col].isin(self.email_popular_domains_['R']).astype(int)

        def consistency_group_row(row):
            """Classifica consistência entre emails / Classifies email consistency"""
            if row['P_email_popular'] and row['R_email_popular']:
                return 3
            elif row['P_email_popular'] and not row['R_email_popular']:
                return 1
            elif not row['P_email_popular'] and row['R_email_popular']:
                return 2
            else:
                return 0

        df_local['email_consistency_group'] = df_local.apply(consistency_group_row, axis=1)

        # Features de risco
        # Risk features
        df_local['high_risk_p_domain'] = df_local[p_col].isin(self.email_fraud_rates_['P_high_risk']).astype(int)
        df_local['high_risk_r_domain'] = df_local[r_col].isin(self.email_fraud_rates_['R_high_risk']).astype(int)

        # Remove colunas intermediárias
        # Removes intermediate columns
        drop_cols = [c for c in [p_col, r_col, 'P_email_popular', 'R_email_popular'] if c in df_local.columns]
        df_local = df_local.drop(columns=drop_cols)

        return df_local

    def _create_device_features(self, df_local):
        """
        Features derivadas de dispositivo
        Device-derived features
        """
        df_local = df_local.copy()
        
        # Tipo de dispositivo
        # Device type
        if 'DeviceType' in df_local.columns:
            df_local["is_mobile"] = df_local["DeviceType"].astype(str).str.lower().eq("mobile").astype(int)
        else:
            df_local["is_mobile"] = 0

        # Informação do dispositivo
        # Device information
        df_local['DeviceInfo'] = df_local.get('DeviceInfo', pd.Series(dtype=object)).astype(str).str.lower().replace({'nan': None})
        df_local['DeviceInfo_clean'] = df_local['DeviceInfo'].str.extract(
            r'(windows|ios|mac|android|samsung|huawei|linux)', expand=False
        ).fillna('other')

        # Classificação de risco baseada em fabricante
        # Risk classification based on manufacturer
        high_risk = ["huawei"]
        medium_risk = ["samsung", "windows", "ios"]
        low_risk = ["android", "mac", "linux", "other"]

        def classify_risk(device):
            if pd.isna(device) or device is None:
                return "missing"
            device = str(device).lower()
            if any(r in device for r in high_risk):
                return "alto"
            elif any(r in device for r in medium_risk):
                return "medio"
            elif any(r in device for r in low_risk):
                return "baixo"
            else:
                return "outro"

        df_local["device_risk_level"] = df_local["DeviceInfo_clean"].apply(classify_risk)
        df_local["is_high_risk_device"] = (df_local["device_risk_level"] == "alto").astype(int)

        # Mudanças de dispositivo
        # Device changes
        if 'card1' in df_local.columns and 'TransactionID' in df_local.columns:
            df_local = df_local.sort_values(['card1', 'TransactionID'])
            changed = (
                df_local.groupby('card1')
                .apply(lambda g: ((g['DeviceInfo_clean'] != g['DeviceInfo_clean'].shift()) & g['DeviceInfo'].notna() & g['DeviceInfo'].shift().notna()).astype(int))
                .reset_index(level=0, drop=True)
            )
            df_local['changed_device'] = changed
            df_local.loc[df_local.groupby('card1').head(1).index, 'changed_device'] = 0
        else:
            df_local['changed_device'] = 0

        df_local['device_missing'] = df_local['DeviceInfo'].apply(lambda x: (str(x) == 'missing')).astype(int)

        # Remove colunas originais
        # Removes original columns
        drop_cols = [c for c in ['DeviceType', 'DeviceInfo', 'DeviceInfo_clean', 'device_risk_level'] if c in df_local.columns]
        df_local.drop(columns=drop_cols, inplace=True)

        return df_local

    def _date_features(self, df_local):
        """
        Extrai features temporais de TransactionDT
        Extracts temporal features from TransactionDT
        """
        df_local = df_local.copy()
        
        if 'TransactionDT2' not in df_local.columns and 'TransactionDT' in df_local.columns:
            df_local['TransactionDT2'] = df_local['TransactionDT']

        # Decomposição temporal
        # Temporal decomposition
        df_local['TransactionDT2'] = pd.to_timedelta(df_local['TransactionDT2'], unit='s')
        date_components = df_local['TransactionDT2'].dt.components.iloc[:, :4]
        date_components['hour_fractional'] = (
            date_components['hours'] +
            date_components['minutes'] / 60 +
            date_components['seconds'] / 3600
        )

        df_local = df_local.assign(
            day=date_components['days'],
            hour=date_components['hours'],
            minute=date_components['minutes'],
            second=date_components['seconds'],
            hour_fractional=date_components['hour_fractional']
        )

        # Features cíclicas para hora
        # Cyclical features for hour
        df_local['hour_sin'] = np.sin(2 * np.pi * df_local['hour_fractional'] / 24)
        df_local['hour_cos'] = np.cos(2 * np.pi * df_local['hour_fractional'] / 24)

        # Períodos de alto risco
        # High risk periods
        df_local['is_high_risk_hour'] = (
            ((df_local['hour'] > 0) & (df_local['hour'] < 6)) |
            ((df_local['hour'] > 18) & (df_local['hour'] < 24))
        ).astype(int)

        # Remove componentes intermediários
        # Removes intermediate components
        drop_cols = [c for c in ['TransactionDT2', 'day', 'hour', 'minute', 'second', 'hour_fractional'] if c in df_local.columns]
        df_local.drop(columns=drop_cols, inplace=True)

        return df_local

    def _transaction_features(self, df_local):
        """
        Estatísticas temporais por cartão em múltiplas janelas
        Temporal statistics per card in multiple windows
        """
        df_local = df_local.sort_values(['card1', 'TransactionDT']).reset_index(drop=True)
        
        # Define todas as janelas temporais
        # Defines all temporal windows
        windows = {
            '90d': 90 * 24 * 3600,
            '30d': 30 * 24 * 3600,
            '15d': 15 * 24 * 3600,
            '7d': 7 * 24 * 3600,
            '24h': 24 * 3600,
            '6h': 6 * 3600,
            '1h': 3600,
            '15m': 15 * 60,
            '5m': 5 * 60
        }
        
        def compute_windows_one_group(times, values, windows):
            """Calcula estatísticas para múltiplas janelas / Computes statistics for multiple windows"""
            n = len(times)
            out = {
                label: {'count': np.zeros(n, dtype=int), 'mean': np.zeros(n, dtype=float), 'std': np.zeros(n, dtype=float)}
                for label in windows
            }
            if n == 0:
                return out

            # Pré-cálculo para eficiência
            # Pre-calculation for efficiency
            psum = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
            psum2 = np.concatenate(([0.0], np.cumsum(values * values, dtype=float)))

            idx = np.arange(n)
            for label, w in windows.items():
                # Encontra limites da janela
                # Finds window boundaries
                left_idx = np.searchsorted(times, times - w, side='left')
                counts = idx - left_idx + 1
                
                counts_safe = counts.copy()
                counts_safe[counts_safe <= 0] = 1

                # Cálculo incremental
                # Incremental calculation
                sums = psum[idx + 1] - psum[left_idx]
                sums2 = psum2[idx + 1] - psum2[left_idx]

                means = sums / counts_safe
                vars_ = (sums2 / counts_safe) - (means * means)
                vars_[vars_ < 0] = 0.0
                stds = np.sqrt(vars_)

                out[label]['count'] = counts
                out[label]['mean'] = means
                out[label]['std'] = stds

            return out

        # Inicializa colunas para todas as janelas
        # Initializes columns for all windows
        for label in windows:
            df_local[f'total_transaction_{label}'] = np.nan
            df_local[f'mean_transaction_value_{label}'] = np.nan
            df_local[f'std_transaction_value_{label}'] = np.nan

        if 'card1' not in df_local.columns:
            return df_local

        # Processa cada cartão
        # Processes each card
        groups = df_local.groupby('card1', sort=False)
        for card, g in tqdm(groups, total=df_local['card1'].nunique(), desc='Calculando estatísticas por cartão'):
            idx = g.index.to_numpy()
            times = g['TransactionDT'].to_numpy()
            vals = g['TransactionAmt'].to_numpy(dtype=float)

            order = np.argsort(times)
            times_o = times[order]
            vals_o = vals[order]
            idx_o = idx[order]

            res = compute_windows_one_group(times_o, vals_o, windows)

            for label in windows:
                df_local.loc[idx_o, f'total_transaction_{label}'] = res[label]['count']
                df_local.loc[idx_o, f'mean_transaction_value_{label}'] = res[label]['mean']
                df_local.loc[idx_o, f'std_transaction_value_{label}'] = res[label]['std']

        # Remove colunas processadas
        # Removes processed columns
        drop_cols = [c for c in ['TransactionAmt', 'TransactionDT'] if c in df_local.columns]
        df_local.drop(columns=drop_cols, inplace=True)

        return df_local

    def get_fitted_params(self):
        """Retorna parâmetros aprendidos / Returns learned parameters"""
        return {
            'fraud_rates_dict': self.fraud_rates_dict_,
            'email_popular_domains': self.email_popular_domains_,
            'email_fraud_rates': self.email_fraud_rates_,
            'global_fraud_rate': self.global_fraud_rate_
        }

    def feature_engineering(self, info_df: pd.DataFrame = None):
        """
        Executa pipeline completo de engenharia de features
        Executes complete feature engineering pipeline
        """
        df = self.merged_df.copy()

        # Preprocessamento básico
        # Basic preprocessing
        df = self._fill_missing_categorical(df)

        # Aplica transformações em sequência
        # Applies transformations in sequence
        df = self._card_features(df)
        df = self._prob_cond_features(df)
        df = self._create_email_features(df)
        df = self._create_device_features(df)
        df = self._date_features(df)
        df = self._transaction_features(df)

        # Ordem final das colunas
        # Final column order
        final_cols = ['TransactionID', 'card1']
        if 'isFraud' in df.columns:
            final_cols.append('isFraud')
        
        other_cols = [col for col in df.columns if col not in final_cols]
        final_cols.extend(other_cols)
        
        self.df_final = df[final_cols].copy()
        return self.df_final