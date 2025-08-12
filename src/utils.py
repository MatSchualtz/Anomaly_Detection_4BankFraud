import pandas as pd 

def df_summary(df):
    data = {
        'Data Name': [],
        'Data Type': [],
        'Data Null (A)': [],
        'Data Null (P%)': []
    }
    
    for col in df.columns:
        data['Data Name'].append(col)
        data['Data Type'].append(df[col].dtype)
        null = df[col].isnull().sum()
        data['Data Null (A)'].append(null)
        data['Data Null (P%)'].append(null / len(df) * 100)
    
    summary = pd.DataFrame(data)
    return summary.sort_values(by='Data Null (A)', ascending=False)