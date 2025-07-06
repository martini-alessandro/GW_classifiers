import pandas as pd


columns_to_drop =  [
                    'rho1','duration0','bandwidth0','frequency0','time1','chirp1','chirp3','chirp5',\
                    'snr2','time0','time2','time3','dtL','dtH','gps0','gps1','phi1','theta0','phi0',\
                    'theta1','lag0','lag1','factor0','Qveto3','era','strain1','size0', 'Lveto1','Lveto2'\
                    ]

def loadAndPreprocess(data_dir):
    """
    Preprocess the input DataFrame by removing rows with NaN values and resetting the index.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: The preprocessed DataFrame with NaN values removed and index reset.
    """
    df = pd.read_csv(data_dir)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df = df.drop(columns = columns_to_drop) 
    
    #Add columns containing XGBoost variables 
    df['noise'] = 1/(df['noise0']**2) + 1/(df['noise1'] ** 2) 
    df['mSNR/L'] = np.minimum(df['sSNR0'], df['sSNR1']) / df['likelihood'] 
    df['Qa'] =  np.sqrt(df['Qveto0'])
    df['Qp'] = df['Qveto1']/(2*np.sqrt(np.log10(np.minimum(200,df['ecor0']))))

    #Scale the data 
    df = df.drop(columns = ['noise0', 'noise1'])

    return df['class'], df.drop(coulmn = 'class')