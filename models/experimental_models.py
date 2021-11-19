def data_prep(next_week):
    import pandas as pd

    #Set file path
    file = "../tables/2021AP.xlsx"

    #Read in df
    df = pd.read_excel(file)
    
    #Drop footer
    df.drop(index = (df.index.stop - 1), inplace = True)

    #Drop overflow
    for index, row in df.iterrows():
        if row['Week'] >= next_week:
            df.drop(index = index, inplace = True)

    #Encode Result column
    for index, row in df.iterrows():
        if row['Result'] == 'W':
            df.at[index, 'Result'] = 1
        else: 
            df.at[index, 'Result'] = 0
    
    return df    

def scale(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    
    # Create a StandardScaler instances
    scaler = StandardScaler()

    # Fit the StandardScaler
    X_scaler = scaler.fit(X_train)

    # Scale the data
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)  

    return X_train_scaled, X_test_scaled

def run_mlr(mlr_params, df, scaled = False, get_table = False):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    import sys
    sys.path.append('../libraries')

    from ordinal.ordinal import get_error

    #Unpack params
    feature_drop = mlr_params['feature_drop']
    #max_rank = mlr_params['max_rank']
    
    #Drop null rows (teams with byes)
    df.dropna(inplace = True)
    
    #Drop irrelevant cols
    df = df.drop(columns = feature_drop)
    
    #Convert cols to num
    for name, values in df.iteritems():
        df[name] = pd.to_numeric(values)

    X = df.drop(columns = ['Next Week Rank'])
    y = df['Next Week Rank']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8)
    
    if scaled == True:
        X_train, X_test = scale(X_train, X_test)
    
    #Multiple Linear Regression
    lin_model = LinearRegression().fit(X_train, y_train)
    lin_model_output = lin_model.predict(X_test)
    
    output = build_output(lin_model_output, y_test)

    #Get metrics 
    r2 = r2_score(output['Adj. Predicted Rank'], output['Actual Rank'])
    total_error = get_error(output['Adj. Predicted Rank'], output['Actual Rank'], mode = 'avg')

    if get_table == True:
        return r2, total_error, output
   
    else:
        return r2, total_error

def run_rf(rf_params, df, scaled = False, get_table = False):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    import sys
    sys.path.append('../libraries')

    from ordinal.ordinal import get_error

    #Unpack params
    feature_drop = rf_params['feature_drop']
    #max_rank = rf_params['max_rank']
    
    #Drop null rows (teams with byes)
    df.dropna(inplace = True)
    
    #Drop irrelevant cols
    df = df.drop(columns = feature_drop)
    
    #Convert cols to num
    for name, values in df.iteritems():
        df[name] = pd.to_numeric(values)

    X = df.drop(columns = ['Next Week Rank'])
    y = df['Next Week Rank']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8)
    
    if scaled == True:
        X_train, X_test = scale(X_train, X_test)
    
    #Multiple Linear Regression
    rf_model = RandomForestRegressor(random_state = 8).fit(X_train, y_train)
    rf_model_output = rf_model.predict(X_test)
    
    output = build_output(rf_model_output, y_test)

    #Get metrics 
    r2 = r2_score(output['Adj. Predicted Rank'], output['Actual Rank'])
    total_error = get_error(output['Adj. Predicted Rank'], output['Actual Rank'], mode = 'avg')

    if get_table == True:
        return r2, total_error, output
   
    else:
        return r2, total_error

def run_ab(ab_params, df, scaled = False, get_table = False):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostRegressor as abr
    from sklearn.metrics import r2_score

    import sys
    sys.path.append('../libraries')

    from ordinal.ordinal import get_error

    #Unpack params
    feature_drop = ab_params['feature_drop']
    #max_rank = rf_params['max_rank']
    
    #Drop null rows (teams with byes)
    df.dropna(inplace = True)
    
    #Drop irrelevant cols
    df = df.drop(columns = feature_drop)
    
    #Convert cols to num
    for name, values in df.iteritems():
        df[name] = pd.to_numeric(values)

    X = df.drop(columns = ['Next Week Rank'])
    y = df['Next Week Rank']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8)
    
    if scaled == True:
        X_train, X_test = scale(X_train, X_test)
    
    #Multiple Linear Regression
    ab_model = abr(random_state = 8).fit(X_train, y_train)
    ab_model_output = ab_model.predict(X_test)
    
    output = build_output(ab_model_output, y_test)

    #Get metrics 
    r2 = r2_score(output['Adj. Predicted Rank'], output['Actual Rank'])
    total_error = get_error(output['Adj. Predicted Rank'], output['Actual Rank'], mode = 'avg')

    if get_table == True:
        return r2, total_error, output
   
    else:
        return r2, total_error

def build_output(y_pred, y_actual):
    import pandas as pd

    output = pd.DataFrame()
    output['Predicted Rank'] = y_pred.tolist()
    output['Actual Rank'] = y_actual.tolist()
    
    output['Predicted Rank'] = pd.to_numeric(output['Predicted Rank'], errors = 'coerce')
    output['Actual Rank'] = pd.to_numeric(output['Actual Rank'])
    
    adj_ranks = []
    for item in output['Predicted Rank']:
        if item <= 25:
            adj_ranks.append(item)

        else:
            adj_ranks.append(26)

    output['Adj. Predicted Rank'] = adj_ranks
    
    return output