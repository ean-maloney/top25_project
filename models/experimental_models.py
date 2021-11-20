import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../libraries')
from ordinal.ordinal import get_error

def data_prep(next_week):
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
    from sklearn.linear_model import LinearRegression

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
    r2 = r2_score(output['Actual Rank'], output['Adj. Predicted Rank'])
    total_error = get_error(output['Adj. Predicted Rank'], output['Actual Rank'], mode = 'avg')

    if get_table == True:
        return r2, total_error, output
   
    else:
        return r2, total_error

def run_rf(rf_params, df, scaled = False, get_table = False):
    from sklearn.ensemble import RandomForestRegressor

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
    r2 = r2_score(output['Actual Rank'], output['Adj. Predicted Rank'])
    total_error = get_error(output['Adj. Predicted Rank'], output['Actual Rank'], mode = 'avg')

    if get_table == True:
        return r2, total_error, output
   
    else:
        return r2, total_error

def run_ab(ab_params, df, scaled = False, get_table = False):
    from sklearn.ensemble import AdaBoostRegressor as abr

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
    
    #AdaBoost Regression
    ab_model = abr(random_state = 8).fit(X_train, y_train)
    ab_model_output = ab_model.predict(X_test)
    
    output = build_output(ab_model_output, y_test)

    #Get metrics 
    r2 = r2_score(output['Actual Rank'], output['Adj. Predicted Rank'])
    total_error = get_error(output['Adj. Predicted Rank'], output['Actual Rank'], mode = 'avg')

    if get_table == True:
        return r2, total_error, output
   
    else:
        return r2, total_error

def build_output(y_pred, y_actual):
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

def run_bin(bin_params, df, scaled = False, get_table = False, model = 'forest'):
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier

    #Unpack params
    feature_drop = bin_params['feature_drop']
    
    #Drop null rows (teams with byes)
    df.dropna(inplace = True)
    
    #Drop irrelevant cols
    df = df.drop(columns = feature_drop)

    #Convert cols to num
    for name, values in df.iteritems():
        df[name] = pd.to_numeric(values)
    
    #Bin ranks
    for index, row in df.iterrows():
        if row['Rank'] <= 5:
            df.at[index, 'Rank'] = 0
        elif row['Rank'] <= 10:
            df.at[index, 'Rank'] = 1
        elif row['Rank'] <= 15:
            df.at[index, 'Rank'] = 2
        elif row['Rank'] <= 20:
            df.at[index, 'Rank'] = 3
        elif row['Rank'] <= 25:
            df.at[index, 'Rank'] = 4
        else:
            df.at[index, 'Rank'] = 5
            
    for index, row in df.iterrows():
        if row['Next Week Rank'] <= 5:
            df.at[index, 'Next Week Rank'] = 0
        elif row['Next Week Rank'] <= 10:
            df.at[index, 'Next Week Rank'] = 1
        elif row['Next Week Rank'] <= 15:
            df.at[index, 'Next Week Rank'] = 2
        elif row['Next Week Rank'] <= 20:
            df.at[index, 'Next Week Rank'] = 3
        elif row['Next Week Rank'] <= 25:
            df.at[index, 'Next Week Rank'] = 4
        else:
            df.at[index, 'Next Week Rank'] = 5 

    X = df.drop(columns = ['Next Week Rank'])
    y = df['Next Week Rank']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8)
    
    if scaled == True:
        X_train, X_test = scale(X_train, X_test)

    if model == 'tree':
        #Decision Tree Classification
        tree_model = tree.DecisionTreeClassifier(random_state = 8).fit(X_train, y_train)
        model_output = tree_model.predict(X_test)

    if model == 'forest':
        rfc_model = RandomForestClassifier(random_state = 8).fit(X_train, y_train)
        model_output = rfc_model.predict(X_test)

    #Get metrics 
    accuracy = accuracy_score(y_test, model_output)
    total_error = get_error(model_output, y_test.tolist(), mode = 'avg')
   
    if get_table == True:
        table = pd.DataFrame()
        table['Predictions'] = model_output
        table['Actual'] = y_test.tolist()
        return accuracy, total_error, table
   
    else:
        return accuracy, total_error

def run_movement(movement_params, df, scaled = False, get_table = False, model = 'forest'):
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier

    #Unpack params
    feature_drop = movement_params['feature_drop']
    
    #Drop null rows (teams with byes)
    df.dropna(inplace = True)
    
    #Drop irrelevant cols
    df = df.drop(columns = feature_drop)

    #Convert cols to num
    for name, values in df.iteritems():
        df[name] = pd.to_numeric(values)

    #Group rankings to categories
    for index, row in df.iterrows():
        if row['Next Week Rank'] == 26:
            df.at[index, 'Next Week Rank Category'] = 0
        elif row['Next Week Rank'] < row['Rank']:
            df.at[index, 'Next Week Rank Category'] = 1
        elif row['Next Week Rank'] > row['Rank']:
            df.at[index, 'Next Week Rank Category'] = 2
        elif row['Next Week Rank'] == row['Rank']:
            df.at[index, 'Next Week Rank Category'] = 3        

    X = df.drop(columns = ['Next Week Rank Category'])
    y = df['Next Week Rank Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8)

    if scaled == True:
        X_train, X_test = scale(X_train, X_test)

    if model == 'tree':
        #Decision Tree Classification
        tree_model = tree.DecisionTreeClassifier(random_state = 8).fit(X_train, y_train)
        model_output = tree_model.predict(X_test)

    if model == 'forest':
        rfc_model = RandomForestClassifier(random_state = 8).fit(X_train, y_train)
        model_output = rfc_model.predict(X_test)
    
    if model == 'svc':
        from sklearn.svm import SVC
        svc_model = SVC(random_state = 8).fit(X_train, y_train)
        model_output = svc_model.predict(X_test)

    #Get metrics 
    accuracy = accuracy_score(y_test, model_output)
    total_error = get_error(model_output, y_test.tolist(), mode = 'avg')
   
    if get_table == True:
        table = pd.DataFrame()
        table['Predictions'] = model_output
        table['Actual'] = y_test.tolist()
        return accuracy, total_error, table
   
    else:
        return accuracy, total_error

    

    
    


    






    
