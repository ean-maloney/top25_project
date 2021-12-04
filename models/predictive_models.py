import pandas as pd
from sklearn.metrics import r2_score

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

def build_output(max_rank, next_teams, next_rank_act, byes_index, model_output):
    #Add back teams with byes to output
    next_rank_pred = model_output.tolist()
    for item in byes_index:
        next_rank_pred.insert(item, 'bye')
        
    #Build output col for pred rank
    model_output = model_output.tolist()

    for item in byes_index:
        model_output.insert(item + 1, item + 1)
    
    model_output.sort()    

    sorted_output = model_output
        
    #Build output df
    df2 = pd.DataFrame()
    df2['Predicted Rank'] = sorted_output
    df2['Predicted Ordinal Rank'] = range(1, 26)
    
    results = pd.DataFrame()
    results['Team'] = next_teams
    results['Predicted Rank'] = next_rank_pred
    results['Actual Rank'] = next_rank_act

    results.reset_index(inplace = True, drop = True)

    #Predict that teams with byes don't change rank
    for index, row in results.iterrows():
        if row['Predicted Rank'] == 'bye':
            results.at[index, 'Predicted Rank'] = index + 1

    results['Predicted Rank'] = pd.to_numeric(results['Predicted Rank'], errors = 'coerce')
    results['Actual Rank'] = pd.to_numeric(results['Actual Rank'])
    results['Previous Rank'] = range(1, 26)

    merged_results = results.merge(df2, how = 'left', left_on = 'Predicted Rank', right_on = 'Predicted Rank')
    
    apor_col = []
    
    #Convert teams with real number rank > 25 to max_rank
    for index, row in merged_results.iterrows():
        if row['Predicted Rank'] > 25:
            apor_col.append(max_rank)

        else:
            apor_col.append(row['Predicted Ordinal Rank'])
    
    merged_results['Adj. Predicted Ordinal Rank'] = apor_col
    
    merged_results = merged_results[['Team', 'Previous Rank', 'Predicted Rank', 'Predicted Ordinal Rank', 'Adj. Predicted Ordinal Rank', 'Actual Rank']]

    return merged_results

def run_mlr(mlr_params, df, next_week, get_table = False):
    from sklearn.linear_model import LinearRegression

    #Unpack params
    feature_drop = mlr_params['feature_drop']
    max_rank = mlr_params['max_rank']
    
    #Get next week teams list
    next_teams = df['Team'].loc[df['Week'] == next_week - 1]
    next_rank_act = df['Next Week Rank'].loc[df['Week'] == next_week - 1]

    #Get byes
    byes_index = []
    for index, row in df.iterrows():
        if row['Week'] == next_week - 1 and not(row['Opp. Rank'] >= 0):
            byes_index.append(index % 25)
    
    #Drop null rows (teams with byes)
    df.dropna(inplace = True)
    
    #Drop irrelevant cols
    df = df.drop(columns = feature_drop)
    
    #Convert cols to num
    for name, values in df.iteritems():
        df[name] = pd.to_numeric(values)
        
    #Set training data to Week < next week
    train_df = df.loc[df['Week'] < next_week - 1]
    X_train = train_df.drop(columns = ['Next Week Rank'])
    y_train = train_df['Next Week Rank']
    
    #Set testing data to Week = next week
    test_df = df.loc[df['Week'] == next_week - 1]
    X_test = test_df.drop(columns = ['Next Week Rank'])
    y_test = test_df['Next Week Rank']
    
    #Multiple Linear Regression
    lin_model = LinearRegression().fit(X_train, y_train)
    lin_model_output = lin_model.predict(X_test)
    
    #Build output
    merged_results = build_output(max_rank, next_teams, next_rank_act, byes_index, lin_model_output)
    
    #Get metrics 
    r2 = r2_score(merged_results['Adj. Predicted Ordinal Rank'], merged_results['Actual Rank'])
    total_error = get_error(merged_results['Adj. Predicted Ordinal Rank'], merged_results['Actual Rank'], mode = 'total')
    
    if get_table == False:
        return r2, total_error
    
    else:
        return r2, total_error, merged_results

def run_rf(rf_params, df, next_week, get_table = False):
    from sklearn.ensemble import RandomForestRegressor

    #Unpack params
    feature_drop = rf_params['feature_drop']
    max_rank = rf_params['max_rank']
    
    #Get next week teams list
    next_teams = df['Team'].loc[df['Week'] == next_week - 1]
    next_rank_act = df['Next Week Rank'].loc[df['Week'] == next_week - 1]

    #Get byes
    byes_index = []
    for index, row in df.iterrows():
        if row['Week'] == next_week - 1 and not(row['Opp. Rank'] >= 0):
            byes_index.append(index % 25)
    
    #Drop null rows (teams with byes)
    df.dropna(inplace = True)
    
    #Drop irrelevant cols
    df = df.drop(columns = feature_drop)
    
    #Convert cols to num
    for name, values in df.iteritems():
        df[name] = pd.to_numeric(values)
        
    #Set training data to Week < next week
    train_df = df.loc[df['Week'] < next_week - 1]
    X_train = train_df.drop(columns = ['Next Week Rank'])
    y_train = train_df['Next Week Rank']
    
    #Set testing data to Week = next week
    test_df = df.loc[df['Week'] == next_week - 1]
    X_test = test_df.drop(columns = ['Next Week Rank'])
    y_test = test_df['Next Week Rank']
    
    #RF Regression
    rf_model = RandomForestRegressor(random_state = 8)
    rf_model.fit(X_train, y_train)
    rf_model_output = rf_model.predict(X_test)
    
    #Build output
    merged_results = build_output(max_rank, next_teams, next_rank_act, byes_index, rf_model_output)
    
    #Get metrics 
    r2 = r2_score(merged_results['Adj. Predicted Ordinal Rank'], merged_results['Actual Rank'])
    total_error = get_error(merged_results['Adj. Predicted Ordinal Rank'], merged_results['Actual Rank'], mode = 'total')
    
    if get_table == False:
        return r2, total_error
    
    else:
        return r2, total_error, merged_results