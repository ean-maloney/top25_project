def ordinalize(raw_vals, order = 'asc'):
    '''Take an input iterable and return a data frame with two cols.
    Col 1 = original values, Col 2 = ordinal rank of value.'''
    import pandas as pd

    orig_vals = raw_vals.copy()
    
    raw_vals.sort()

    temp_df = pd.DataFrame()
    
    if order == 'asc':
        temp_df['rank'] = range(1, len(raw_vals) + 1)
        temp_df['values'] = raw_vals

        for i in range(1, len(raw_vals)):
            if raw_vals[i] == raw_vals[i - 1]:
                temp_df.drop(index = i, inplace = True)

    elif order == 'desc':
        temp_df['rank'] = range(len(raw_vals), 0, -1)
        temp_df['values'] = raw_vals

        for i in range(len(raw_vals) - 1, 0, -1):
            if raw_vals[i] == raw_vals[i - 1]:
                temp_df.drop(index = i - 1, inplace = True)

    else: 
        raise RuntimeError("Order must be 'asc' or 'desc'.")

    df = pd.DataFrame()
    df['values'] = orig_vals
    merged = df.merge(temp_df, how = 'left', left_on = 'values', right_on = 'values')

    return merged

def get_error(pred, actual, mode = 'total'):
    '''Find absolute error.'''

    error = 0

    for i in range(0, len(pred)):
        error += abs(pred[i] - actual[i])
    
    if mode == 'avg':
        error /= len(pred)

    return error

class Ord():
    def __init__(self, input):
        self.input = input.copy()
        self.parallel = ordinalize(input.copy())
        self.dparallel = ordinalize(input.copy(), order = 'desc')
        self.ranks = list(self.parallel['rank'])

    def order(self, order = 'asc'):
        if order == 'asc':
            return sorted(self.input)
        
        elif order == 'desc':
            return sorted(self.input, reverse = True)
        
        else:
            raise RuntimeError('Argument "order" must be "asc" or "desc".')

    
    def get_error(self, actual, mode = 'total'):
        '''Find absolute error.'''

        error = 0
        pred = list(self.parallel['rank'])

        for i in range(0, len(pred)):
            error += abs(pred[i] - actual[i])
        
        if mode == 'avg':
            error /= len(pred)

        return error

