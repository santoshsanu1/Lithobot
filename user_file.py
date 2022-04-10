from distutils.log import error
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

comb_tab = {
        'comb1' :[   'CALI', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'SP', 'BS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'RXO', ],
        'comb2' : [   'CALI', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 
                'RMIC', 'RXO' ],
        'comb3' : [   'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'DTS', 'DCAL', 
                'DRHO', 'MUDWEIGHT', 'RMIC', 'RXO'  ],
        'comb4' : [   'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS', 
                'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO' ]
    }

def get_comb(fpath='test.csv', sep=',',ct=comb_tab):
    '''
    It takes file name as input and finds best matching combination for given log headers
    in the data. It returns available log headers and selected combination

    Input: File Path, separater in csv file, combination table
    '''
    ############## Their header analysis  #####################
    try:
        clm = pd.read_csv(fpath, index_col=0, nrows=0,sep=sep).columns.tolist()
        if len(clm) == 0:
            sys.exit(1)
        else:
            corr = { 
                'comb1' : len(set(clm).intersection(ct['comb1'])) / len(set(ct['comb1'])),
                'comb2' : len(set(clm).intersection(ct['comb2'])) / len(set(ct['comb2'])),
                'comb3' : len(set(clm).intersection(ct['comb3'])) / len(set(ct['comb3'])),
                'comb4' : len(set(clm).intersection(ct['comb4'])) / len(set(ct['comb4']))
            }
            # Find most suitable column based on %ge header correlation
            comb = max(zip(corr.values(), corr.keys()))[1]  
            ## More improvement can be done by %Null value comparision in data for each combination

            available_clm = []
            for col in ct[comb]:
                if col in clm:
                    available_clm.append(col)
            return available_clm,comb 
    except:
        print('Data could not be read. Check filename or path')
        sys.exit(1)
    

def read_data(comb,available_clm,fname='test.csv',sep=',',ct=comb_tab):
    try:
        raw_data = pd.read_csv(fname, sep=sep)
        temp_clm =  [x for x in ct[comb] if x not in available_clm]  # Missing headers in the data for selected combination
        temp_df = pd.DataFrame(columns=temp_clm, index=raw_data.index[:10]) # dataframe for missing logs initialized with Nan
        modified_raw_data = pd.concat([raw_data[available_clm], temp_df], axis=1) # Final data is generated with stacking of available data and missing data
        return modified_raw_data[ct[comb]]
    except:
        print('Data could not be read. Check filename, path and headers')

def preprocessing(data):
    ############################ Thresholding  ############################
    data['CALI'] = np.array(np.where((data['CALI'] <= 0) | (data['CALI'] >=30), np.nan, data['CALI'])) 
    data['RSHA'] = np.array(np.where((data['RSHA'] <= 0) | (data['RSHA'] >=100), np.nan, data['RSHA']))
    data['RMED'] = np.array(np.where((data['RMED'] <= 0) | (data['RMED'] >=100), np.nan, data['RMED']))
    data['RDEP'] =np.array(np.where((data['RDEP'] <= 0) | (data['RDEP'] >= 100 ), np.nan, data['RDEP']  ))
    data['RHOB'] = np.array(np.where((data['RHOB'] <= 1) | (data['RHOB'] >= 3), np.nan, data['RHOB'])  )
    data['GR'] = np.array(np.where((data['GR'] <= 0) | (data['GR'] >=200), np.nan, data['GR']))
    data['SGR'] = np.array(np.where((data['SGR'] <= 0) | (data['SGR'] >=200), np.nan, data['SGR']))
    data['NPHI'] = np.array(np.where((data['NPHI'] <= 0) | (data['NPHI'] >= 0.6), np.nan, data['NPHI'])  )
    data['PEF'] =np.array(np.where((data['PEF'] <=0 ) | (data['PEF'] >=10 ), np.nan, data['PEF']  ))
    data['DTC'] =np.array(np.where((data['DTC'] <= 0) | (data['DTC'] >=300 ), np.nan, data['DTC']  ))
    data['SP'] =np.array(np.where((data['SP'] <= -200) | (data['SP'] >= 200), np.nan, data['SP']  ))
    data['BS'] =np.array(np.where((data['BS'] <= 0) | (data['BS'] >=30 ), np.nan, data['BS']  ))
    data['ROP'] =np.array(np.where((data['ROP'] <= 0) | (data['ROP'] >=60 ), np.nan, data['ROP']  ))
    data['DTS'] =np.array(np.where((data['DTS'] <= 0) | (data['DTS'] >=300 ), np.nan, data['DTS']  ))
    data['DCAL'] = np.array(np.where((data['DCAL'] <= 0) | (data['DCAL'] >=2), np.nan, data['DCAL']))
    data['DRHO'] =np.array(np.where((data['DRHO'] <= -0.1) | (data['DRHO'] >=0.1 ), np.nan, data['DRHO']  ))
    data['MUDWEIGHT'] =np.array(np.where((data['MUDWEIGHT'] <= 0) | (data['MUDWEIGHT'] >=1.5 ), np.nan, data['MUDWEIGHT']  ))
    data['RMIC'] =np.array(np.where((data['RMIC'] <= 0) | (data['RMIC'] >=30 ), np.nan, data['RMIC']  ))
    data['ROPA'] =np.array(np.where((data['ROPA'] <= 0) | (data['ROPA'] >=60 ), np.nan, data['ROPA']  ))
    data['RXO'] =np.array(np.where((data['RXO'] <= 0) | (data['RXO'] >=20 ), np.nan, data['RXO']  ))

    ####################### Removing NaN values  #############################
    def replace_nan(x):
        if pd.isnull(x):
            return 9999
        else:
            return x
    for header in data.columns:
        data[header] = data[header].apply(replace_nan) 
    return data

def load_model(comb):
    try:
        file = open('Models/Model_'+comb[-1],'rb')
        clf = pickle.load(file)
        file.close()
        return clf
    except:
        print('Model could not be loaded')

def feedback(clf,comb_tab,comb,available_clm):
    importances = clf.feature_importances_
    # keys = 
    # values = 
    # imp_dict = {}
    # for i,col in enumerate(keys):
    #     imp_dict[col] = values[i]
    # print(imp_dict.keys())
    temp_clm =  [x for x in comb_tab[comb] if x not in available_clm]  # Missing headers in the data for selected combination
    df = pd.DataFrame()
    df['Missing_log'] = comb_tab[comb]
    df['Expected Accuracy Improvement'] = importances
    df = df.loc[df['Missing_log'].isin(temp_clm )]
    return df.reset_index(drop=True)



available_clm, comb = get_comb()
data = read_data(comb,available_clm,ct=comb_tab)
data = preprocessing(data)
clf = load_model(comb)
predictions = clf.predict(data)
data['Predicted_lithofacies'] = np.array(predictions)
data.to_csv('output.csv')
df = feedback(clf,comb_tab,comb,available_clm)
print('Below table represents the missing columns alog with improvement accuracy')
print(df)