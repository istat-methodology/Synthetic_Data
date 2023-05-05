import os
import sys
import pdb
import numpy as np
import pandas as pd
from italian_holidays import italian_holidays

# All Functions Definitions

def explore_data(data): 
  print("\nHead of Data: \n", data.head())
  print("\nTail of Data: \n", data.tail())
  print("\nShape of Data: ", data.shape)
  print("\nInformation about Data: \n")
  try: 
    data.info()
  except: 
    pass
  print("\nTypes of Data attributes: \n")
  try: 
    data.dtypes
  except: 
    pass
  print("\nSummary of all numerical fields in the dataset: \n")
  try: 
    data.describe(include = [np.number])
  except: 
    pass
  print("\nSummary of all categorical fields in the dataset: \n")
  try: 
    data.describe(include = ['O'])
  except: 
    pass
  print("\nLoop Through Each Column and Check for nulls: \n")
  try: 
    for i in range(len(data.columns)):
        print(data.columns[i] + ": " + str(data[data.columns[i]].isna().sum()))
  except: 
    pass

def data_download(file_to_download, gdrive_code, OS, uncompress = True):
  if not os.path.exists(file_to_download):
    os.system('gdown --id "'+gdrive_code+'" --output '+file_to_download)
    if OS == "Linux" and uncompress:
        os.system('unzip -o -n "./'+file_to_download+'" -d "./"')
    if OS == "Windows" and uncompress: 
        os.system('tar -xf "./'+file_to_download)
    return True
  else: 
    return None

def preprocess_telephony_data(data, verbose):

    data_new = data.copy()
      
    # Categorical Variables
    data_new.COD_CELLA_CHIAMATA=data_new.COD_CELLA_CHIAMATA.astype("int64").astype("str")
    data_new.CHIAVE_NUM_CHIAMANTE=data_new.CHIAVE_NUM_CHIAMANTE.astype("int64").astype("str")
      
    # Datetime Continuous Variables 
    data_new.DATA_CHIAMATA = data_new.DATA_CHIAMATA.astype("int64").astype("str")
    data_new.ORA_MIN_CHIAMATA = data_new.ORA_MIN_CHIAMATA.astype("int64").astype("str").str.pad(width=6, side='left', fillchar='0')
      
    data_new["DATA_ORA"] = data_new[["DATA_CHIAMATA", "ORA_MIN_CHIAMATA"]].apply("".join, axis=1)
    data_new.drop(["DATA_CHIAMATA", "ORA_MIN_CHIAMATA"], axis = 1, inplace = True)                          # Drop old columns
      
    data_new.DATA_ORA=pd.to_datetime(data_new.DATA_ORA, format='%Y%m%d%H%M%S')                              # Reduce the dataset to 3 columns 

    # Aggiunta delle Variabili Festivo e Week End che aggiungano informazione ulteriore alle colonne 
    
    holidays = italian_holidays() 
    
    holidays.is_holiday(data_new.iloc[0,2])
    
    data_new["FESTIVO"] = data_new["DATA_ORA"].apply(lambda x: holidays.is_holiday(x))        # check for holidays dates
    data_new["FESTIVO"] = data_new["DATA_ORA"].apply(lambda x: x.weekday() >= 5)              # check for week end dates

    data_new.FESTIVO = data_new.FESTIVO.astype("int64")

    if verbose is True:
        print("\n\nHolidays and Week-ends counting: \n\n", data_new[data_new["FESTIVO"] == True].count())

    # Categorical Data Adjustment

    data_new["COD_CELLA_CHIAMATA"]="C"+data_new.COD_CELLA_CHIAMATA
    data_new["CHIAVE_NUM_CHIAMANTE"]="C"+data_new.CHIAVE_NUM_CHIAMANTE
    
    data_new.rename(columns = {'COD_CELLA_CHIAMATA':'CELL_CALL_CODE', 
                          'CHIAVE_NUM_CHIAMANTE':'NUM_CALLER_KEY',
                          'DATA_ORA':'TIME_CALL',
                          'FESTIVO':'FESTIVE'}, inplace = True)    
    
    return data_new

    