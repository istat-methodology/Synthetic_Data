import os
import sys
import pdb
import platform

# Initializations 

colab_active = 'google.colab' in sys.modules
print("Colab Active: ", colab_active)
# Operating System
OS = platform.system()                           # returns 'Windows', 'Linux', etc

if not os.path.exists('./Output'):
   os.makedirs('./Output')

if not os.path.exists('./Output/Pictures'):
   os.makedirs('./Output/Pictures')

# All Globals

benchmark = False
gaussian_copula_synth_model = False 
ctgan_synth_model = False
copula_gan_synth_model = True
dataset = 'telephony'   # satgpa, acs, telephony
model_to_test = "telephony_copulagan_1000_epochs.pkl"
#model_to_test = "telephony_copulagan_3_epochs.pkl"
model_names = []
limit_to_generate = None
training = True
save_score = False
install_libraries = False
incremental_training = True
it_gap = 1000

# Libraries Installation Section

if install_libraries is True: 
    os.system('pip install --upgrade --no-cache-dir gdown')
    os.system('pip install sdgym==0.5.0')
    #os.system('pip install pandas')
    os.system('pip install matplotlib==3.1.3')
    os.system('pip install openpyxl')
    os.system('pip install pandas==1.3.4')
    os.system('pip install italian_holidays')

# All Imports

import timeit
import numpy as np
import pandas as pd
from sdv.demo import load_tabular_demo
from sdv.tabular import GaussianCopula, CTGAN, CopulaGAN
from sdv.evaluation import evaluate
from synth_utility_libs import explore_data, data_download, preprocess_telephony_data

# All Hyper-parameters

epochs = 50

# All Settings

start_global_time = timeit.default_timer()
pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500) 
if ctgan_synth_model == True and copula_gan_synth_model == True: # Only one Gan 
  ctgan_synth_model = False

# Mounting Google Drive via Code

if colab_active is True: 
  from google.colab import drive
  drive.mount('/content/drive')
  sys.path.append('/content/drive/My Drive')
  model_path = "/content/drive/MyDrive/DL_Models/"
  if os.path.isdir(model_path) is False: 
    os.makedirs(model_path) 
else: 
  model_path = "./"

# Data Download - ACS, SatGPA, Telephony

if benchmark == True: 
  data = load_tabular_demo('student_placements')
  n_to_generate = data.shape[0]
else: 
  if dataset is 'satgpa':
    out = data_download("./satgpa.csv", "1NNVF1LhBDkW_KKp5_QW8cAiQDFatzWMy", OS, False)
    data = pd.read_csv('./satgpa.csv')
    data = data.drop(['sat_sum'], axis=1)
    data.to_csv('satgpa_no_sum.csv', sep=',')
    n_to_generate = data.shape[0]
  elif dataset is 'acs':
    out = data_download("./acs_dataset.zip", "1mKZfDieGBJP-cS-R7_i3zVKVawXThfUc", OS)
    if limit_to_generate is not None: 
      data = pd.read_csv('./acs_dataset.csv', nrows = limit_to_generate)
      n_to_generate = limit_to_generate
    else: 
      data = pd.read_csv('./acs_dataset.csv')
      n_to_generate = data.shape[0]
  elif dataset is 'telephony':
    out = data_download("./syntetic_telephony.zip", "1knlC9DQ-iQhxwpmd9qzFWSS0OcgI9-LA", OS)
    xl_file = pd.ExcelFile('./syntetic_telephony.xlsx')
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    data = dfs['Sheet1']
    n_to_generate = data.shape[0]

# Data Preparation

if dataset is 'telephony':
    print("\n\nSample of Real Telephony Data: \n\n", data.head)
    
    data_new = preprocess_telephony_data(data)                                  # Data Preprocessing for Telephony

    print("\n\nSample of Real Preprocessed Telephony Data: \n\n", data_new.head)

    explore_data(data_new)

# Synthetic Data Generation via Gaussian Copula Method

if gaussian_copula_synth_model == True:
  model = GaussianCopula()

  if training is True:
    model.fit(data_new)
    model_names.append(model_path+dataset+'_gaussian_copula_'+str(epochs)+'_epochs_'+str(len(data_new))+'_obs.pkl')
    model.save(model_names[-1])

# Synthetic Data Generation via Conditional GAN

if ctgan_synth_model == True:
  model = CTGAN(
    epochs=epochs,
    batch_size=100,
    generator_dim=(256, 256, 256),
    discriminator_dim=(256, 256, 256)
  )

  if training is True:
    model.fit(data_new)
    model_names.append(model_path+dataset+'_ctgan_'+str(epochs)+'_epochs_'+str(len(data_new))+'_obs.pkl')
    model.save(model_names[-1])

# Synthetic Data Generation via Copula GAN

if incremental_training is True:
  if copula_gan_synth_model == True: 
    
    for i in range(1, int(len(data_new) / it_gap+1)):

      print("Training on incremental data chunk:",str(i)," - ", str(int(i*it_gap)), " data points...")

      data_temp = data_new[0:i*it_gap]

      model = CopulaGAN(
        epochs=epochs,
        batch_size=100,
        generator_dim=(256, 256, 256),
        discriminator_dim=(256, 256, 256)
      )

      model.fit(data_temp)
      model_names.append(model_path+dataset+'_copulagan_'+str(epochs)+'_epochs_'+str(int(i*it_gap))+'_data_points.pkl')
      model.save(model_names[-1])
else: 
    if copula_gan_synth_model == True: 
      model = CopulaGAN(
        epochs=epochs,
        batch_size=100,
        generator_dim=(256, 256, 256),
        discriminator_dim=(256, 256, 256)
      )
    
      if training is True:
        model.fit(data_new)
        model_names.append(model_path+dataset+'_copulagan_'+str(epochs)+'_epochs_'+str(len(data_new))+'_obs.pkl')
        model.save(model_names[-1])

# Model Loading
if training is False: 
  model_names.append(model_path+model_to_test)

print(model_names)

# Model Loading and Preparation

model_file = []
model_to_load = []
if gaussian_copula_synth_model == True:
  model_file.append(model_names[0])
  model_to_load.append(("GaussianCopula", GaussianCopula))
if ctgan_synth_model == True:
  model_file.append(model_names[-1])
  model_to_load.append(("CTGAN", CTGAN))
elif copula_gan_synth_model == True:
  model_file.append(model_names[-1])
  model_to_load.append(("COPULAGAN", CopulaGAN))

loaded_model = []
for mf,ml in zip(model_file, model_to_load): 
  loaded_model.append((ml[0], ml[1].load(mf)))

# Synthetic Data Generation

synthetic_data = []
for lm in loaded_model: 
  synthetic_data.append((lm[0], lm[1].sample(n_to_generate)))

# Synthetic Data Exploratory Analysis

scored_and_synth_data = []
for sd in synthetic_data:
  try:
    print("\nMethod: ",sd[0])
    explore_data(sd[1])
    if save_score is True: 
      score = evaluate(sd[1], data_new)
      pdb.set_trace()
      print("\n\nScore: ", score)
    else: 
      score = -1
    
    scored_and_synth_data.append((sd[0], sd[1], score))  
  except:
    print("Error")

total_time = timeit.default_timer() - start_global_time

print("Global Exectution Time: ", total_time)

for sas in scored_and_synth_data:
  sas[1].to_csv('./Output/'+dataset+'_synth_data_generated_by_method_'+sas[0].lower()+'total_time_'+str(epochs)+'_epochs_'+str(len(data_new))+'_obs_'+str(round(total_time,2))+'_score_'+str(round(sas[2],3))+'.csv', sep=',')

for sas in scored_and_synth_data:
  sas[1].to_excel('./Output/'+dataset+'_synth_data_generated_by_method_'+sas[0].lower()+'total_time_'+str(epochs)+'_epochs_'+str(len(data_new))+'_obs_'+str(round(total_time,2))+'_score_'+str(round(sas[2],3))+'.xlsx')
    