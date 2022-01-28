# -*- coding: utf-8 -*-
"""gan_sdgym_istat_synt_challenge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X4K8zXZD85Q0lxENOWPzrGG7CHFyyrem
"""

import os
import platform
# Operating System
OS = platform.system()                                                             # returns 'Windows', 'Linux', etc

"""# Libraries Installation Section

Installation of all required libraries: SDGym
"""

os.system('pip install gdown')
os.system('pip install sdgym')
os.system('pip install pandas')

"""# All Imports"""

import timeit
import numpy as np
import pandas as pd
from sdv.demo import load_tabular_demo
from sdv.tabular import GaussianCopula, CTGAN, CopulaGAN
from sdv.evaluation import evaluate

"""# All Globals"""

benchmark = False
#benchmark = True
gaussian_copula_synth_model = False
ctgan_synth_model = False
copula_gan_synth_model = True
#dataset = 'satgpa'
dataset = 'acs'
model_names = []

"""# All Settings"""

start_global_time = timeit.default_timer()
pd.set_option('display.max_columns', 500) 
pd.set_option('display.max_rows', 500) 
if ctgan_synth_model == True and copula_gan_synth_model == True: # Only one Gan 
  ctgan_synth_model = False

"""# All Functions Definitions"""

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

"""# Data Download - ACS and SatGPA"""

if benchmark == True: 
  data = load_tabular_demo('student_placements')
  n_to_generate = data.shape[0]
else: 
  if dataset is 'satgpa':
    if not os.path.exists("./satgpa.csv"):
      os.system('gdown --id "1NNVF1LhBDkW_KKp5_QW8cAiQDFatzWMy" --output "./satgpa.csv"')
      data = pd.read_csv('./satgpa.csv')
      data = data.drop(['sat_sum'], axis=1)
      data.to_csv('satgpa_no_sum.csv', sep=',')
      n_to_generate = data.shape[0]
  elif dataset is 'acs':
    if not os.path.exists("./acs_dataset.zip"):
      os.system('gdown --id "1mKZfDieGBJP-cS-R7_i3zVKVawXThfUc" --output "./acs_dataset.zip"')
      if OS == "Linux":
          os.system('unzip -o -n "./acs_dataset.zip" -d "./"')      
      #data = pd.read_csv('./acs_dataset.csv')
      #n_to_generate = data.shape[0]

      data = pd.read_csv('./acs_dataset.csv', nrows = 1000)
      n_to_generate = 1000

"""# Exploratory Analysis"""

explore_data(data)

"""# Synthetic Data Generation via Gaussian Copula Method

In mathematical terms, a copula is a distribution over the unit cube [0,1]d which is constructed from a multivariate normal distribution over Rd by using the probability integral transform. Intuitively, a copula is a mathematical function that allows us to describe the joint distribution of multiple random variables by analyzing the dependencies between their marginal distributions.
"""

if gaussian_copula_synth_model == True:
  model = GaussianCopula()
  model.fit(data)
  model_names.append(dataset+'_gaussian_copula.pkl')
  model.save(model_names[-1])

"""# Synthetic Data Generation via Conditional GAN

Modeling the probability distribution of rows in tabular data and generating realistic synthetic data is a non-trivial task. Tabular data usually contains a mix of discrete and continuous columns. Continuous columns may have multiple modes whereas discrete columns are sometimes imbalanced making the modeling difficult. Existing statistical and deep neural network models fail to properly model this type of data. We design TGAN, which uses a conditional generative adversarial network to address these challenges. To aid in a fair and thorough comparison, we design a benchmark with 7 simulated and 8 real datasets and several Bayesian network baselines. TGAN outperforms Bayesian methods on most of the real datasets whereas other deep learning methods could not.
"""

if ctgan_synth_model == True:
  model = CTGAN(
    epochs=500,
    batch_size=100,
    generator_dim=(256, 256, 256),
    discriminator_dim=(256, 256, 256)
  )
  model.fit(data)
  model_names.append(dataset+'_ctgan.pkl')
  model.save(model_names[-1])

"""# Synthetic Data Generation via Copula GAN

The CopulaGAN model is a variation of the CTGAN Model which takes advantage of the CDF based transformation that the GaussianCopulas apply to make the underlying CTGAN model task of learning the data easier.
"""

if copula_gan_synth_model == True:
  model = CopulaGAN(
    epochs=500,
    batch_size=100,
    generator_dim=(256, 256, 256),
    discriminator_dim=(256, 256, 256)
  )
  model.fit(data)
  model_names.append(dataset+'_copulagan.pkl')
  model.save(model_names[-1])

"""# Model Loading and Preparation"""

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

"""# Synthetic Data Generation"""

synthetic_data = []
for lm in loaded_model: 
  synthetic_data.append((lm[0], lm[1].sample(n_to_generate)))

"""# Synthetic Data Exploratory Analysis"""

scored_and_synth_data = []
for sd in synthetic_data:
  try:
    print("\nMethod: ",sd[0])
    explore_data(sd[1])
    score = evaluate(sd[1], data)
    print("\n\nScore: ", score)
    scored_and_synth_data.append((sd[0], sd[1], score))  
  except:
    print("Error")

total_time = timeit.default_timer() - start_global_time

for sas in scored_and_synth_data:
  sas[1].to_csv(dataset+'_synth_data_generated_by_method_'+sas[0].lower()+'total_time_'+str(round(total_time,2))+'_score_'+str(round(sas[2],3))+'.csv', sep=',')

print("Global Exectution Time: ", total_time)