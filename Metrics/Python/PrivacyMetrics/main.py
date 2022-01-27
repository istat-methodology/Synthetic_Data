import argparse
import pandas as pd
#import sdnist
import privacy_metric
import matplotlib.pyplot as plt
import numpy as np
#from sdv.evaluation import evaluate
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser.add_argument("--dataset", type=argparse.FileType("r"), help="this is the synthetic dataset (.csv file)")
    parser.add_argument("--groundtruth", type=argparse.FileType("r"), help="this is the original dataset (.csv file)")
    parser.add_argument("--privacy-metric", dest="privacy_metric", type=bool, help="compute the privacy metric")
    parser.add_argument("-x", "--exclude-columns", dest="x", help="list of columns to exclude") 
    parser.add_argument("-q", "--quasi", dest="q", help="list of quasi columns ")
   
    args = parser.parse_args()
    
    # Load datasets 
    dataset = pd.read_csv(args.dataset)  
    #groundtruth = pd.read_csv(args.groundtruth, dtypye=dtypes)
    groundtruth = pd.read_csv(args.groundtruth)
    print(args.dataset)
    pdb.set_trace()
    
    
    q = args.q
    Qs = q.split(",")
    Qs = [s.strip(" ") for s in Qs]
    x = args.x
    Xs = x.split(",")
    Xs = [s.strip(" ") for s in Xs]
    (percents, len_df), uniques1, uniques2, matched  = privacy_metric.cellchange(groundtruth, dataset, Qs, Xs)
        
    print("Matched: \n",matched)
    print("Percents: \n",percents)
    print("Number of Apparent Matches: ",len_df)
    #score = evaluate(dataset, groundtruth)
    #print("SDGym Score: ", score)
    
    #Histogram
    plt.figure(figsize = (10,10))
    plt.title('Percentage of raw synthetic records that had an apparent match with groundtruth dataset')
    percents.hist()
    plt.xlim(0,100)
    plt.savefig('privacy_metric.png')
    plt.show()