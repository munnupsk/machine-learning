#required stuff for the apriori model in apriori dataset folder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("Market_Basket_Optimisation.csv",header=None)
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
#training apriori

from apyori import apriori
rules =apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#visuaizing the results
results=list(rules)

results_list = []

for i in range(0, len(results)):

    results_list.append('\RESULTS:\t' + str(results[i][2]))
