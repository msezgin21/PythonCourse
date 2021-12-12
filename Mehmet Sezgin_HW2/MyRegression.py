import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib import pyplot as plt

import os

os.chdir('C:\pyt')



data = pd.read_stata('Fearon_Replication.dta')
data.head()
data.describe()
data.columns
data = data[['gdpen', 'polity2','pop']].dropna()
y = data['polity2']
x = pd.DataFrame(data['gdpen'])
X = pd.DataFrame(data[['gdpen', 'pop' ]])

x['constant'] = 1
regres1 = sm.OLS(y, x,).fit()

print(regres1.summary().as_latex())

fig = sns.regplot(x="gdpen", y="polity2", data=data)
sns.set_style("whitegrid")
plt.title("Regression Results")
plt.xlabel('GDP Level')
plt.ylabel('Democracy Score')
plt.show()



fig = sns.regplot(x="pop", y="polity2", data=data)
sns.set_style("whitegrid")
plt.title("Regression Results")
plt.xlabel('Population')
plt.ylabel('Democracy Score')
plt.show()

    
   
 

