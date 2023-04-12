import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

df = pd.read_csv('Data_opg_2.csv')

print(df)


df.drop(columns=["Consumer prices",'Administration margins','Housing taxes, Copenhagen','Administration margins','Rentefradrag','Disposable income, Denmark','Disposable income, Copenhagen','Housing stock, Denmark / 1000','Housing stock CPH / 1000','Flat price, Copenhagen 2006=100','Unnamed: 22','House price, Denmark 2006=100'], inplace=True)
# Here we drop the unnessaary coloumns for our data analysis.
df = df.rename(columns = {"Unnamed: 0":"year","Consumer Price Index: 2000Q4 = 100": "PRICE_INDEX","Inflation Yearly" : "inflation","Real Disposible Income 2000Q4 = 100":"Income_DK", "Real Disposible Income Copenhagen 2000Q4 = 100": "Income_CPH","Housing stock, Denmark":"Housing_DK","Housing stock, Copenhagen":"Housing_CPH","Real House Price Denmark 2000Q4 = 100":"House Price_DK","Real Flat Price Copenhagen 2000Q4 = 100":"Flat Price_CPH"})
# Here we rename the reaminiing columns to clariy their meaning.
print(df)
#Here we print our remaning data to verify we have dropped and renamed everything.
from scipy.stats import linregress

# Here we create scatter plot of prices
plt.scatter(df.index, df['House Price_DK'])

# Here we calculate trendline using linear regression
slope, intercept, r_value, p_value, std_err = linregress(df.index, df['House Price_DK'])
trendline = slope * df.index + intercept

# Here we add trendline to plot
plt.plot(df.index, trendline, color='red')

# Here we add labels and title
plt.xlabel('Quarters since Q1 1996')
plt.ylabel('Price Index')
plt.title('Price trendline')

# Here we show our plot
plt.show()





# Here we calculate the ratio of Price_CPH to Income_CPH
df['Price_Income_Ratio'] = df['House Price_DK'] / df['Income_DK']

# Here we calculate the average of Price_Income_Ratio column
avg_price_income_ratio = np.mean(df['Price_Income_Ratio'])

# Here we plot the ratio as a graph with a trendline
plt.plot(df['Price_Income_Ratio'])
plt.plot([0, len(df)], [avg_price_income_ratio, avg_price_income_ratio], 'r--')
plt.xlabel('Quarters since Q1 1996')
plt.ylabel('Price to Income Ratio')
plt.title('Price to Income Ratio Graph with Trendline')
plt.legend(['Price to Income Ratio', 'Average Price-Income Ratio'], loc='best')
plt.show()


ff = pd.read_csv('Pricerent.csv')

#Here we open another dataset containing house price and rent levels for Denmark since 1996.

ff['Real house price to real rent'] = ff['index real houseprices'] / ff['indeks real husleje']
#Here we define our ratio Real house prices to real rent levels.

avg_real_house_to_rent_ratio = np.mean(ff['Real house price to real rent'])
# Here we define an average of the ratio.

plt.plot(ff['Real house price to real rent'])
plt.plot([0, len(ff)], [avg_real_house_to_rent_ratio, avg_real_house_to_rent_ratio], 'r--')
plt.xlabel('Quarters since Q1 1996')
plt.ylabel('Price to Income Ratio')
plt.title('Price to Income Ratio Graph with Trendline')
plt.legend(['Price to Income Ratio', 'Average Price-Income Ratio'], loc='best')
plt.show()


