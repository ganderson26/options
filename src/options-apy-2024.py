# Plot Monthly Premiums
# Calculate Monthly APY
# Calculate Total APY

# Pandas

import pandas as pd
import matplotlib.pyplot as plt

## Read a CSV file into a DataFrame
# Contains various TYPES such as BAL, DOI and TRD
df = pd.read_csv('2025-06-16-AccountStatement.csv')

## Viewing Data
#print(df.head())
#print(df.tail())
#print(df.info())
#print(df.describe())


# Wrangling

## Remove all rows where TYPE != TRD
# Very odd issue dropping the first amount due to type = BAL? I changed the TYPE to TRD on the row.
# This emphasizes the need to understand the data
df = df[df["TYPE"].str.contains("BAL") == False]
df = df[df["TYPE"].str.contains("EFN") == False]
df = df[df["TYPE"].str.contains("RAD") == False]
df = df[df["TYPE"].str.contains("CRC") == False]

# To see just credit/debit on assignment + DOI interest, remove TRD
#df = df[df["TYPE"].str.contains("TRD") == False]

# To see just premiums, remove DOI and EXP
df = df[df["TYPE"].str.contains("DOI") == False]
df = df[df["TYPE"].str.contains("EXP") == False]

#print(df.head())

# Changed 2 TRD to XXX due to MRNA outlier


## Now there are only Trade rows

## To fix ValueError: could not convert string to float: '-13,800.00'
df['AMOUNT'] = [float(str(i).replace(',', '')) for i in df['AMOUNT']]

## Remove unnecessary columns
df.drop('TIME', axis=1, inplace=True)
df.drop('REF #', axis=1, inplace=True)
df.drop('BALANCE', axis=1, inplace=True)
#print(df)

## Need to get AMOUNT by MONTH
## Create a column for recasted DATE
df['DATE_RECAST'] = pd.to_datetime(df['DATE'],format='%m/%d/%y')
#print(df['DATE_RECAST'])

## https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-separately-from-pandas-datetime-column
## Define list of attributes required    
al = ['year', 'month', 'day']

## Define generator expression of series, one for each attribute
date_gen = (getattr(df['DATE_RECAST'].dt, i).rename(i) for i in al)

## Concatenate results and join to original dataframe
df = df.join(pd.concat(date_gen, axis=1))
#print(df)

## Create a column for recasted AMOUNT
## To fix ValueError: could not convert string to float: '-13,800.00'
df['AMOUNT'] = [float(str(i).replace(',', '')) for i in df['AMOUNT']]

df['AMOUNT_RECAST'] = df['AMOUNT'].astype(float)
print(df['AMOUNT_RECAST'])

print(df['month'], df['year'])


## Group By MONTH and AMOUNT
#df['monthly'] = df.groupby(df['month'])['AMOUNT_RECAST'].sum()
#print(df['monthly'])

df_new = df.groupby(df['month'])['AMOUNT_RECAST'].sum().reset_index()
#df_new = df.groupby(df['month'])['AMOUNT_RECAST'].sum()


df_new.to_csv('latest.csv')




# Calculate APY
# Keep it simple and just use 31 days per month
number_of_days = 31

# APY
# rate = 1 + (total/investment_amount)
# days = days_in_year/number_of_days
# apy = ((rate to the power of days) -1) * 100
investment_amount = 80000.00
rate = 1 + (df_new['AMOUNT_RECAST'] / investment_amount)

# Get days value for Timedelta number_of_days variable
days_in_year = 365
days = days_in_year / number_of_days

apy = ((pow(rate, days)) - 1) * 100
df_new['monthly_apy'] = apy



##df.dropna(subset=['monthly_apy'], inplace=True)

for index, row in df_new.iterrows():
    print('Month ', row['month'], 'Premiums =', row['AMOUNT_RECAST'], 'APY =', row['monthly_apy'])

# Calculate Total APY
# Keep it simple and just use number of months/rows
total_apy = df_new['monthly_apy'].sum() / df_new.shape[0]
print('Total APY =', total_apy)

# Visualization

## Plot the data

## Create a Plot
#plt.plot(df_new['monthly'], color='red', marker='o')
#plt.title('Monthly Amounts', fontsize=14)
#plt.xlabel('Month', fontsize=14)
#plt.ylabel('Amount', fontsize=14)

#plt.grid(True)

# Setting the number of ticks and show Month
# Only show monthly ticks not the half points ie. 1, 2, 3, 4 not 1, 1.5, 2, 2.5...
# https://www.geeksforgeeks.org/how-to-change-the-number-of-ticks-in-matplotlib/
# https://stackoverflow.com/questions/2497449/plot-string-values-in-matplotlib
#plt.xticks([1, 2, 3, 4, 5, 6], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])

#plt.show(block=True)
