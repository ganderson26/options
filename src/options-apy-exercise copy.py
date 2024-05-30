# Plot Monthly Premiums
# Calculate Monthly APY
# Calculate Total APY

# Pandas

import pandas as pd
from datetime import date

## Read a CSV file into a DataFrame
# Contains various TYPES such as BAL, DOI and TRD
df = pd.read_csv('2024-04-27-AccountStatement.csv')

## Viewing Data
#print(df.head())
#print(df.tail())
#print(df.info())
#print(df.describe())


# Wrangling

## Remove all rows where TYPE != TRD
# Very odd issue dropping the first amount due to type = BAL? I changed the TYPE to TRD on the row.
# This emphasizes the need to understand the data
df = df[df["TYPE"].str.contains("TRD") == True]
#print(df.head())

## Now there are only Trade rows

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
#print(df['AMOUNT_RECAST'])

## Group By MONTH and AMOUNT
df['monthly'] = df.groupby(df['month'])['AMOUNT_RECAST'].sum()
#print(df['monthly'])

## Calculate Monthly APY
# Keep it simple and just use 31 days per month
mnumber_of_days = 31

# APY
# rate = 1 + (total/investment_amount)
# days = days_in_year/number_of_days
# apy = ((rate to the power of days) -1) * 100
minvestment_amount = 50000.00
mrate = 1 + (df['monthly'] / minvestment_amount)

# Get days value for Timedelta number_of_days variable
mdays_in_year = 365
mdays = mdays_in_year / mnumber_of_days

mtotal_apy = ((pow(mrate, mdays)) - 1) * 100
df['monthly_apy'] = mtotal_apy
print(df.head())

# Visualization

## Plot the data
import matplotlib.pyplot as plt


## Create a Plot
plt.plot(df['monthly'], color='red', marker='o')
plt.title('Monthly Amounts', fontsize=14)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Amount', fontsize=14)

plt.grid(True)

# Setting the number of ticks and show Month
# Only show monthly ticks not the half points ie. 1, 2, 3, 4 not 1, 1.5, 2, 2.5...
# https://www.geeksforgeeks.org/how-to-change-the-number-of-ticks-in-matplotlib/
# https://stackoverflow.com/questions/2497449/plot-string-values-in-matplotlib
plt.xticks([1, 2, 3, 4, 5], ['Jan', 'Feb', 'Mar', 'Apr', 'May'])

plt.show(block=True)


# Calulate APY for the total period
# Recast AMOUNT from string to float
total_amount = (df['AMOUNT'].astype(float)).sum()
#print(total_amount)

# Calulate number of days
# Get first and last date
# Recast from string to date
first_date = pd.to_datetime(df['DATE'].iloc[0])
last_date = pd.to_datetime(df['DATE'].iloc[-1])
#print(first_date)
#print(last_date)

number_of_days = last_date - first_date
#print(number_of_days)

# APY
# rate = 1 + (total/investment_amount)
# days = days_in_year/number_of_days
# apy = ((rate to the power of days) -1) * 100
investment_amount = 50000.00
rate = 1 + (total_amount / investment_amount)
#print(rate)

# Get days value for Timedelta number_of_days variable
days_in_year = 365
days = days_in_year / number_of_days.days
#print(days)

total_apy = ((pow(rate, days)) - 1) * 100

print(total_apy)


