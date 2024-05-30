# Plot monthly amounts and calculate APY.

# Pandas
import pandas as pd

## Read a CSV file into a DataFrame
df = pd.read_csv('2024-04-27-AccountStatement.csv')

#df = pd.read_csv('file1.csv')

# Wrangling

## Remove all rows where TYPE != TRD, seems to drop the first amount due to type = BAL?
## Had to make the first row use TRD instead of BAL
df = df[df["TYPE"].str.contains("TRD") == True]

print(df.head())

## Need to get AMOUNT by MONTH
## Create a column for recasted DATE
df['DATE_RECAST'] = pd.to_datetime(df['DATE'],format='%m/%d/%y')

## https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-separately-from-pandas-datetime-column
## Define list of attributes required    
al = ['year', 'month', 'day']

## Define generator expression of series, one for each attribute
date_gen = (getattr(df['DATE_RECAST'].dt, i).rename(i) for i in al)

## Concatenate results and join to original dataframe
df = df.join(pd.concat(date_gen, axis=1))

## Create a column for recasted AMOUNT
## To fix ValueError: could not convert string to float: '-13,800.00'
df['AMOUNT'] = [float(str(i).replace(',', '')) for i in df['AMOUNT']]

df['AMOUNT_RECAST'] = df['AMOUNT'].astype(float)

## Group By MONTH and AMOUNT

df['monthly'] = df.groupby(df['month'])['AMOUNT_RECAST'].sum()
#df['monthly'] = df.groupby(df['DATE_RECAST'])['AMOUNT_RECAST'].sum()

print(df.head())
#print(df['month'])
