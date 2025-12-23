import pandas as pd
from datetime import date

# Read a CSV file into a DataFrame
# Make sure to clean up the csv removing all but the options data
df = pd.read_csv('2024-11-28-AccountStatement.csv')
print(df.head())

# Remove all rows where TYPE != TRD
#df = df[df["TYPE"].str.contains("TRD") == True]
#print(df)

# Now there are only Trade rows

## To fix ValueError: could not convert string to float: '-13,800.00'
df['AMOUNT'] = [float(str(i).replace(',', '')) for i in df['AMOUNT']]

# Remove unnecessary columns
df.drop('TIME', axis=1, inplace=True)
df.drop('REF #', axis=1, inplace=True)
df.drop('BALANCE', axis=1, inplace=True)
print(df)

# Calulate sum including fees
# Recast AMOUNT from string to float
total = (df['AMOUNT'].astype(float)).sum()
print(total)

# Calulate number of days
# Get first and last date
# Recast from string to date
first_date = pd.to_datetime(df['DATE'].iloc[0])
last_date = pd.to_datetime(df['DATE'].iloc[-1])
print(first_date)
print(last_date)

number_of_days = last_date - first_date
print(number_of_days)

# APY
# rate = 1 + (total/investment_amount)
# days = days_in_year/number_of_days
# apy = ((rate to the power of days) -1) * 100
investment_amount = 50000.00
rate = 1 + (total / investment_amount)
print(rate)

# Get days value for Timedelta number_of_days variable
days_in_year = 365
days = days_in_year / number_of_days.days
print(days)

apy = ((pow(rate, days)) - 1) * 100

print(apy)


# Plot the data
import matplotlib.pyplot as plt

# Get the average of the months
#df['Avg'] = df[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']].mean(axis=1)

df.index = pd.to_datetime(df['DATE'],format='%m/%d/%y')
#monthly = df.groupby(by=[df.index.month, df.index.year])

#monthly = df.groupby(by=[df.index.month, df['AMOUNT']])

#print(df.index.month)

#monthly.get_group('01')

df['AMOUNT_RECAST'] = (df['AMOUNT'].astype(float))

monthly = df.groupby(df.index.month)['AMOUNT_RECAST'].sum()

print(monthly)

# Create a Plot
#plt.plot(monthly)
plt.plot(monthly, color='red', marker='o')
#df.groupby(by=[df.index.month, df.index.year]).plot(kind="bar", title="DataFrameGroupBy Plot")
plt.title('Monthly Amounts', fontsize=14)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Amount', fontsize=14)

plt.grid(True)
plt.show(block=True)







