# Plot Monthly Premiums
# Calculate Monthly APY
# Calculate Total APY
#
# Need to have a way to use different investments for time periods.

import pandas as pd
import matplotlib.pyplot as plt

# Calculate the Annual Percentage Yield (APY)
def calculate_apy(investment, earnings, number_of_days):
    """
    Calculate the Annual Percentage Yield (APY).

    :param investment: Initial principal amount
    :param earnings: Premiums or DOI or P/L from assignments
    :param number_of_days: Number of compounding periods per year
    :return: APY as a percentage
    """

    rate = 1 + (earnings / investment)
    days_in_year = 365
    days = days_in_year / number_of_days
    apy = ((pow(rate, days)) - 1) * 100

    return apy

# Pandas 
#     
## Read a CSV file into a DataFrame
# Contains various TYPES such as BAL, DOI and TRD
# For Inception and all TRD, EXP and DOI, need to add back any equity holdings

# Average investment does not work
df = pd.read_csv('2025-11-12-AccountStatement.csv')
investment_amount = 68000.00

#df = pd.read_csv('1-2025-01-01-to-2025-01-25-AccountStatement.csv')
#investment_amount = 52000.00

#df = pd.read_csv('3-2025-01-28-to-2025-02-04-72580-AccountStatement.csv')
#investment_amount = 86000.00

#df = pd.read_csv('4-2025-02-05-to-2025-06-25-AccountStatement.csv')
#investment_amount = 81000.00

#df = pd.read_csv('5-2025-06-26-to-2025-09-18-AccountStatement.csv')
#investment_amount = 56000.00

#df = pd.read_csv('6-2025-09-19-to-2025-11-05-AccountStatement.csv')
#investment_amount = 56000.00

#df = pd.read_csv('7-2025-11-4-to-2025-11-12-AccountStatement.csv')
#investment_amount = 81000.00

## Viewing Data
#print(df.head())
#print(df.tail())
#print(df.info())
#print(df.describe())


# Wrangling

## Changed 2 TRD to EXP due to MRNA outlier. These are Assignment transactions.
#5/20/24,13:11:11,EXP,1000538387766,BOT +1 MRNA 100 21 JUN 24 75 CALL @65.55 CBOE,-0.01,-0.65,"-6,555.00","41,356.68"
#5/20/24,13:12:02,EXP,1000538387808,SOLD -100 MRNA @140.15,-0.13,,"14,015.00","55,371.55"

## TRD Only
df = df[df["TYPE"].str.contains("TRD") == True]

## DOI Only
##df = df[df["TYPE"].str.contains("DOI") == True]

## Assignment Only
#df = df[df["TYPE"].str.contains("EXP") == True]

## Or TRD + DOI + Assignments, remove all but those 3
#df = df[df["TYPE"].str.contains("BAL") == False]
#df = df[df["TYPE"].str.contains("EFN") == False]
#df = df[df["TYPE"].str.contains("RAD") == False]
#df = df[df["TYPE"].str.contains("CRC") == False]
#df = df[df["TYPE"].str.contains("EXP") == False]

# Save the dataframe for troubleshooting
df.to_csv('2025-trd-only.csv')

# Total number of trades
print('Total Number of Trades:', df.count())
print('Total Number of PUTS:', (df["DESCRIPTION"].str.contains("PUT")).sum())
print('Total Number of CALLS:', (df["DESCRIPTION"].str.contains("CALL")).sum())

## To fix ValueError: could not convert string to float: '-13,800.00'
df['AMOUNT'] = [float(str(i).replace(',', '')) for i in df['AMOUNT']]

## Remove unnecessary columns
df.drop('TIME', axis=1, inplace=True)
df.drop('REF #', axis=1, inplace=True)
df.drop('BALANCE', axis=1, inplace=True)

# was here
## Need to get AMOUNT by MONTH
## Create a column for recasted DATE
df['DATE_RECAST'] = pd.to_datetime(df['DATE'],format='%m/%d/%y')

## Create a column for recasted AMOUNT
## To fix ValueError: could not convert string to float: '-13,800.00'
df['AMOUNT'] = [float(str(i).replace(',', '')) for i in df['AMOUNT']]

df['AMOUNT_RECAST'] = df['AMOUNT'].astype(float)

## Group By MONTH and AMOUNT
## Interesting issue with how the groupby index will be (year, month). But, I want
## year and month in the dataframe.
## Add year and month.
df['YEAR'] = df['DATE_RECAST'].dt.year
df['MONTH'] = df['DATE_RECAST'].dt.month

df_new = df.groupby( [ df['YEAR'], df['MONTH'] ] ).agg({'AMOUNT_RECAST': sum})

## reset index
df_new = df_new.reset_index()


# Calculate APY
## Keep it simple and just use 31 days per month
number_of_days = 31
#investment_amount = 80000.00

## Monthly APY

total_amount = df_new['AMOUNT_RECAST']

calc_apy = calculate_apy(investment_amount, total_amount, number_of_days)

df_new['monthly_apy'] = calc_apy

## Save the dataframe for troubleshooting
df_new.to_csv('latest.csv')

for index, row in df_new.iterrows():
    print('Date ', row['MONTH'], row['YEAR'], 'Premiums =', row['AMOUNT_RECAST'], 'APY =', row['monthly_apy'])

# For Inception and all TRD, EXP and DOI, need to add back any equity holdings
# Add back Wayfarer equity of 2 contracts at $62 = 12200
##print('Total Gain/Loss=', (df_new['AMOUNT_RECAST'].sum())  + 12200.00)  

# Just Premiuns
print('Total Gain/Loss=', (df_new['AMOUNT_RECAST'].sum()))  

## Calculate Total APY
total_amount = df_new['AMOUNT_RECAST'].sum()

## Or periods needs to be total days
periods = number_of_days * df_new.shape[0]

print('periods=', periods)

# For Inception and all TRD, EXP and DOI, need to add back any equity holdings
# Add back Wayfarer equity of 2 contracts at $62 = 12200
##calc_apy = calculate_apy(investment_amount, total_amount  + 12200.00, periods)

# Add back RDDT equity of 3 contracts at $55000
##calc_apy = calculate_apy(investment_amount, total_amount  + 55000.00, periods)

# Just Premiums
calc_apy = calculate_apy(investment_amount, total_amount, periods)

print('Calc APY=', calc_apy)

# Visualization

## Plot the data

## Create a Plot
plt.plot(df_new['AMOUNT_RECAST'], color='red', marker='o')
plt.title('Monthly Amounts', fontsize=14)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Amount', fontsize=14)

plt.grid(True)

# Setting the number of ticks and show Month
# Only show monthly ticks not the half points ie. 1, 2, 3, 4 not 1, 1.5, 2, 2.5...
# https://www.geeksforgeeks.org/how-to-change-the-number-of-ticks-in-matplotlib/
# https://stackoverflow.com/questions/2497449/plot-string-values-in-matplotlib
#plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])

plt.show(block=True)



