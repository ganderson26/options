# Plot Monthly Premiums
# Calculate Monthly APY
# Calculate Total APY

# Pandas

import pandas as pd
import matplotlib.pyplot as plt

def calculate_apy(principal, final_amount, compounding_periods):
    """
    Calculate the Annual Percentage Yield (APY).

    :param principal: Initial principal amount
    :param final_amount: Final amount after one year
    :param compounding_periods: Number of compounding periods per year
    :return: APY as a percentage
    """
    # Calculate the interest rate per compounding period
    r_per_period = (final_amount / principal) ** (1 / compounding_periods) - 1
    
    # Calculate the APY
    apy = (1 + r_per_period) ** compounding_periods - 1
    
    # Convert to percentage
    apy_percentage = apy * 100
    
    return apy_percentage


## Read a CSV file into a DataFrame
# Contains various TYPES such as BAL, DOI and TRD
df = pd.read_csv('2024-07-05-AccountStatement-Inception.csv')

## Viewing Data
#print(df.head())
#print(df.tail())
#print(df.info())
#print(df.describe())


# Wrangling

## Remove all rows where TYPE != TRD
# Very odd issue dropping the first amount due to type = BAL? I changed the TYPE to TRD on the row.
# This emphasizes the need to understand the data
###df = df[df["TYPE"].str.contains("TRD") == True]



## Outlirers Made them EXP from TRD
##5/20/24,13:11:11,TRD,1000538387766,BOT +1 MRNA 100 21 JUN 24 75 CALL @65.55 CBOE,-0.01,-0.65,"-6,555.00","41,290.35"
##5/20/24,13:12:02,TRD,1000538387808,SOLD -100 MRNA @140.15,-0.13,,"14,015.00","55,305.22"


df = df[df["TYPE"].str.contains("EXP") == True]

##df = df[df["TYPE"].str.contains("BAL") == False]
##df = df[df["TYPE"].str.contains("EFN") == False]
##df = df[df["TYPE"].str.contains("RAD") == False]
##df = df[df["TYPE"].str.contains("CRC") == False]

# To see just credit/debit on assignment + DOI interest, remove TRD
##df = df[df["TYPE"].str.contains("TRD") == False]

# To see just premiums, remove DOI and EXP

## This works START shows month 5 and 6 IF you comment out the DOI ###
### IF you remove the DOI rows, this fails and does not show month 5, but 2 roms for month 6 ###
## 3 rows for month 5, 1 row for month 6 wne remove DOI

##df = df[df["TYPE"].str.contains("DOI") == False]

## Whem commented out, 4 rows for month 5 and 3 for month 6

## This works END ###


#df = df[df["TYPE"].str.contains("EXP") == False]


df.to_csv('latest.csv')



#print(df.head())


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

##print(df)

## https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-separately-from-pandas-datetime-column
## Define list of attributes required    
al = ['year', 'month', 'day']

## Define generator expression of series, one for each attribute
date_gen = (getattr(df['DATE_RECAST'].dt, i).rename(i) for i in al)

## Concatenate results and join to original dataframe
df = df.join(pd.concat(date_gen, axis=1))
###print(df)

## Create a column for recasted AMOUNT
## To fix ValueError: could not convert string to float: '-13,800.00'
df['AMOUNT'] = [float(str(i).replace(',', '')) for i in df['AMOUNT']]

df['AMOUNT_RECAST'] = df['AMOUNT'].astype(float)
#print(df['AMOUNT_RECAST'])


#df['month'] = df['month'] - 8
#print(df['month'])


## Group By MONTH and AMOUNT

# Groupby seems to act odd.
#df['monthly'] = df.groupby(df['month'])['AMOUNT_RECAST'].sum()
#df['monthly'] = df.groupby(df['month'])['AMOUNT_RECAST'].transform('sum')

# Instead, iterate through the rows by month. Sum the amounts and insert new row to new collection.
grouped_data = []

month = df['month'].iat[0]
monthly_amount = 0.0

#print(month)
#print(monthly_amount)

for index, row in df.iterrows():
    # should have 13 rows of data
    print(row['month'])
    
    # If NEW month
    # just read month 6, so save month 5 to array
    if month != row['month']:
        #print('month !=')
        #print(row['month'])

        current_amount = row['AMOUNT_RECAST']
        current_month = row['month']
        
        row['AMOUNT_RECAST'] = monthly_amount
        row['month'] = month
        
        grouped_data.append(row)  # should add month 5

        month = current_month  # set to 6
        monthly_amount = current_amount 
    else:
        #print('month +')
        monthly_amount = monthly_amount + row['AMOUNT_RECAST']  
        #print(monthly_amount)  

# Save last row  
row['AMOUNT_RECAST'] = monthly_amount
row['month'] = month
grouped_data.append(row) 

grouped_df = pd.DataFrame(grouped_data)
grouped_df.rename(columns={'AMOUNT_RECAST': 'monthly'}, inplace=True)

#print(grouped_df['month'], grouped_df['monthly'])






# Calculate APY
# Keep it simple and just use 31 days per month
number_of_days = 31

# APY
# rate = 1 + (total/investment_amount)
# days = days_in_year/number_of_days
# apy = ((rate to the power of days) -1) * 100
investment_amount = 50000.00
rate = 1 + (grouped_df['monthly'] / investment_amount)

# Get days value for Timedelta number_of_days variable
days_in_year = 365
days = days_in_year / number_of_days

apy = ((pow(rate, days)) - 1) * 100
grouped_df['monthly_apy'] = apy


# Drop duplicate rows based on 'col1' and 'col2'
####df.drop_duplicates(subset=['month'], inplace=True)

##df.to_csv('latest.csv')



####df.dropna(subset=['monthly_apy'], inplace=True)

##monthly_amount = 50000.00 + grouped_df['monthly']
##print(monthly_amount)

##grouped_df['monthly_apy'] = calculate_apy(50000.00, monthly_amount, 6)



for index, row in grouped_df.iterrows():
    print('Month ', row['month'], 'Premiums =', row['monthly'], 'APY =', row['monthly_apy'])

# Calculate Total APY
# Keep it simple and just use number of months/rows
total_premiums = grouped_df['monthly'].sum()
##total_apy = grouped_df['monthly_apy'].sum()
total_apy = grouped_df['monthly_apy'].sum() / grouped_df.shape[0]
print('Total Premiums =', total_premiums, 'Total APY =', total_apy)

# Visualization

## Plot the data

grouped_df.reset_index(drop=True, inplace=True)

#print(grouped_df)

## Create a Plot
plt.plot(grouped_df['monthly'], color='red', marker='o')
plt.title('Monthly Amounts', fontsize=14)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Amount', fontsize=14)

plt.grid(True)

# Setting the number of ticks and show Month
# Only show monthly ticks not the half points ie. 1, 2, 3, 4 not 1, 1.5, 2, 2.5...
# https://www.geeksforgeeks.org/how-to-change-the-number-of-ticks-in-matplotlib/
# https://stackoverflow.com/questions/2497449/plot-string-values-in-matplotlib
plt.xticks([0, 1, 2, 3, 4, 5], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])




plt.show(block=True)




# 2023 actual monthly amounts of 2755 for 4 months = average of 688.75 x 12 = 8265
#principal = 30000  # Initial principal amount
#final_amount = 32755  # Final amount after one year
#compounding_periods = 4  # Compounded monthly

#apy = calculate_apy(principal, final_amount, compounding_periods)
#print(f"The 2023 Annual Percentage Yield (APY) is: {apy:.2f}%")

# 2024 6 months of premiums x 2
#principal = 50000  # Initial principal amount
#final_amount = 64392  # Final amount after one year
#compounding_periods = 12  # Compounded monthly

#apy = calculate_apy(principal, final_amount, compounding_periods)
#print(f"The 2024 Annual Percentage Yield (APY) is: {apy:.2f}%")


# 2024 6 months of premiums
#principal = 50000  # Initial principal amount
#final_amount = 57196  # Final amount after one year
#compounding_periods = 6  # Compounded monthly

#apy = calculate_apy(principal, final_amount, compounding_periods)
#print(f"The Actual 2024 Annual Percentage Yield (APY) is: {apy:.2f}%")

# 2024 1 months of premiums
#principal = 50000  # Initial principal amount
#final_amount = 51265  # Final amount after one year
#compounding_periods = 1  # Compounded monthly

#apy = calculate_apy(principal, final_amount, compounding_periods)
#print(f"The Actual 2024 Annual Percentage Yield (APY) is: {apy:.2f}%")


