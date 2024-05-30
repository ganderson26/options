import pandas as pd
from datetime import date

# Read a CSV file into a DataFrame
# Make sure to clean up the csv removing all but the options data
df = pd.read_csv('2024-05-06-AccountStatement-FULL.csv')
#print(df.head())

# Remove all rows where TYPE != TRD
df = df[df["TYPE"].str.contains("TRD") == True]
#print(df)


# Remove all rows where DESCRIPTION != your stock
df = df[df["DESCRIPTION"].str.contains("CVNA") == True]

# Remove all rows where DESCRIPTION != PUT or CALL
##df = df[df["DESCRIPTION"].str.contains("PUT") == True]
print(df)

# Now there are only Trade rows for Specific Stock

# Remove unnecessary columns
df.drop('TIME', axis=1, inplace=True)
df.drop('REF #', axis=1, inplace=True)
df.drop('BALANCE', axis=1, inplace=True)
#print(df)

## To fix ValueError: could not convert string to float: '-13,800.00'
df['AMOUNT'] = [float(str(i).replace(',', '')) for i in df['AMOUNT']]
# Calulate sum including fees
# Recast AMOUNT from string to float
total = (df['AMOUNT'].astype(float)).sum()
print(total)







