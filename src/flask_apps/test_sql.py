
import mysql.connector

mydb = mysql.connector.connect(
host="localhost",
user="root",
password="Marathon#262",
database="OPTIONS"
)

#mydb = mysql.connector.connect(
#host="localhost",
#user="jejtxlk4zmlg",
#password="Marathon#262",
#database="OPTIONS"
#)

mycursor = mydb.cursor()

#str_stock = str(stock)
#str_expiry = str(expiry)
#str_strike = str(strike)

# local
#local_future = str(year) + '-' + str(month) + '-' + str(Day)
#str_expiry = local_future


#sql = "INSERT INTO OPTIONS.OPTIONS_DATA (USER_NAME, TICKER, EXPIRATION_DATE, STRIKE, CALL_PUT, BUY_SELL, DELTA, VOLUME, BID_ASK, ROR, IV, NOTES) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
#values = (name, str_stock, str_expiry, str_strike, call_put, buy_sell, option_delta, option_volume, option_bid, option_ror, option_iv, notes)

sql = "INSERT INTO OPTIONS.OPTIONS_DATA (USER_NAME, TICKER, EXPIRATION_DATE, STRIKE, CALL_PUT, BUY_SELL, DELTA, VOLUME, BID_ASK, ROR, IV, NOTES) VALUES ('max', 'mstr', '2024-12-20', 300.00, 'PUT', 'SELL', 0.99, 1234.0, 234.0, 1.0, 22.2, 'TESTING')"
mycursor.execute(sql)
#mycursor.execute(sql, values)

mydb.commit()

print(mycursor.rowcount, "record inserted.")

mydb.close()