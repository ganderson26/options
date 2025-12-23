import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm
import pandas_datareader.data as web
import datetime

input_data_path = 'ticker-short.csv'
ticker_cik = pd.read_csv(input_data_path, delimiter=',')
sym_cik = ticker_cik.copy(deep=True)
sym_cik.set_index('Ticker', inplace=True)
cik_sym = ticker_cik.copy(deep=True)
cik_sym.set_index('CIK', inplace=True)

#Looks up Edgar CIK Number
def symbol_to_cik(symbols):
    new_symbols = [i.lower() for i in symbols]
    cik = [sym_cik.loc[i, 'CIK'] for i in new_symbols]
    return cik
#Looks up Symbol from CIK Number:
def cik_to_symbol(ciks):
    tickers = [cik_sym.loc[i, 'Ticker'] for i in ciks]
    new_tickers = [i.upper() for i in tickers]
    return new_tickers
#Turns URL into Soup object
def to_soup(url):
    url_response = requests.get(url)
    webpage = url_response.content
    soup = BeautifulSoup(webpage, 'html.parser')
    return soup
#Calculates the period return
def return_calc(symbol,start,end):
    data = web.DataReader(symbol, 'yahoo', start, end)
    end_price = data['Adj Close'][-1]
    beg_price = data['Adj Close'][0]
    ret = round(((end_price/beg_price)-1)*100, 2)
    return ret

#Picks up the Insider Trades data
def insider_trading_all(symbol_list, end_date, export_file=0, file_name=''):
    symbols = symbol_list
    end = end_date
    start_yahoo = datetime.datetime(int(end[:4]),int(end[5:7]),int(end[8:]))
    end_yahoo = datetime.date.today()
    dfs = []
    with tqdm(total = len(symbols)) as pbar:
        for i in range(len(symbols)):
            try:
                lst = [symbols[i]]
                cik = symbol_to_cik(lst)
                page = 0
                beg_url = 'https://www.sec.gov/cgi-bin/own-disp?action=getissuer&CIK='+str(cik[0])+'&type=&dateb=&owner=include&start='+str(page*80)
                urls = [beg_url]
                df_data = []
                for url in urls:
                    soup = to_soup(url)
                    print('url=', soup)
                    #response = requests.get(url)
                    #html = response.text
                    #soup = BeautifulSoup(html, "lxml")
                    transaction_report = soup.find('table', id='transaction-report')

                    print('transaction_report=', transaction_report)

                    print('HERE 1')

                    t_chil = [i for i in transaction_report.children]
                    print('HERE 2')
                    t_cont = [i for i in t_chil if i != '\n']
                    print('HERE 3')

                    headers = [ i for i in t_cont[0].get_text().split('\n') if i != '']
                    data_rough = [i for lst in t_cont[1:] for i in lst.get_text().split('\n') if i != '' ]
                    data = [data_rough[i:i+12] for i in range(0,len(data_rough), 12)]
                    last_line = data[-1]

                    print('t_chil' + t_chil)

                    for i in data:
                        if (end > i[1]):
                            break
                        else:
                            if (i != last_line):
                                df_data.append(i)
                            else:
                                df_data.append(i)
                                page += 1
                                urls.append('https://www.sec.gov/cgi-bin/own-disp?action=getissuer&CIK='+str(cik[0])+'&type=&dateb=&owner=include&start='+str(page*80))

                df = pd.DataFrame(df_data,columns = headers)  

                print(df)

              
            except:
                print('ERROR')
                pbar.update(1)
                continue
    
   

#User Input:
all_tickers = input('Do you want to search through all listed symbols? Y/N:  ')
if (all_tickers == 'Y') | (all_tickers == 'y'):
    symbols = [i.upper() for i in ticker_cik.Ticker]
elif (all_tickers =='N') | (all_tickers =='n'):
    symbol_input = input('Enter symbols you would like analyzed(Ex: AAPL TSLA):  ')
    symbols = [i for i in symbol_input.split()]
date = input('How far back would you like to go?(YYYY-MM-DD):  ')
export_file = eval(input('Export file?(0 or 1):  '))
if (export_file == 1):
    file_name = input('What would you like to name the file?:  ')
insider_trading_all(symbols, date, export_file, file_name)
