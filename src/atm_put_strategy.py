# Strategy mto always sell put ATM weekly
# Get closing cost for each Friday

# Install Python
#https://www.python.org/downloads/

# Install packages from command line
#pip install pandas scikit-learn yfinance


# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Function to get historical stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    print(data)
    return data

# Function to prepare the data for training and testing
def prepare_data(data, window_size=5):
    data['Price_Up'] = data['Close'].shift(-window_size) > data['Close']
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    y = data['Price_Up'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X[:-window_size], y[:-window_size]

# Function to train the SVM model
def train_svm(X_train, y_train):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

# Main function
def main():
    # Define the stock and date range
    ticker = 'AAPL'  
    start_date = '2024-11-01'
    end_date = '2025-11-30'

    # Get historical stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Set the 'date_column' as the DataFrame index
    stock_data['Date'] = stock_data.index

    fridays_df = stock_data[stock_data['Date'].dt.dayofweek == 4]
   

    # Initialize a list to store results and a variable for the previous value
    current_value = 0.0
    previous_value = 0.0
    fridays_df['Win'] = ''
    results = []

    #print(fridays_df['Date'], fridays_df['Close'])
    #print(fridays_df.columns)

    for index, row in fridays_df.iterrows():
        #print(row['Close', 'AAPL'])
        current_value = row['Close', 'AAPL']
        #print('current_value', current_value, 'previous_value', previous_value)
        
        if previous_value is None:
            previous_value = current_value 
        else:
            if (current_value >= previous_value):
                row['Win'] = 'Win'
            else:
                row['Win'] = 'Loss'

        results.append(row.to_dict())  
        previous_value = current_value 
        

        

    results_df = pd.DataFrame(results)
    print(results_df) 
    results_df.to_csv('results_df.csv')       

    
    #print(results_df['Date'], results_df['Close'], results_df['Win'])

    '''
    # Prepare the data
    X, y = prepare_data(stock_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM model
    svm_model = train_svm(X_train, y_train)

    # Make predictions on the test data
    predictions = svm_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Predict the next few days
    last_data = X[-1].reshape(1, -1)
    prediction = svm_model.predict(last_data)
    
    if prediction[0]:
        print('Predicted price for ' + ticker + ' will go up by ' + end_date)
    else:
        print('Predicted price for ' + ticker + ' will go down by ' + end_date)

    '''

if __name__ == "__main__":
    main()
