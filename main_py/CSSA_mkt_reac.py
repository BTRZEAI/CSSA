import pandas as pd
import numpy as np
import requests
import datetime as dt

# Set the ticker and the time interval
ticker = 'ABEV3.SA'
start_date = '2001-01-01'
end_date = '2022-12-31'
api_key = '5d1a94f98fcd7b987c1187895de56ff5'

# Define the benchmark index
benchmark_ticker = '^BVSP'

# Get the stock and benchmark data
stock_url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={api_key}'
benchmark_url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{benchmark_ticker}?from={start_date}&to={end_date}&apikey={api_key}'

stock_data = requests.get(stock_url).json()
benchmark_data = requests.get(benchmark_url).json()

stock_df = pd.DataFrame(stock_data['historical']).set_index('date')
benchmark_df = pd.DataFrame(benchmark_data['historical']).set_index('date')

stock_returns = stock_df['adjClose'].pct_change()
benchmark_returns = benchmark_df['adjClose'].pct_change()

# Get the surprise data from the earnings announcements
earnings_dates = requests.get(f'https://financialmodelingprep.com/api/v3/earning_calendar?from={start_date}&to={end_date}&apikey={api_key}').json()
earnings_dates = [dt.datetime.strptime(date['date'], '%Y-%m-%d') for date in earnings_dates]
surprises = []
for date in earnings_dates:
    res = requests.get(f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?limit=1&date={date.date()}&apikey={api_key}")
    surprise = res.json()[0]['epsSurprise']
    surprises.append(surprise)

# Calculate the average surprise in the 48 and 72 hour intervals after the earnings announcement
surprises_48h = []
surprises_72h = []
for i in range(len(earnings_dates)):
    date = earnings_dates[i]
    try:
        surprise = surprises[i]
        idx = np.where(stock_returns.index == date)[0][0]
        surprise_48h = stock_returns.iloc[idx+1:idx+4].sum() - surprise
        surprise_72h = stock_returns.iloc[idx+1:idx+6].sum() - surprise
        surprises_48h.append(surprise_48h)
        surprises_72h.append(surprise_72h)
    except:
        pass

# Calculate the abnormal returns
CAR_48h = stock_returns.iloc[idx+1:idx+4].sum() - benchmark_returns.iloc[idx+1:idx+4].sum()
CAR_72h = stock_returns.iloc[idx+1:idx+6].sum() - benchmark_returns.iloc[idx+1:idx+6].sum()

# Print the results
print(f"Average surprise in the 48-hour interval: {np.mean(surprises_48h)}")
print(f"Average surprise in the 72-hour interval: {np.mean(surprises_72h)}")
print(f"Cumulative abnormal return in the 48-hour interval: {CAR_48h}")
print(f"Cumulative abnormal return in the 72-hour interval: {CAR_72h}")