import json
import pandas as pd
import statsmodels.formula.api as smf

# Load the market and company data from saved CSV files
market_data = pd.read_csv('market_data.csv')
company_data = pd.read_csv('company_data.csv')

# Extract the sentiment logits from the logits_dict for each company and create a new DataFrame
logits_dict = json.load(open('logits.json'))
sentiment_data = pd.DataFrame(columns=['Date', 'Symbol'] + labels)
for symbol, symbol_logits in logits_dict.items():
    for filename, data in symbol_logits.items():
        date = data['Date']
        logits = data['logits']
        sentiment_data = sentiment_data.append({'Date': date, 'Symbol': symbol, **dict(zip(labels, logits))}, ignore_index=True)

# Merge the new DataFrame with the company data DataFrame using the date as the key
company_sentiment_data = pd.merge(company_data, sentiment_data, on=['Date', 'Symbol'], how='left')

# Merge the company data DataFrame with the market data DataFrame using the date as the key
merged_data = pd.merge(company_sentiment_data, market_data, on='Date', how='left')

# Drop any rows with NaN values in the new merged DataFrame
merged_data = merged_data.dropna()

# Create the regression model using the statsmodels library
model_formula = 'Returns ~ ' + ' + '.join(labels)
model = smf.ols(formula=model_formula, data=merged_data)

# Fit the regression model to the merged DataFrame
results = model.fit()

# Print the summary of the regression model
print(results.summary())

