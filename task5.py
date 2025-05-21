import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/2016-weather-data-seattle.csv"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# Convert date to datetime
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Weekday'] = data['Date'].dt.day_name()

# Plot 1: Average temperature by month
plt.figure(figsize=(10, 5))
sns.barplot(x='Month', y='Mean_TemperatureC', data=data, palette='coolwarm')
plt.title('Average Monthly Temperature in Seattle (2016)')
plt.tight_layout()
plt.savefig('avg_temp_by_month.png')
plt.show()
plt.close()

# Plot 2: Weather conditions frequency
plt.figure(figsize=(10, 5))
top_weather = data['Events'].fillna('None').value_counts()
sns.barplot(x=top_weather.index, y=top_weather.values, palette='viridis')
plt.title('Weather Events Frequency')
plt.xlabel('Weather Event')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('weather_event_frequency.png')
plt.show()
plt.close()

# Plot 3: Temperature trend over time
plt.figure(figsize=(12, 5))
plt.plot(data['Date'], data['Mean_TemperatureC'], color='orange')
plt.title('Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Mean Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.savefig('temp_trend.png')
plt.show()
plt.close()
