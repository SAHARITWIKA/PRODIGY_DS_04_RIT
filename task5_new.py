import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample traffic accident dataset
data = {
    'Start_Time': ['2022-01-01 08:00:00', '2022-01-01 17:30:00', '2022-01-02 13:15:00', '2022-01-03 22:00:00'],
    'Weather_Condition': ['Clear', 'Rain', 'Cloudy', 'Snow'],
    'Road_Condition': ['Dry', 'Wet', 'Dry', 'Snowy'],
    'Severity': [2, 3, 2, 4],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Denver'],
    'Latitude': [40.7128, 34.0522, 41.8781, 39.7392],
    'Longitude': [-74.0060, -118.2437, -87.6298, -104.9903]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert Start_Time to datetime and extract Hour
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Hour'] = df['Start_Time'].dt.hour

# Set seaborn style
sns.set(style="whitegrid")

# 1. Accidents by Weather Condition
plt.figure(figsize=(6, 4))
sns.countplot(x='Weather_Condition', data=df, palette='viridis')
plt.title('Accidents by Weather Condition')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('accidents_by_weather.png')
plt.show()

# 2. Accidents by Time of Day (Hour)
plt.figure(figsize=(6, 4))
sns.histplot(df['Hour'], bins=8, kde=False, color='skyblue')
plt.title('Accidents by Time of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('accidents_by_hour.png')
plt.show()

# 3. Accident Severity by Road Condition
plt.figure(figsize=(6, 4))
sns.boxplot(x='Road_Condition', y='Severity', data=df, palette='pastel')
plt.title('Severity by Road Condition')
plt.tight_layout()
plt.savefig('severity_by_road_condition.png')
plt.show()

# 4. Accident Map (City-wise Location Plot)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Longitude', y='Latitude', hue='City', data=df, s=100, palette='Set2')
plt.title('Accident Locations by City')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('accident_map_citywise.png')
plt.show()
