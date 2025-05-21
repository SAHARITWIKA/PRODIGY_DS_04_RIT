import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Simulated gender population data
data = {
    'Gender': ['Male', 'Female'],
    'Population (in billions)': [4.05, 3.95]
}

df = pd.DataFrame(data)

# Bar Plot (Updated as per future warning)
plt.figure(figsize=(8, 6))
sns.barplot(x='Gender', y='Population (in billions)', hue='Gender', data=df, palette='Set2', legend=False)
plt.title('Global Gender Distribution (Simulated)', fontsize=14)
plt.xlabel('Gender')
plt.ylabel('Population (in billions)')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("gender_bar_chart_updated.png")  # Saving output as image
plt.show()
