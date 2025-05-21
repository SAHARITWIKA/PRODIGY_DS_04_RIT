import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import requests
from io import StringIO

# Step 1: Load dataset from GitHub
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# Step 2: Prepare the data
data = data[['label', 'tweet']]
data.columns = ['sentiment', 'tweet']
data['sentiment'] = data['sentiment'].map({0: 'Negative', 1: 'Positive'})

# Step 3: Plot sentiment distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='sentiment', palette='Set2')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()
plt.close()

# Step 4: WordCloud for each sentiment
stopwords = set(STOPWORDS)
for sentiment in data['sentiment'].unique():
    text = " ".join(data[data['sentiment'] == sentiment]['tweet'])
    wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {sentiment} Tweets")
    plt.tight_layout()
    plt.savefig(f"wordcloud_{sentiment.lower()}.png")
    plt.show()
    plt.close()
