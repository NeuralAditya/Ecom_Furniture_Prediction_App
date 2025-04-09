import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from wordcloud import WordCloud

# Loading the dataset
df = pd.read_csv('data/ecommerce_furniture_dataset_2024.csv')

# Cleaning price columns (remove $ and convert to float)
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
df['originalPrice'] = df['originalPrice'].replace(r'[\$,]', '', regex=True)
df['originalPrice'] = pd.to_numeric(df['originalPrice'], errors='coerce')

# Droping rows with missing values
df.dropna(inplace=True)

# Creating discount_percentage feature
df['discount_percentage'] = ((df['originalPrice'] - df['price']) / df['originalPrice']) * 100

# Replacing rare shipping tags
df['tagText'] = df['tagText'].apply(lambda x: x if x in ['Free shipping', '+Shipping: $5.09'] else 'others')

# Encoding categorical column
le = LabelEncoder()
df['tagText'] = le.fit_transform(df['tagText'])

# TF-IDF vectorization for productTitle
tfidf = TfidfVectorizer(stop_words='english', max_features=100)
productTitle_tfidf = tfidf.fit_transform(df['productTitle'])
productTitle_df = pd.DataFrame(productTitle_tfidf.toarray(), columns=tfidf.get_feature_names_out())
df = pd.concat([df.reset_index(drop=True), productTitle_df.reset_index(drop=True)], axis=1)
df.drop(columns=['productTitle'], inplace=True)

# Defining features and target
X = df.drop(columns=['sold'])
y = df['sold']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation:\nMSE: {mse:.2f}\nR2 Score: {r2:.2f}")

# Creating model directory
os.makedirs('model', exist_ok=True)

# Saving model and vectorizer
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Creating graphs directory
os.makedirs('static/graphs', exist_ok=True)

# 1. Feature Importance
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="skyblue")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("static/graphs/feature_importance.png")
plt.close()

# 2. Predicted vs Actual
plt.figure(figsize=(8, 6))
sns.barplot(x=list(range(len(y_test[:50]))), y=y_test[:50], color='skyblue', label='Actual')
sns.barplot(x=list(range(len(y_pred[:50]))), y=y_pred[:50], color='salmon', alpha=0.7, label='Predicted')
plt.title("Predicted vs Actual (Sample)")
plt.legend()
plt.tight_layout()
plt.savefig("static/graphs/predicted_vs_actual.png")
plt.close()

# 3. Price vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['price'], y=y_pred, alpha=0.6)
plt.title("Price vs Predicted Sales")
plt.xlabel("Price")
plt.ylabel("Predicted Sold")
plt.tight_layout()
plt.savefig("static/graphs/price_vs_predicted.png")
plt.close()

# 4. Distribution of Sold
plt.figure(figsize=(8, 6))
sns.histplot(df['sold'], kde=True, bins=30)
plt.title("Distribution of Items Sold")
plt.tight_layout()
plt.savefig("static/graphs/distribution_sold.png")
plt.close()

# 5. Sales by Shipping Tag
plt.figure(figsize=(8, 6))
tag_means = df.groupby('tagText')['sold'].mean()
tag_labels = le.inverse_transform(tag_means.index)
sns.barplot(x=tag_labels, y=tag_means.values)
plt.title("Average Sold by Shipping Tag")
plt.tight_layout()
plt.savefig("static/graphs/sales_by_tag.png")
plt.close()

# 6. Discount % vs Sold
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['discount_percentage'], y=df['sold'], alpha=0.5)
plt.title("Discount % vs Sold")
plt.tight_layout()
plt.savefig("static/graphs/discount_vs_sold.png")
plt.close()

# 7. Top Keywords
word_freq = productTitle_df.sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
word_freq.plot(kind='bar', color='skyblue')
plt.title("Top 10 Product Title Keywords")
plt.tight_layout()
plt.savefig("static/graphs/top_keywords.png")
plt.close()

# 8. Price Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], bins=30, kde=True)
plt.title("Price Distribution")
plt.tight_layout()
plt.savefig("static/graphs/price_distribution.png")
plt.close()

# 9. Sold by Price Bracket
bins = [0, 50, 100, 200, 500, 1000, 5000]
labels = ['<50', '50-100', '100-200', '200-500', '500-1k', '1k+']
df['price_bracket'] = pd.cut(df['price'], bins=bins, labels=labels)
plt.figure(figsize=(8, 6))
sns.barplot(x='price_bracket', y='sold', data=df)
plt.title("Sold Items by Price Bracket")
plt.tight_layout()
plt.savefig("static/graphs/sold_by_price_bracket.png")
plt.close()

# 10. Sales by Discount Bin
discount_bins = [0, 10, 20, 30, 50, 100]
df['discount_bin'] = pd.cut(df['discount_percentage'], bins=discount_bins)
plt.figure(figsize=(8, 6))
sns.barplot(x='discount_bin', y='sold', data=df)
plt.title("Sales by Discount Bins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/graphs/sales_by_discount_bin.png")
plt.close()

# 11. Tag Text Breakdown
tag_counts = df['tagText'].value_counts().sort_index()
tag_labels = le.inverse_transform(tag_counts.index)
explode = [0.05 if val < tag_counts.max() * 0.1 else 0 for val in tag_counts.values]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green

plt.figure(figsize=(6, 6))
plt.pie(
    tag_counts.values,
    labels=tag_labels,
    autopct='%1.1f%%',
    explode=explode,
    startangle=140,
    colors=colors,
    wedgeprops={'edgecolor': 'white'}
)
plt.title("Tag Text Breakdown")
plt.tight_layout()
plt.savefig("static/graphs/tag_text_breakdown.png")
plt.close()
