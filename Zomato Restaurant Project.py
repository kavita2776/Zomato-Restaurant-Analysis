#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("zomato.csv")


# In[3]:


df.head(5)


# In[4]:


df = df[['name', 'location', 'rate', 'votes', 'approx_cost(for two people)', 'cuisines']]
df.columns = ['name', 'location', 'rating', 'votes', 'cost', 'cuisines']


# In[5]:


# Clean rating column — remove nulls, NEW, and - values
df = df[df['rating'].notnull()]
df = df[df['rating'] != 'NEW']
df = df[df['rating'] != '-']


# In[6]:


# Convert rating from "4.1/5" format to float 4.1
df['rating'] = df['rating'].str.replace(' ', '')
df['rating'] = df['rating'].str.split('/').str[0]
df['rating'] = df['rating'].astype(float)


# In[7]:


# Clean cost column — remove commas and currency symbols
df['cost'] = df['cost'].astype(str)
df['cost'] = df['cost'].str.replace(',', '')
df['cost'] = df['cost'].str.replace('₹', '')
df['cost'] = df['cost'].astype(float)


# In[8]:


# Clean votes column
df['votes'] = df['votes'].astype(int)


# In[9]:


# Drop any remaining nulls
df = df.dropna()

print("=== Data Cleaning Complete ===")
print(f"Clean dataset shape: {df.shape}")


# In[10]:


# Save cleaned data
df.to_csv(r"C:\Users\KAVITA PATIL\Downloads\Zomato-Restaurant-Analysis\cleaned_zomato.csv", index=False)
print("cleaned_zomato.csv saved.")


# In[11]:


# ============================================================
# PART 2 — K-MEANS CLUSTERING
# Goal: Segment restaurants into groups based on
#       rating, votes, and cost
# ============================================================


# In[12]:


print("\n=== Starting Clustering Analysis ===")


# In[13]:


# ── Step 1: Select the 3 features we will cluster on ────────
# We use rating, votes, and cost because these 3 numbers
# together describe a restaurant's quality, popularity, and
# price point — which is exactly what segmentation needs.


# In[14]:


features = df[['rating', 'votes', 'cost']].copy()


# In[15]:


# ── Step 2: Scale the features ──────────────────────────────
# Problem: votes can be in thousands (e.g. 5000) while
# rating is between 1-5. If we don't scale, votes will
# dominate the clustering just because the numbers are bigger.
# StandardScaler brings everything to the same scale.


# In[16]:


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print("Features scaled successfully.")


# In[17]:


# ── Step 3: Run K-Means with 4 clusters ─────────────────────
# We choose 4 clusters because we expect 4 natural groups:
# high/low rating crossed with high/low popularity.
# random_state=42 just means results are reproducible —
# running the script twice gives the same output.


# In[18]:


kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features_scaled)

print("K-Means clustering complete.")
print(f"Cluster distribution:\n{df['cluster'].value_counts()}")


# In[19]:


# ── Step 4: Understand what each cluster means ──────────────
# Look at the average rating, votes, and cost per cluster
# to understand what kind of restaurants are in each group.


# In[20]:


cluster_summary = df.groupby('cluster').agg(
    avg_rating = ('rating', 'mean'),
    avg_votes  = ('votes',  'mean'),
    avg_cost   = ('cost',   'mean'),
    count      = ('name',   'count')
).round(2)

print("\n=== Cluster Summary ===")
print(cluster_summary)


# In[21]:


# ── Step 5: Label each cluster with a meaningful name ───────
# Based on the summary above, we assign business-friendly
# names to each cluster number so it is easy to understand.


# In[22]:


def label_cluster(row):
    # Thresholds based on actual cluster summary from this dataset:
    # High votes = above 500, High rating = above 4.0
    if row['avg_rating'] >= 4.3 and row['avg_votes'] >= 1000:
        return 'Star restaurants'       # high quality + very popular (cluster 2)
    elif row['avg_rating'] >= 4.0 and row['avg_votes'] >= 500:
        return 'Hidden gems'            # good quality + moderate visibility (cluster 1)
    elif row['avg_rating'] >= 3.5 and row['avg_votes'] >= 100:
        return 'Budget crowd-pleasers'  # average quality + some popularity (cluster 3)
    else:
        return 'Underperformers'        # low quality + low popularity (cluster 0)


# In[23]:


cluster_labels = cluster_summary.apply(label_cluster, axis=1)
cluster_label_map = cluster_labels.to_dict()

df['segment'] = df['cluster'].map(cluster_label_map)

print("\n=== Segment Distribution ===")
print(df['segment'].value_counts())


# In[24]:


# ── Step 6: Save the segmented data ─────────────────────────

df.to_csv(r"C:\Users\KAVITA PATIL\Downloads\Zomato-Restaurant-Analysis\segmented_zomato.csv", index=False)
print("\nsegmented_zomato.csv saved.")


# In[26]:


# ── Step 7: Visualise the clusters ──────────────────────────
# Plot rating vs votes, coloured by segment.
# This chart is a screenshot you can add to your README
# and your Power BI dashboard.

colors = {
    'Star restaurants':      '#1a6fc4',
    'Hidden gems':           '#0f6e56',
    'Budget crowd-pleasers': '#ba7517',
    'Underperformers':       '#993c1d'
}

plt.figure(figsize=(10, 6))

for segment, group in df.groupby('segment'):
    plt.scatter(
        group['rating'],
        group['votes'],
        c=colors.get(segment, 'gray'),
        label=segment,
        alpha=0.5,
        s=20
    )

plt.title('Restaurant Segmentation — Rating vs Votes', fontsize=14)
plt.xlabel('Rating')
plt.ylabel('Votes')
plt.legend(title='Segment')
plt.tight_layout()
plt.savefig(r"C:\Users\KAVITA PATIL\Downloads\Zomato-Restaurant-Analysis\clustering_chart.png", dpi=150)
plt.show()
print("Chart saved as clustering_chart.png")


# In[27]:


# ── Step 8: Top hidden gems by location ─────────────────────
# Most useful business insight from clustering:
# Where are the hidden gems concentrated?
# These are restaurants with great quality but low online reach.

hidden_gems = df[df['segment'] == 'Hidden gems']

print("\n=== Top Locations with Hidden Gems ===")
print(hidden_gems['location'].value_counts().head(10))

print("\n=== Sample Hidden Gem Restaurants ===")
print(hidden_gems[['name', 'location', 'rating', 'votes', 'cost']].head(10).to_string())


# In[ ]:





# In[ ]:




