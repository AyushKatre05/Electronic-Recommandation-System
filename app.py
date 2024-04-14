import pandas as pd
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Snowball stemmer
stmer = SnowballStemmer('english')

# Read the dataset
df = pd.read_csv('DatafinitiElectronicsProductsPricingData.csv')

# Select only the desired columns
df = df[['id', 'name','brand', 'categories','sourceURLs']]

# Drop rows with missing values and duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Preprocess text data
df["name"] = df["name"].str.lower().replace("[^a-zA-Z0-9]", " ", regex=True)

# Function for tokenization and stemming
def tokenization(txt):
    tokens = nltk.word_tokenize(txt.lower())
    stemming = [stmer.stem(w) for w in tokens if not w in stopwords.words('english') and w.isalnum()]
    return " ".join(stemming)

# Apply tokenization to text columns
df['name'] = df['name'].apply(lambda x:tokenization(x))
df['categories'] = df['categories'].apply(lambda x:tokenization(x))
df['name_cate'] = df['name'] + " " + df['categories']

# Function to compute cosine similarity
def cosine_sim(txt1,txt2):
    obj_tfidf = TfidfVectorizer(tokenizer=tokenization)
    tfidfmatrix = obj_tfidf.fit_transform([txt1,txt2])
    similarity = cosine_similarity(tfidfmatrix)[0][1]
    return similarity

# Function to recommend products based on query
def recommendation(query):
    tokenized_query = tokenization(query)
    df['similarity'] = df['name_cate'].apply(lambda x: cosine_sim(tokenized_query,x))
    final_df = df.sort_values(by=['similarity'],ascending=False).head(20)[['name','brand','categories','sourceURLs']]
    return final_df

# Streamlit app
def main():
    st.title('Product Recommendation System')

    # Text input for user query
    query = st.text_input('Enter product name:', '')

    # Button to trigger recommendation
    if st.button('Get Recommendations'):
        final_df = recommendation(query)
        st.write(final_df)

if __name__ == "__main__":
    main()
