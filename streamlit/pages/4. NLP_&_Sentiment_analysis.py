# Standard libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
import json
import requests
import folium
from folium.plugins import FastMarkerCluster, Fullscreen, MiniMap, HeatMap, HeatMapWithTime, LocateControl
from wordcloud import WordCloud
from collections import Counter
from PIL import Image

# DataPrep
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# Reading all the files
raw_path = '../data/'
olist_customer = pd.read_csv(raw_path + 'olist_customers_dataset.csv')
olist_geolocation = pd.read_csv(raw_path + 'olist_geolocation_dataset.csv')
olist_orders = pd.read_csv(raw_path + 'olist_orders_dataset.csv')
olist_order_items = pd.read_csv(raw_path + 'olist_order_items_dataset.csv')
olist_order_payments = pd.read_csv(raw_path + 'olist_order_payments_dataset.csv')
olist_order_reviews = pd.read_csv(raw_path + 'olist_order_reviews_dataset.csv')
olist_products = pd.read_csv(raw_path + 'olist_products_dataset.csv')
olist_sellers = pd.read_csv(raw_path + 'olist_sellers_dataset.csv')

def without_hue(ax, feature):
    total = len(feature)
    
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2 - 0.12  
        y =  p.get_height()
        
        ax.annotate(percentage, (x, y), size = 10)


st.set_page_config(page_title="NLP & Sentiment analysis" , page_icon="💻")

st.sidebar.markdown("# 💻 NLP & Sentiment analysis")

st.title("Sentiment analysis and EDA of a Brezilian E-commerce")
st.header("Part 4 : NLP & Sentiment analysis")

st.markdown("""Dans cette section, j'utiliserai les notes et les commentaires conservés par les clients après leur achat pour construire un modèle d'analyse des sentiments capable de prédire le sentiment des futurs clients (qu'il soit positif ou négatif).

Je passerai par une série d'étapes telles que le prétraitement du texte, l'extraction des caractéristiques, l'étiquetage des données, la construction d'un pipeline, la classification des sentiments et enfin la mise en œuvre pour voir si l'algorithme peut prédire avec précision le sentiment du client sur la base de son évaluation.""")

#Extract the score and review left by the client and drop NAs from dataset
commentsData = olist_order_reviews.loc[:, ['review_score', 'review_comment_message']]
commentsData = commentsData.dropna(subset = ['review_comment_message'])
commentsData = commentsData.reset_index(drop = True)

#Get the shape of the dataset
print(f'Dataset shape: {commentsData.shape}')

#Rename the columns to 'score' and 'comment'
commentsData.columns = ['score', 'comment']

st.write(commentsData)

st.markdown("""Le tableau ci-dessus permet de déduire qu'il y a environ 42 000 avis qui peuvent être utilisés pour former un modèle d'analyse des sentiments.

On peut également en déduire que l'échelle de notation va de 1 à 5 (1 étant la plus mauvaise note et 5 la meilleure).""")

st.header("Text preprocessing")
st.markdown("""Cette partie est la plus cruciale de l'analyse, car nous devons nous assurer que nous disposons de données textuelles propres, exemptes de lignes de rupture, d'espaces, de caractères spéciaux, d'hyperliens, de structures non uniformes (majuscules et minuscules), etc. Pour ce faire, j'utiliserai le paquet d'expressions régulières de Python (RegEx), qui est utile pour les transformations de texte.""")

def find_patterns(re_pattern, text_list):
  
    #Compile the regular expressions passed as arguments
    p = re.compile(re_pattern) #re_pattern is the regex pattern that will be used in the search
    positions_dict = {} #the index of the text of interest
    i = 0
    for c in text_list:
        match_list = []
        iterator = p.finditer(c)
        for match in iterator:
            match_list.append(match.span())
        control_key = f'Text idx {i}'
        if len(match_list) == 0:
            pass
        else:
            positions_dict[control_key] = match_list
        i += 1

    return positions_dict


def print_step_result(text_list_before, text_list_after, idx_list):
  
    #Iterate over string examples
    i = 1
    for idx in idx_list:
        print(f'--- Text {i} ---\n')
        print(f'Before: \n{text_list_before[idx]}\n')
        print(f'After: \n{text_list_after[idx]}\n')
        i += 1

#Import regex package as it will be used for all RegEx transformations
import re

def re_breakline(text_list):
    
    #Apply regex
    return [re.sub('[\n\r]', ' ', r) for r in text_list]
#Create list of comment reviews
reviews = list(commentsData['comment'].values)

#Apply regex and add the column to dataset
reviews_breakline = re_breakline(reviews)
commentsData['re_breakline'] = reviews_breakline



def re_hyperlinks(text_list):
    
    #Apply regex
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(pattern, ' link ', r) for r in text_list]
#Apply regex and add the column to dataset
reviews_hyperlinks = re_hyperlinks(reviews_breakline)
commentsData['re_hyperlinks'] = reviews_hyperlinks

#Verify results
print_step_result(reviews_breakline, reviews_hyperlinks, idx_list = [10607])


def re_dates(text_list):
    
    #Apply regex
    pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [re.sub(pattern, ' date ', r) for r in text_list]
#Apply regex and add the column to dataset
reviews_dates = re_dates(reviews_hyperlinks)
commentsData['re_dates'] = reviews_dates

#Verify results
print_step_result(reviews_hyperlinks, reviews_dates, idx_list = [178])


def re_money(text_list):
    
    #Apply regex
    pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [re.sub(pattern, ' money ', r) for r in text_list]
#Apply regex and add the column to dataset
reviews_money = re_money(reviews_dates)
commentsData['re_money'] = reviews_money

#Verify results
print_step_result(reviews_dates, reviews_money, idx_list = [1076])

def re_numbers(text_list):

    #Apply regex
    return [re.sub('[0-9]+', ' number ', r) for r in text_list]
#Apply regex and add the column to dataset
reviews_numbers = re_numbers(reviews_money)
commentsData['re_numbers'] = reviews_numbers

#Verify results
print_step_result(reviews_money, reviews_numbers, idx_list = [68])


def re_special_chars(text_list):
    
    #Apply regex
    return [re.sub('\W', ' ', r) for r in text_list]
#Apply regex and add the column to dataset
reviews_special_chars = re_special_chars(reviews_numbers)
commentsData['re_special_chars'] = reviews_special_chars

#Verify results
print_step_result(reviews_money, reviews_special_chars, idx_list = [40972])

def re_whitespaces(text_list):
    
    #Apply regex
    white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
    white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
    return white_spaces_end
#Apply regex and add the column to dataset
reviews_whitespaces = re_whitespaces(reviews_special_chars)
commentsData['re_whitespaces'] = reviews_whitespaces

#Verify results
print_step_result(reviews_special_chars, reviews_whitespaces, idx_list = [3342])


#Define function to remove the stopwords and to make the comments in lowercase
def stopwords_removal(text, cached_stopwords = stopwords.words('portuguese')):
    
    return [c.lower() for c in text.split() if c.lower() not in cached_stopwords] #cached_ are the stopwords to use
#Remove stopwords and add column to dataset
reviews_stopwords = [' '.join(stopwords_removal(review)) for review in reviews_whitespaces]
commentsData['stopwords_removed'] = reviews_stopwords

#Verify results
print_step_result(reviews_whitespaces, reviews_stopwords, idx_list = [43, 45])

def stemming_process(text, stemmer=RSLPStemmer()):
    
    return [stemmer.stem(c) for c in text.split()]
# Applying stemming and looking at some examples
reviews_stemmer = [' '.join(stemming_process(review)) for review in reviews_stopwords]
commentsData['stemming'] = reviews_stemmer

print_step_result(reviews_stopwords, reviews_stemmer, idx_list=[45])

#Get old shape of dataset
print('The old dataset has the following shape: ', commentsData.shape)

#Drop all rows that contain the word 'number' as it is abundant
commentsData_new = commentsData[~commentsData.stemming.str.contains("number")]

#Get new shape of dataset
print('The new dataset has the following shape: ', commentsData_new.shape)

def extract_features_from_corpus(corpus, vectorizer, df = False):
    
    #Extract features
    corpus_features = vectorizer.fit_transform(corpus).toarray()
    features_names = vectorizer.get_feature_names_out()
    
    #Transform into a dataframe
    df_corpus_features = None
    if df:
        df_corpus_features = pd.DataFrame(corpus_features, columns = features_names)
    
    return corpus_features, df_corpus_features

#Define and store english stopwords in 'en_stopwords'
en_stopwords = stopwords.words('portuguese')

#Create object for the CountVectorizer class
count_vectorizer = CountVectorizer(max_features = 500, min_df = 7, max_df = 0.8, stop_words = en_stopwords)

#Extract features for the corpus
countv_features, df_countv_features = extract_features_from_corpus(commentsData_new['stemming'],
                                                                   count_vectorizer, df = True)

#Get shape of dataframe
print(f'Shape of countv_features matrix: {countv_features.shape}\n')

#Create an object for the TF-IDF class
tfidf_vectorizer = TfidfVectorizer(max_features = 500, min_df = 7, max_df = 0.8, stop_words = en_stopwords)

#Extracting and add features into the corpus
tfidf_features, df_tfidf_features = extract_features_from_corpus(commentsData_new['stemming'],
                                                                 tfidf_vectorizer, df = True)

#Get shape of dataframe
print(f'Shape of tfidf_features matrix: {tfidf_features.shape}\n')

#Get snapshot of data
df_tfidf_features.head()

#Initialize plot
fig, ax = plt.subplots(figsize = (10,10))

#Plot the countplot
large_small = commentsData_new.groupby("score").size().sort_values(ascending=False).index[::1]
sns.countplot(commentsData_new, x = 'score', ax = ax, order= large_small)
without_hue(ax , commentsData_new.score)

st.header("Lebelling data")
st.markdown("""Pour entraîner un modèle d'analyse des sentiments, j'aurai besoin d'étiqueter mes principaux sentiments (positifs et négatifs) afin d'appliquer un algorithme de machine learning puisque mon ensemble de données ne contient pas encore cet étiquetage.

Pour ce faire, j'utiliserai les notes (1 --> 5) laissées par le client afin de catégoriser son commentaire comme positif ou négatif.""")
st.pyplot(fig)


st.header("Build scoring map and plot pie chart with frequency of sentiments")
st.markdown("""Pour cette analyse, je considérerai que les scores 4 et 5 représentent un retour d'information positif tandis que les scores 1, 2 et 3 représentent un retour d'information négatif.""")

#Build a scoring map that gives each score its respective sentiment
score_map = {
    1: 'negative',
    2: 'negative',
    3: 'positive',
    4: 'positive',
    5: 'positive'
}
#Add the sentiment label to the dataset and map the scores created
commentsData_new['sentiment_label'] = commentsData_new['score'].map(score_map)
pie = commentsData_new['sentiment_label'].value_counts()


#Plot the pie chart



#Define the plot
# fig = plt.subplots(figsize = (10,10))

# colors = sns.color_palette('pastel')[0:5]
# y = [pie[0],pie[1]]
# plt.pie(y, labels =["positive","negative"] , colors = colors, autopct='%.0f%%')


# st.pyplot(fig)

st.header("Get the main n-grams presented in the corpus on positive and negative classes (unigram, bigram, trigram)")

st.markdown("""In this part, I will extract the unigram, bigram and trigrams of text data for positive and negative sentiments and see if the model works well at identify positive words from negative words. To do so, I will define a function that returns the ngams from the bag of words.

""")

def ngrams_count(corpus, ngram_range, n = -1, cached_stopwords = stopwords.words('english')):
    
    #Use countvectorizer to build a bag of words using the given corpus
    vectorizer = CountVectorizer(stop_words = cached_stopwords, ngram_range = ngram_range).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    sum_words = bag_of_words.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    total_list = words_freq[:n]
    
    #Return a dataframe with the ngrams count
    count_df = pd.DataFrame(total_list, columns = ['ngram', 'count'])
    return count_df

#Split corpus into positive and negative comments
positive_comments = commentsData_new.query('sentiment_label == "positive"')['stemming']
negative_comments = commentsData_new.query('sentiment_label == "negative"')['stemming']

#Extract the top 10 unigrams by sentiment
unigrams_pos = ngrams_count(positive_comments, (1, 1), 10)
unigrams_neg = ngrams_count(negative_comments, (1, 1), 10)

#Extract the top 10 bigrams by sentiment
bigrams_pos = ngrams_count(positive_comments, (2, 2), 10)
bigrams_neg = ngrams_count(negative_comments, (2, 2), 10)

#Extracting the top 10 trigrams by sentiment
trigrams_pos = ngrams_count(positive_comments, (3, 3), 10)
trigrams_neg = ngrams_count(negative_comments, (3, 3), 10)

#Create dictionary with the n-grams created
ngram_dict_plot = {
    'Top Unigrams on Positive Comments': unigrams_pos,
    'Top Unigrams on Negative Comments': unigrams_neg,
    'Top Bigrams on Positive Comments': bigrams_pos,
    'Top Bigrams on Negative Comments': bigrams_neg,
    'Top Trigrams on Positive Comments': trigrams_pos,
    'Top Trigrams on Negative Comments': trigrams_neg,
}

#Plot the n-grams
#Initialize plot
fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (15, 18))
i, j = 0, 0
colors = ['Greens_d', 'Reds_d']

#Iterate through the data and plot
for title, ngram_data in ngram_dict_plot.items():
    ax = axs[i, j]
    sns.barplot(x = 'count', y = 'ngram', data = ngram_data, ax = ax, palette = colors[j])
    
    #Customize plots
    ax.set_title(title, size = 14)
    ax.set_ylabel('')
    ax.set_xlabel('')
    
    #Increment index
    j += 1
    if j == 2:
        j = 0
        i += 1

#Display plot        
plt.tight_layout()
st.pyplot(fig)

#Define class that will apply all regex transformations performed
class ApplyRegex(BaseEstimator, TransformerMixin):
    
    def __init__(self, regex_transformers):
        self.regex_transformers = regex_transformers
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        #Apply all regex transformations performed above
        for regex_name, regex_function in self.regex_transformers.items():
            X = regex_function(X)
            
        return X

#Define class for stopwords removal from the corpus
class StopWordsRemoval(BaseEstimator, TransformerMixin):
    
    def __init__(self, text_stopwords):
        self.text_stopwords = text_stopwords
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return [' '.join(stopwords_removal(comment, self.text_stopwords)) for comment in X]


# Class for apply the stemming process
class StemmingProcess(BaseEstimator, TransformerMixin):
    
    def __init__(self, stemmer):
        self.stemmer = stemmer
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [' '.join(stemming_process(comment, self.stemmer)) for comment in X]
    

#Define class thats extracts features from corpus
class TextFeatureExtraction(BaseEstimator, TransformerMixin):
    
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return self.vectorizer.fit_transform(X).toarray()
    


# Defining regex transformers to be applied
regex_transformers = {
    'break_line': re_breakline,
    'hyperlinks': re_hyperlinks,
    'dates': re_dates,
    'money': re_money,
    'numbers': re_numbers,
    'special_chars': re_special_chars,
    'whitespaces': re_whitespaces
}


# Defining the vectorizer to extract features from text
pt_stopwords = stopwords.words('portuguese')
vectorizer = TfidfVectorizer(max_features=300, lowercase=False, min_df=7, max_df=0.8, stop_words=pt_stopwords)

# Building the Pipeline
text_pipeline = Pipeline([
    ('regex', ApplyRegex(regex_transformers)),
    ('stopwords', StopWordsRemoval(stopwords.words('portuguese'))),
    ('stemming', StemmingProcess(RSLPStemmer())),
    ('text_features', TextFeatureExtraction(vectorizer))
])

#Import required package
from sklearn.model_selection import train_test_split

#Define X (inputs) and y (target)
idx_reviews = olist_order_reviews['review_comment_message'].dropna().index
score = olist_order_reviews['review_score'][idx_reviews].map(score_map)

#Split train/test sets
X = list(olist_order_reviews['review_comment_message'][idx_reviews].values)
y = score.apply(lambda x: 1 if x == 'positive' else 0).values

#Apply pipeline and split the data
X_processed = text_pipeline.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size = .33, random_state = 42)

# Verify results
print(f'Length of X_train_processed: {len(X_train)} - Length of one element: {len(X_train[0])}')
print(f'Length of X_test_processed: {len(X_test)} - Length of one element: {len(X_test[0])}')

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print('Accuracy logistic regression:', accuracy_score(y_pred, y_test))

#Evaluate metrics
conf_mx = confusion_matrix(y_test, y_pred)

fig = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mx, cmap="Greens",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            annot=True, square=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
st.pyplot(fig)



