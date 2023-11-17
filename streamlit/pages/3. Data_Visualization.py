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
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
from wordcloud import WordCloud
from collections import Counter
from PIL import Image


import lightgbm as lgb
import streamlit as st
import warnings

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

st.set_page_config(page_title="Data Vizualisation", page_icon="📊")

st.sidebar.markdown("# 📊 Data Vizualisation")

st.title("Sentiment analysis and EDA of a Brezilian E-commerce")
st.header("Part 3 : Data Vizualisation")

st.markdown(
    """Nous allons maintenant procéder à une analyse exploratoire des données afin d'obtenir des informations sur le E-commerce au Brésil. L'objectif est de diviser cette session en thèmes afin de pouvoir explorer les graphiques pour chaque sujet (commandes, clients, produits, articles et autres).
"""
)

st.header("**Total Orders on E-Commerce**")

st.markdown("""Nous savons que le E-commerce est une tendance en pleine expansion à l'échelle mondiale. Plongeons dans l'ensemble de données des commandes pour voir comment cette tendance peut être présentée au Brésil, du moins dans la limite de notre dataset.
En examinant les colonnes de notre dataset, nous pouvons voir les commandes avec différents statuts et différentes colonnes d'horodatage comme l'achat, l'approbation, la livraison et la livraison estimée. Examinons tout d'abord le statut des commandes figurant dans cet ensemble de données.""")

def without_hue(ax, feature):
    total = len(feature)
    
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2 - 0.12  
        y =  p.get_height()
        
        ax.annotate(percentage, (x, y), size = 10)

fig = plt.figure(figsize=(14, 8))
ax = sns.countplot(olist_orders, x='order_status',  dodge ='auto')

without_hue(ax , olist_orders.order_status)
st.pyplot(fig)

# Changing the data type for date columns
timestamp_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
                  'order_estimated_delivery_date']
df_orders = olist_orders.merge(olist_customer, how='left', on='customer_id')
for col in timestamp_cols:
    df_orders[col] = pd.to_datetime(df_orders[col])
    
# Extracting attributes for purchase date - Year and Month
df_orders['order_purchase_year'] = df_orders['order_purchase_timestamp'].apply(lambda x: x.year)
df_orders['order_purchase_month'] = df_orders['order_purchase_timestamp'].apply(lambda x: x.month)
df_orders['order_purchase_month_name'] = df_orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%b'))
df_orders['order_purchase_year_month'] = df_orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%Y%m'))
df_orders['order_purchase_date'] = df_orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%Y%m%d'))

# Extracting attributes for purchase date - Day and Day of Week
df_orders['order_purchase_day'] = df_orders['order_purchase_timestamp'].apply(lambda x: x.day)
df_orders['order_purchase_dayofweek'] = df_orders['order_purchase_timestamp'].apply(lambda x: x.dayofweek)
df_orders['order_purchase_dayofweek_name'] = df_orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%a'))

# Extracting attributes for purchase date - Hour and Time of the Day
df_orders['order_purchase_hour'] = df_orders['order_purchase_timestamp'].apply(lambda x: x.hour)
hours_bins = [-0.1, 6, 12, 18, 23]
hours_labels = ['Dawn', 'Morning', 'Afternoon', 'Night']
df_orders['order_purchase_time_day'] = pd.cut(df_orders['order_purchase_hour'], hours_bins, labels=hours_labels)

fig = plt.figure(constrained_layout=True, figsize=(13, 10))

# Axis definition
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Lineplot - Evolution of e-commerce orders along time 
sns.lineplot(data=df_orders['order_purchase_year_month'].value_counts().sort_index(), ax=ax1, 
             color='darkslateblue', linewidth=2)
ax1.annotate(f'Highest orders \nreceived', (13, 7500), xytext=(-75, -25), 
             textcoords='offset points', bbox=dict(boxstyle="round4", fc="w", pad=.8),
             arrowprops=dict(arrowstyle='-|>', fc='w'), color='dimgrey', ha='center')
ax1.annotate(f'Noise on data \n(huge decrease)', (23, 0), xytext=(48, 25), 
             textcoords='offset points', bbox=dict(boxstyle="round4", fc="w", pad=.5),
             arrowprops=dict(arrowstyle='-|>', fc='w'), color='dimgrey', ha='center')
 
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
ax1.set_title('Evolution of Total Orders in Brazilian E-Commerce', size=14, color='dimgrey')

# Barchart - Total of orders by day of week
sns.countplot(df_orders, x='order_purchase_dayofweek', ax=ax2, palette='YlGnBu')
weekday_label = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax2.set_xticklabels(weekday_label)
ax2.set_title('Total Orders by Day of Week', size=14, color='dimgrey', pad=20)
without_hue(ax2 , df_orders.order_purchase_dayofweek)

# Barchart - Total of orders by time of the day
day_color_list = ['darkslateblue', 'deepskyblue', 'darkorange', 'purple']
sns.countplot(df_orders, x='order_purchase_time_day', ax=ax3, palette=day_color_list)
ax3.set_title('Total Orders by Time of the Day', size=14, color='dimgrey', pad=20)
without_hue(ax3 , df_orders.order_purchase_time_day)

plt.tight_layout()


st.markdown("""Le graphique ci-dessus permet de déduire que la majorité des commandes (97%) ont déjà été livrées au client final. Les 3% restants appartiennent aux 7 autres statuts (expédié, annulé, indisponible, facturé, en cours de traitement, créé et approuvé).""")

st.markdown("""**E-commerce trend in Brazil between 2016 and 2018**""")

st.markdown("Dans cette section, j'examinerai la tendance des achats en ligne au Brésil entre 2016 et 2018. L'objectif est de voir s'il y a une tendance à la hausse ou à la baisse, ou en d'autres termes, si les achats en ligne gagnent en popularité ou non au Brésil.")

st.pyplot(fig)

st.markdown("""
            Les graphiques ci-dessus permettent de déduire que :

            - Le commerce électronique au Brésil affiche une tendance à la hausse au fil des ans (entre 2016 et 2018).

            - Les achats les plus élevés ont été effectués en novembre 2017 (la raison possible est l'achat de cadeaux avant Noël ou le Black Friday).

            - Les lundis sont les jours préférés des clients pour faire des achats en ligne et ils ont également tendance à acheter davantage l'après-midi

            - Forte diminution entre août 2018 et septembre 2018 et peut-être que l'origine de cette diminution est liée au bruit dans les données.""")

# Creating figure
fig = plt.figure(constrained_layout=True, figsize=(13, 5))

# Axis definition
gs = GridSpec(1, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1:])

# Annotation - Grown on e-commerce orders between 2017 and 2018
df_orders_compare = df_orders.query('order_purchase_year in (2017, 2018) & order_purchase_month <= 8')
year_orders = df_orders_compare['order_purchase_year'].value_counts()
growth = int(round(100 * (1 + year_orders[2017] / year_orders[2018]), 0))
ax1.text(0.00, 0.73, f'{year_orders[2017]}', fontsize=40, color='mediumseagreen', ha='center')
ax1.text(0.00, 0.64, 'orders registered in 2017\nbetween January and August', fontsize=10, ha='center')
ax1.text(0.00, 0.40, f'{year_orders[2018]}', fontsize=50, color='darkslateblue', ha='center')
ax1.text(0.00, 0.31, 'orders registered in 2018\nbetween January and August', fontsize=10, ha='center')
signal = '+' if growth > 0 else '-'
ax1.text(0.00, 0.20, f'{signal}{growth}%', fontsize=14, ha='center', color='white', style='italic', weight='bold',
         bbox=dict(facecolor='darkslateblue', alpha=0.6, pad=10, boxstyle='round, pad=.7'))
ax1.axis('off')

# Bar chart - Comparison between monthly sales between 2017 and 2018
sns.countplot(df_orders_compare, x='order_purchase_month', hue='order_purchase_year', ax=ax2,
                 palette='YlGnBu')
month_label = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
ax2.set_xticklabels(month_label)
ax2.set_title('Total Orders Comparison Between 2017 and 2018 (January to August)', size=12, color='dimgrey', pad=20)
plt.legend(loc='lower right')
st.pyplot(fig)

st.markdown("Ce garphique nous permet de mesurer l'évolution des ventes entre 2017 et 2018 qui ont augmentés de 143%, ce qui est considérable")

st.header("Brazil Geospatial Analysis")

st.markdown("J'effectuerai une analyse géospatiale sur l'ensemble des données afin d'identifier les régions du Brésil qui font le plus d'achats en ligne. Je comparerai également les régions et tenterai d'interpréter la raison de ces différences dans l'utilisation des achats en ligne (peut-être que certaines régions sont plus riches que d'autres, l'accessibilité à l'internet, etc.)")

import requests
import urllib3
import ssl


class CustomHttpAdapter (requests.adapters.HTTPAdapter):
    # "Transport adapter" that allows us to use custom ssl_context.

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, ssl_context=self.ssl_context)


def get_legacy_session():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    session = requests.session()
    session.mount('https://', CustomHttpAdapter(ctx))
    return session

# Merging orders and order_items
df_orders_items = df_orders.merge(olist_order_items, how='left', on='order_id')

# Using the API to bring the region to the data
headers = {'Host': 'www.goodreturns.in',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:105.0) Gecko/20100101 Firefox/105.0'}
r = get_legacy_session().get("https://servicodados.ibge.gov.br/api/v1/localidades/mesorregioes")

content = [c['UF'] for c in json.loads(r.text)]
br_info = pd.DataFrame(content)
br_info['nome_regiao'] = br_info['regiao'].apply(lambda x: x['nome'])
br_info.drop('regiao', axis=1, inplace=True)
br_info.drop_duplicates(inplace=True)

# Threting geolocations outside brazilian map

#Brazils most Northern spot is at 5 deg 16′ 27.8″ N latitude.;
geo_prep = olist_geolocation[olist_geolocation.geolocation_lat <= 5.27438888]
#it’s most Western spot is at 73 deg, 58′ 58.19″W Long.
geo_prep = geo_prep[geo_prep.geolocation_lng >= -73.98283055]
#It’s most southern spot is at 33 deg, 45′ 04.21″ S Latitude.
geo_prep = geo_prep[geo_prep.geolocation_lat >= -33.75116944]
#It’s most Eastern spot is 34 deg, 47′ 35.33″ W Long.
geo_prep = geo_prep[geo_prep.geolocation_lng <=  -34.79314722]
geo_group = geo_prep.groupby(by='geolocation_zip_code_prefix', as_index=False).min()

# Merging all the informations
df_orders_items = df_orders_items.merge(br_info, how='left', left_on='customer_state', right_on='sigla')
df_orders_items = df_orders_items.merge(geo_group, how='left', left_on='customer_zip_code_prefix', 
                                        right_on='geolocation_zip_code_prefix')
# Filtering data between 201701 and 201808
df_orders_filt = df_orders_items[(df_orders_items['order_purchase_year_month'].astype(int) >= 201701)]
df_orders_filt = df_orders_filt[(df_orders_filt['order_purchase_year_month'].astype(int) <= 201808)]

# Grouping data by region
df_regions_group = df_orders_filt.groupby(by=['order_purchase_year_month', 'nome_regiao'], as_index=False)
df_regions_group = df_regions_group.agg({'customer_id': 'count', 'price': 'sum'}).sort_values(by='order_purchase_year_month')
df_regions_group.columns = ['month', 'region', 'order_count', 'order_amount']
df_regions_group.reset_index(drop=True, inplace=True)

# Grouping data by city (top 10)
df_cities_group = df_orders_filt.groupby(by='geolocation_city', 
                                       as_index=False).count().loc[:, ['geolocation_city', 'order_id']]
df_cities_group = df_cities_group.sort_values(by='order_id', ascending=False).reset_index(drop=True)
df_cities_group = df_cities_group.iloc[:10, :]

# Creating and preparing figure and axis
fig = plt.figure(constrained_layout=True, figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

# Count of orders by region
sns.lineplot(x='month', y='order_count', ax=ax1, data=df_regions_group, hue='region', 
             size='region', style='region', palette='magma', markers=['o'] * 5)

ax1.set_title('Evolution of E-Commerce Orders on Brazilian Regions', size=12, color='dimgrey')
ax1.set_ylabel('')
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)

# Top cities with more customers orders in Brazil
sns.barplot(y='geolocation_city', x='order_id', data=df_cities_group, ax=ax2, palette='magma')


ax2.set_title('Top 10 Brazilian Cities with More Orders', size=12, color='dimgrey')
ax2.set_ylabel('')

# Total orders by state
large_to_small = df_orders_filt.groupby('customer_state').size().sort_values().index[::-1]

sns.countplot(df_orders_filt, x='customer_state', ax=ax3,  palette='viridis', order = large_to_small)
ax3.set_title('Total of Customers Orders by State', size=12, color='dimgrey')
ax3.set_ylabel('')

st.pyplot(fig)

st.markdown("""Les graphiques ci-dessus montrent que la région sudeste (sud-est) a connu la plus forte évolution de commandes de clients entre 2017 et 2018, ce qui est conforme aux attentes puisque cette région du Brésil est relativement plus riche que les autres régions. On peut également déduire que Sao Paulo et Rio de Janeiro sont les deux premières villes brésiliennes en termes de nombre d'utilisateurs, ce qui est également conforme aux attentes, car ces deux villes sont les plus riches du Brésil et ont donc plus d'argent à dépenser pour les achats en ligne. Il semble y avoir une corrélation entre la richesse de la région et la quantité d'achats en ligne effectués (par exemple, les habitants de Salvador ne commandent pas beaucoup de produits en ligne, ce qui pourrait s'expliquer par le fait que Salvador est considérée comme une ville très pauvre au Brésil). """)

# Zipping locations
lats = list(df_orders_items.query('order_purchase_year == 2018')['geolocation_lat'].dropna().values)[:30000]
longs = list(df_orders_items.query('order_purchase_year == 2018')['geolocation_lng'].dropna().values)[:30000]
locations = list(zip(lats, longs))

# Creating a mapa using folium
map1 = folium.Map(location=[-15, -50], zoom_start=4.0)

# Plugin: FastMarkerCluster
FastMarkerCluster(data=locations).add_to(map1)

st_data = st_folium(map1)

st.header("Product Categories with Highest Scores")

st.markdown("Pour cette partie, je travaillerai avec les ensembles de données olist_order_items, olist_order_reviews et olist_products afin d'identifier les catégories de produits les plus demandées au Brésil.")
#Merge productsData and itemsData into one unique dataset
products_items = pd.merge(olist_products, olist_order_items, how = 'inner', on = 'product_id')

#Merge products_items with reviewsData for final dataset
categoriesData = pd.merge(products_items, olist_order_reviews, how = 'inner', on = 'order_id')

#Create an aggregation of categories with their respective scores (based on review scores)
avg_score_per_category = categoriesData.groupby('product_category_name', as_index = False).agg({'review_score': ['count', 'mean']})
avg_score_per_category.columns = ['product_category_name', 'count', 'mean']

#Filter to show categories with more than 50 reviews
avg_score_per_category = avg_score_per_category[avg_score_per_category['count'] > 50]
avg_score_per_category = avg_score_per_category.sort_values(by = 'mean', ascending = False)[:15]

#Set the plot
fig = plt.figure(figsize=(14, 8))
ax = sns.barplot(x = "mean", y = "product_category_name", data = avg_score_per_category)
ax.set_title('Categories Review Score')

st.pyplot(fig)

st.markdown("Les produits les mieux notés sont les livres d'intérêt général et les livres importés.")

st.header("Most Frequent Payment Type & Price Distribution")
#Create dataset methodPurchase using itemsData and paymentData datasets
methodPurchase = pd.merge(olist_order_items, olist_order_payments, on = 'order_id')
#Get the logarithmic price
methodPurchase['price_log'] = np.log(methodPurchase['price'] + 1.5)

#Get length of dataset
total = len(methodPurchase)

#Initialize plot and set main title
fig = plt.figure(figsize = (14,6))
plt.suptitle('Payment Type Distributions', fontsize = 22)

#Plot payment type count distribution
plt.subplot(121)
g = sns.countplot(x = 'payment_type', data = methodPurchase[methodPurchase['payment_type'] != 'not_defined'])
g.set_title("Payment Type Count Distribution", fontsize = 20)
g.set_xlabel("Payment Type Name", fontsize = 17)
g.set_ylabel("Count", fontsize = 17)

#Set plot and label format
sizes = []
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x() + p.get_width()/2.,
            height + 3,
            '{:1.1f}%'.format(height/total*100),
            ha = "center", fontsize = 14) 
    
g.set_ylim(0, max(sizes) * 1.1)

#Plot payment type by price distributions
plt.subplot(122)
g = sns.boxplot(x = 'payment_type', y = 'price_log',
                data = methodPurchase[methodPurchase['payment_type'] != 'not_defined'])

#Set plot labels
g.set_title("Payment Type by Price Distributions", fontsize = 20)
g.set_xlabel("Payment Type Name", fontsize = 17)
g.set_ylabel("Price(Log)", fontsize = 17)

plt.subplots_adjust(hspace = 0.5, top = 0.8)

st.pyplot(fig)

st.markdown("""Plus de 73,8 % des ventes sont réglées par carte de crédit. Le deuxième type de paiement le plus courant est le "boleto" (bordereau bancaire), avec près de 19,4 %. Le troisième type de paiement le plus courant est le bon d'achat avec 5,3 %.""")

st.header("Online sales growth over time")

#Group data to look at evolution (use df_orders_filt df created before that looks at data between 2017 and 2018 only)
df_month_aggreg = df_orders_filt.groupby(by = ['order_purchase_year', 'order_purchase_year_month'], as_index = False)

#Create dictionary
df_month_aggreg = df_month_aggreg.agg({
    'order_id': 'count',
    'price': 'sum'
})

st.markdown("**Plot the evolution of sales between 2017 and 2018**")
df_orders_filt_order = df_orders_filt['order_purchase_year_month'].value_counts().sort_index()

#Initialize the plot
fig = plt.figure(constrained_layout = True, figsize = (15, 12))

#Define axes
gs = GridSpec(2, 3, figure = fig)
ax1 = fig.add_subplot(gs[0, :])

#Plot the evolution of total orders and tnp.sortotal sales on e-commerce
sns.lineplot(x = 'order_purchase_year_month', y = 'price', ax = ax1, data = df_month_aggreg, linewidth = 2, 
             color = 'darkslateblue', marker = 'o', label = 'Total Amount')
ax1_twx = ax1.twinx()

#sort the array of unique to plot the countplot with the good order
large_to_small_2 = np.sort(df_orders_filt['order_purchase_year_month'].unique())

sns.countplot(df_orders_filt, x = 'order_purchase_year_month', ax = ax1_twx, order = large_to_small_2, palette = 'YlGnBu_r')
ax1_twx.set_yticks(np.arange(0, 20000, 2500))

#Customize the plot
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
for x, y in df_month_aggreg.price.items():
    ax1.annotate(str(round(y/1000, 1))+'K', xy = (x, y), textcoords = 'offset points', xytext = (0, 10),
                ha = 'center', color = 'dimgrey')
ax1.annotate(f'Highest Value Sold in History \n(Black Friday)', (10, 1000000), xytext = (-120, -20), 
             textcoords = 'offset points', bbox = dict(boxstyle = "round4", fc = "w", pad = .8),
             arrowprops = dict(arrowstyle = '-|>', fc = 'w'), color = 'dimgrey', ha = 'center')
ax1.set_title('Evolution of E-commerce: Total Orders and Total Amount Sold (R$)', size = 14, color = 'dimgrey', pad = 20)

#Show plot
plt.tight_layout()
st.pyplot(fig)

st.markdown("Le graphique ci-dessus montre que les ventes augmentent généralement au fil du temps, ce qui peut indiquer que les achats en ligne gagnent en popularité au Brésil. Le pic a été enregistré en novembre 2017 (cela peut être lié au Black Friday qui se produit généralement au mois de novembre ~ à cette période, on s'attend généralement à ce que les achats en ligne augmentent de manière exponentielle car de nombreuses personnes ont tendance à acheter à cette période).")