import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Data Exploration", page_icon="📈")

st.sidebar.markdown("# 📈 Data Exploration")

st.title("Sentiment analysis and EDA of a Brezilian E-commerce")
st.header("Part 2 : Data Exploration")

st.markdown(
    """
    Pour cette tâche, nous disposons de différentes sources de données, chacune décrivant un sujet spécifique lié aux ventes en ligne. Les fichiers sont les suivants:
    -  olist_customers_dataset.csv
    -  olist_geolocation_dataset.csv
    -  olist_orders_dataset.csv
    -  olist_order_items_dataset.csv
    -  olist_order_payments_dataset.csv
    -  olist_order_reviews_dataset.csv
    -  olist_products_dataset.csv
    -  olist_sellers_dataset.csv
    -  product_category_name_translation.csv \n
    Chacun de ces datasets correspondent à une spécificité du marché bresilien, nous allons devoir manipuler ces derniers afin d'en tirer le meilleur parti
"""
)

st.markdown("""**Overview of the Data**"""
)

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

# Collections for each dataset
datasets = [olist_customer, olist_geolocation, olist_orders, olist_order_items, olist_order_payments,
            olist_order_reviews, olist_products, olist_sellers]
names = ['olist_customer', 'olist_geolocation', 'olist_orders', 'olist_order_items', 'olist_order_payments',
         'olist_order_reviews', 'olist_products', 'olist_sellers']
# Creating a DataFrame with useful information about all datasets
data_info = pd.DataFrame({})
data_info['dataset'] = names
data_info['dataset'] = names
data_info['n_rows'] = [df.shape[0] for df in datasets]
data_info['n_cols'] = [df.shape[1] for df in datasets]
data_info['null_amount'] = [df.isnull().sum().sum() for df in datasets]
data_info['qty_null_columns'] = [len([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]
data_info['null_columns'] = [', '.join([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]

st.write(data_info.style.background_gradient())