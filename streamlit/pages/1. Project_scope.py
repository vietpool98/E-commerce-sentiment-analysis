import os
import pandas as pd
import numpy as np
import streamlit as st
import io


import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Description du projet",
    page_icon="📄",
)

st.sidebar.markdown("# 📄 Description du projet")

st.title("Sentiment analysis and EDA of a Brezilian E-commerce")

st.header("Part 1 : Description du projet")

st.markdown(
    """
    L'objectif de mon projet est de proposer une vision analytique du E-commerce au Brésil. Pour ce faire, nous allons d'abord procéder à une analyse exploratoire des données (EDA) à l'aide d'outils graphiques afin de créer des graphiques explicatifs pour mieux comprendre ce qui se cache derrière les achats en ligne des Brésiliens. J'analyserai les tendances des achats en ligne au Brésil entre 2016 et 2018 à l'aide d'Olist (un site de E-commerce brésilien comme Amazon). Je commencerai d'abord par effectuer une AED approfondie sur les 8 ensembles de données dont je dispose afin de comprendre les variables initiales à portée de main (croissance des ventes, produits les plus achetés, villes avec le plus d'achats en ligne, etc.) Enfin, je travaillerai à la construction d'un modèle d'analyse sentimentale utilisant la NLP qui prédit si le client est satisfait ou non de son achat en ligne sur la base des avis et des notes qu'il conserve après avoir effectué l'achat et reçu le produit.

L'objectif est de comprendre les données et tracer des graphiques utiles pour clarifier les concepts et obtenir des insights et, à la fin, nous ferons un code étape par étape sur la préparation du texte et la classification des sentiments en utilisant les avis laissés par les clients sur les plates-formes en ligne.
"""
)