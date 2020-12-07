"""This library contain functions to scrap some data"""
import bs4
import pandas as pd
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go
import numpy as np
import folium
from geopy.geocoders import Nominatim
import json
from numpy import NaN
import requests
from requests import get
from bs4 import BeautifulSoup as BS
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering 
import scipy.cluster.hierarchy as sch
from branca.element import Template, MacroElement
from sklearn.metrics import silhouette_score


#################### New York scraping

def scraping_NY(full_df_city):
    """Main function"""
    url = "https://www.health.ny.gov/statistics/cancer/registry/appendix/neighborhoods.htm"
    titres, lignes = import_table(url)
    df = build_df(lignes, titres)
    df = replace_borough(df)
    return set_neighborhood(full_df_city, clear_zip_NY(df))

def import_table(url):
    """Get the 'raw' data from the website, return title and row not cleaned."""
    page_brute = get(url=url)
    texte = page_brute.text
    soupe = BS(texte, features="lxml")
    balises_tables = soupe.find_all(name="table")
    table = balises_tables[0]
    titres, *lignes = table.find_all(name="tr") 
    return titres, lignes 

def gestion_ligne(balise_tr):
    """Clean the lignes"""
    if len(balise_tr)==7:
        borough, neighborhood, ZIP_Codes = balise_tr.find_all(name="td")
        return [
            borough.text.strip(),
            neighborhood.text.strip(),
            ZIP_Codes.text.strip()
        ]
    else:
        neighborhood, ZIP_Codes = balise_tr.find_all(name="td")
        return [
            "",
            neighborhood.text.strip(),
            ZIP_Codes.text.strip()
        ]
    
    
    
def build_df(lignes, titres):
    """Build the df"""
    table_liste = [gestion_ligne(ma_ligne) for ma_ligne in lignes]
    titres = [titre.text for titre in titres.find_all("th")]
    return pd.DataFrame(table_liste, columns = titres)




########################## SF scraping

def scraping_SF(full_df_city):
    """Main for SF scraping"""
    url = "http://www.healthysf.org/bdi/outcomes/zipmap.htm"
    titres, lignes = import_table_sf(url)
    titres_clean = clean_title_sf(titres)
    table_liste = [gestion_ligne_SF(ma_ligne) for ma_ligne in lignes]
    del(table_liste[-1])
    df = pd.concat([pd.DataFrame(table_liste, columns=titres_clean),pd.DataFrame([["94101", "Hayes Valley/Tenderloin/North of Market"], ["94105", "Financial District South"]], columns = titres_clean) ]).reset_index(drop=True)
    return add_neighborhood(full_df_city, df)

def import_table_sf(url):
    """Get the 'raw' data from the website, return title and row not cleaned."""
    page_brute = get(url=url)
    texte = page_brute.text
    soupe = BS(texte, features = "lxml")
    balises_tables = soupe.find_all(name="table")
    table = balises_tables[3]
    titres, *lignes = table.find_all(name="tr") 
    return titres, lignes 

def clean_title_sf(titles):
    """Clean the titles list"""
    titles_clean=[]
    titles_clean.append(titles.find_all(name="td")[0].text.replace(" ","").replace("\n",""))
    titles_clean.append(titles.find_all(name="td")[1].text.replace(" ","").replace("\n",""))
    return titles_clean

def gestion_ligne_SF(balise_tr):
    """Find and clear zip code, neighborhood and population"""
    ZIP_Codes, neighborhood, population = balise_tr.find_all(name="td")
    return [
        ZIP_Codes.text.strip(),
        neighborhood.text.strip().replace("              ","").replace("\n","")
    ]

def replace_borough(df):
    """Remplace les espaces vides dans la colonne 'Borough' par le dernier nom de quartier rencontré"""
    nom_quartier=""
    j=0
    for i in df["Borough"]:
        if i != "":
            nom_quartier = i

        else:
            df.iloc[j,0]=nom_quartier
        j+=1
    return df

def clear_zip_NY(df):
    """Clear zip codes"""
    new_df= df["ZIP Codes"].replace(" ", "").str.split(",", expand = True) 
    df = df.join(new_df)
    del(df["ZIP Codes"])
    return df

def add_neighborhood(data, df):
    """Check if zip code is available, and add neighborhood name to column"""
    row_final_df = 0
    for zip_code in data["zipCode"]:
        row = find_row(df, zip_code)
        if row is not None:
            data.iloc[row_final_df, -1]=df.iloc[row,1]
        else:
            data.iloc[row_final_df, -1] = NaN
        row_final_df+=1
    return data


###### Fonctions communes


def find_row(df_code, postal_code):
    """Return the row of the postal code to find the neighborhood"""
    for row in range(len(df_code)):
        if check_code(df_code, row, postal_code) is True :
            return row
        
def check_code(df_code, line, code):
    """Check if the zip code is present on a row"""
    for i in df_code.iloc[line]:
        if i is not None:
            if i.strip() == code:
                return True
    return False

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    """Find all nerby venues with restaurant location"""
    CLIENT_ID = 
    CLIENT_SECRET = 
    VERSION = '20180605' # Foursquare API version
    LIMIT = 100 # A default Foursquare API limit value
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):

            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

def set_neighborhood(data_model, df_code):
    """Set borough and neighborhood name by zip code"""
    data = data_model
    data["Borough"] = ""
    data["Neighborhood"] = ""
    row_final_df = 0
    for zip_code in data["zipCode"]:
        row = find_row(df_code, zip_code)
        if row is not None:
            data.iloc[row_final_df, -2]=df_code.iloc[row,0]
            data.iloc[row_final_df, -1]=df_code.iloc[row,1]
        else:
            data.iloc[row_final_df, -2]=NaN
            data.iloc[row_final_df, -1]=NaN
        row_final_df+=1
    return data

