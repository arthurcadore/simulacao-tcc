import re
import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import argparse
import urllib.error

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

INPUT_DATA = "data/sinda_pcds.csv"
CSV_PATH = "data/geoencoded.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"

def extract_df(path):
    r"""
    Extrai as colunas de interesse do arquivo CSV

    Args:
        path (str): Caminho do arquivo CSV

    Returns:
        list: Lista de pcds e cidades
    """
    df = pd.read_csv(
        path,
        header=None,
        names=['pcd_id', 'uf', 'city', 'extra'],
        engine='python',
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines='skip'
    )
    df = df.dropna(subset=['pcd_id', 'uf', 'city'])
    return [
        (str(row['pcd_id']), f"{str(row['city']).strip()}, {str(row['uf']).strip()}")
        for _, row in df.iterrows()
    ]

def geoencode_location(pcd_city_list):
    r"""
    Geocodifica as cidades

    Args:
        pcd_city_list (list): Lista de pcds e cidades

    Returns:
        list: Lista de resultados de geocodificação
    """
    geolocator = Nominatim(user_agent="pcd_mapper")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['pcd_id', 'city', 'lat', 'lon']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for pcd_id, city in tqdm(pcd_city_list, desc="Geocodificando cidades"):
            try:
                location = geocode(city + ", Brasil")
                if location:
                    row = {
                        'pcd_id': pcd_id,
                        'city': city,
                        'lat': location.latitude,
                        'lon': location.longitude
                    }
                    writer.writerow(row)
                    csvfile.flush()
            except urllib.error.URLError as e:
                print(f"\nErro de rede ao geocodificar '{city}': {e.reason}")
                sys.exit(1)
            except PermissionError as e:
                print(f"\nErro de permissão ao geocodificar '{city}': {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Erro ao geocodificar '{city}': {e}")

if __name__ == "__main__":
    pcd_city_list = extract_df(INPUT_DATA)
    geoencode_location(pcd_city_list)

