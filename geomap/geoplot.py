import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import scienceplots

from matplotlib.patches import Patch
from shapely.geometry import Point

# Configurações globais de visualização
plt.style.use('science')
plt.rc('font', size=16)
plt.rc('axes', titlesize=22)     
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=16)    
plt.rc('ytick', labelsize=16)    
plt.rc('legend', fontsize=12)     
plt.rc('figure', titlesize=22)   

# Constantes de entrada e saída
INPUT_DATA = "data/geoencoded.csv"
OUTPUT_DATA = "../out/geoplot.pdf"
GEOJSON_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"

def import_df(input_data):
    r"""
    Importa o dataframe
    
    Args:
        input_data (str): Caminho do arquivo CSV
    
    Returns:
        pd.DataFrame: DataFrame importado
    """
    df = pd.read_csv(input_data, header=None, names=["id", "local", "lat", "lon"])
    return df


def create_geodf(df):
    r"""
    Cria um GeoDataFrame a partir do DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame com as coordenadas
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame criado
    """
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf_pcds = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)
    return gdf_pcds

def process_uf():
    r"""
    Processa os dados de estados
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com os dados de estados
    """
    estados = gpd.read_file(GEOJSON_URL)
    estados = estados[["name", "geometry"]].to_crs(epsg=3857)
    return estados

def process_pcd(gdf_pcds, estados):
    r"""
    Realiza o join entre PCDs e estados
    
    Args:
        gdf_pcds (gpd.GeoDataFrame): GeoDataFrame com os dados de PCDs
        estados (gpd.GeoDataFrame): GeoDataFrame com os dados de estados
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com os dados de PCDs e estados
    """
    pcd_com_estado = gpd.sjoin(gdf_pcds, estados, how="inner", predicate="within")
    pcd_por_estado = pcd_com_estado.groupby("name").size().reset_index()
    pcd_por_estado.columns = ["name", "pcd_count"]
    estados = estados.merge(pcd_por_estado, on="name", how="left").fillna(0)
    return estados

def steps(step):
    r"""
    Cria categorias para os dados
    
    Args:
        step (int): Valor a ser categorizado
    
    Returns:
        str: Categoria criada
    """
    if step == 0:
        return "0"
    elif step <= 10:
        return "1-10"
    elif step <= 20:
        return "11-20"
    elif step <= 30:
        return "21-30"
    elif step <= 50:
        return "31-50"
    else:
        return "50+"

def collor_mapping(uf):
    r"""
    Mapeia as cores de acordo com a categoria
    
    Args:
        uf (gpd.GeoDataFrame): GeoDataFrame com os dados de estados
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com as cores mapeadas
    """
    collors = {
        "0": "#f7fbff",
        "1-10": "#deebf7",   
        "11-20": "#c6dbef",  
        "21-30": "#9ecae1",  
        "31-50": "#6baed6",  
        "50+": "#3182bd"     
    }
    uf['cor'] = uf['categoria'].apply(lambda x: collors[x])
    return uf, collors

def plot_uf(ax, uf):
    r"""
    Plota os estados no gráfico
    
    Args:
        ax (matplotlib.axes.Axes): Eixo do gráfico
        uf (gpd.GeoDataFrame): GeoDataFrame com os dados de estados
    """
    uf.plot(
        ax=ax,
        color=uf['cor'],
        edgecolor="#1e2129",
        linewidth=0.8,
        zorder=1
    )

def plot_pcd(ax, gdf_pcds):
    r"""
    Plota os PCDs no gráfico
    
    Args:
        ax (matplotlib.axes.Axes): Eixo do gráfico
        gdf_pcds (gpd.GeoDataFrame): GeoDataFrame com os dados de PCDs
    """
    gdf_pcds.plot(
        ax=ax,
        color='red',
        markersize=8,
        alpha=0.7,
        label="PCDs",
        zorder=2
    )

def add_legend(ax, collors):
    r"""
    Adiciona a legenda ao gráfico
    
    Args:
        ax (matplotlib.axes.Axes): Eixo do gráfico
        collors (dict): Dicionário com as cores de cada categoria
    """
    legend_elements = [
        Patch(facecolor=collors["0"], edgecolor='black', label='0 PCDs'),
        Patch(facecolor=collors["1-10"], edgecolor='black', label='1-10 PCDs'),
        Patch(facecolor=collors["11-20"], edgecolor='black', label='11-20 PCDs'),
        Patch(facecolor=collors["21-30"], edgecolor='black', label='21-30 PCDs'),
        Patch(facecolor=collors["31-50"], edgecolor='black', label='31-50 PCDs'),
        Patch(facecolor=collors["50+"], edgecolor='black', label='50+ PCDs'),
    ]
    ax.legend(
        handles=legend_elements,
        title='Quantidade de PCDs por Estado',
        title_fontsize=20,
        loc='upper right',
        frameon=True,
        edgecolor='black',
        facecolor='white',
        fontsize=16,
        fancybox=True
    )

    frame = ax.get_legend().get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(0)

def gerar_mapa():
    r"""
    Função principal para gerar o mapa
    """

    # Importa os dados
    df = import_df(INPUT_DATA)
    
    # Cria e processa os GeoDataFrames
    gdf_pcds = create_geodf(df)
    data = process_uf()
    data = process_pcd(gdf_pcds, data)
    data['categoria'] = data['pcd_count'].apply(steps)
    data, collors = collor_mapping(data)
    
    # Cria o gráfico
    fig, ax = plt.subplots(figsize=(16, 16))
    plot_uf(ax, data)
    plot_pcd(ax, gdf_pcds)
    
    # Define os limites do gráfico
    map = data.geometry.union_all()
    x_min, y_min, x_max, y_max = map.bounds
    margin = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - x_range * margin, x_max + x_range * margin)
    ax.set_ylim(y_min - y_range * margin, y_max + y_range * margin)

    # Adiciona labels e legenda
    add_legend(ax, collors)
    ax.axis("off")
    plt.tight_layout()

    # Salva o gráfico
    plt.savefig(OUTPUT_DATA, dpi=1500, bbox_inches='tight', transparent=True)

if __name__ == "__main__":
    gerar_mapa()
