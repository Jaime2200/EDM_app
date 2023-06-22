#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from streamlit_folium import folium_static
import folium 
from geopy.geocoders import Nominatim
from datetime import datetime


from streamlit import components

# Establecer el estilo CSS personalizado
st.markdown(
    """
    <style>
    .title {
        font-family: 'Arial', sans-serif;
        font-size: 42px;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 30px;
    }
    .text {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
        color: #444444;
        text-align: justify;
        margin-bottom: 20px;
    }
    .authors {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        color: #888888;
        text-align: center;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título
st.markdown('<h1 class="title">Interactive Data Analysis for Air Quality Assessment</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="title">A Case Study of Valencia City</h2>', unsafe_allow_html=True)

# Autores
st.markdown('<p class="authors">Authors: Iván Arcos, Jaime Pérez, Pablo Llobregat</p>', unsafe_allow_html=True)

# Texto
st.markdown('<p class="text">User-friendly application designed for the comprehensive examination of '
            'air quality variables over time. Leveraging a series of sophisticated analytical tools '
            'such as time-series analysis, correlation matrices, Principal Component Analysis (PCA), '
            'Random Forest models, and LSTM neural networks, the application allows users to personalize, '
            'explore, and forecast air pollution scenarios.</p>', unsafe_allow_html=True)

valencia_coords = [39.4699, -0.3763]

m = folium.Map(location=valencia_coords, zoom_start=13)

# Agregar marcador de Valencia
marker = folium.Marker(valencia_coords, popup="Valencia, España")
marker.add_to(m)

data = pd.read_csv("rvvcca.csv", sep= ";")

def month(x):
    return x.month

def year(x):
    return x.year

def eliminar_columnas_nulas(df, porcentaje_limite):
    num_filas = len(df)
    columnas_a_eliminar = []
    
    for columna in df.columns:
        porcentaje_nulos = df[columna].isnull().sum() / num_filas
        if porcentaje_nulos > porcentaje_limite:
            columnas_a_eliminar.append(columna)
    
    df_filtrado = df.drop(columns=columnas_a_eliminar)
    return df_filtrado

df = eliminar_columnas_nulas(data, 0.7)

import calendar
month_mapping = {i: calendar.month_abbr[i] for i in range(1, 13)}

df['Fecha'] = pd.to_datetime(df['Fecha'])
df['Year'] = df['Fecha'].apply(year)
df['Mes'] = df['Fecha'].apply(month).map(month_mapping)

meses_mapeo = pd.get_dummies(df['Mes'])

from shapely.geometry import Polygon

dic_barrios = {}

import json

# Ruta al archivo GeoJSON
ruta_archivo = 'distritos.json'

# Cargar el archivo GeoJSON
with open(ruta_archivo) as archivo:
    data = json.load(archivo)
    
#for i in range(len(data['features'])):
#    print(data['features'][i]['properties']['nombre'])

for i in range(len(data['features'])):
    coords = data['features'][i]['geometry']['coordinates'][0]
    polygon = Polygon(coords)
    centroid = polygon.centroid
    dic_barrios[data['features'][i]['properties']['nombre']] = (centroid.x, centroid.y)
    
    
df_estaciones_coords = {'Pista Silla': (-0.40920905435347177, 39.36362774237338),
                       'Viveros': (-0.3663977327015507, 39.48054769066214),
                        'Politecnico': (-0.3371319774654, 39.47944194789279),
                       'Avda. Francia': (-0.3446352976820953, 39.45882881227946),
                       'Moli del Sol': (-0.41723225789592067, 39.48081154364131),
                       'Bulevard Sud': (-0.376228642501315, 39.44515356154037),
                       'Valencia Centro': (-0.3781308986256013, 39.47042599316491),
                       'Conselleria Meteo': (-0.3628269979941923, 39.47431640998867),
                       'Nazaret Meteo': (-0.33309646099915524, 39.45226819150639),
                       'Puerto Moll Trans. Ponent': (-0.32620564262865215, 39.45675190826043),
                       'Puerto llit antic Turia': (-0.3387015154919772, 39.45409149210057),
                       'Valencia Olivereta': (-0.40073611668449727, 39.471373728459405),
                       'Puerto Valencia': (-0.31549250359203407, 39.44932507441851)}


import math
barrios_estaciones = {}

for estacion, estacion_coords in df_estaciones_coords.items():
    min_distance = float('inf')
    nearest_station = None
    for barrio, barrio_coords in dic_barrios.items():
        distance = math.sqrt((barrio_coords[0] - estacion_coords[0])**2 + (barrio_coords[1] - estacion_coords[1])**2)
        if distance < min_distance:
            min_distance = distance
            nearest_station = barrio
    barrios_estaciones[estacion] = nearest_station
print(barrios_estaciones)
barrios_sin_estacion = {}
for barrio, barrio_coords in dic_barrios.items():
    if barrio not in barrios_estaciones.values():
        min_distance = float('inf')
        nearest_station = None
        for estacion, estacion_coords in df_estaciones_coords.items():
            distance = math.sqrt((barrio_coords[0] - estacion_coords[0])**2 + (barrio_coords[1] - estacion_coords[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_station = estacion
        barrios_sin_estacion[barrio] = nearest_station
print(barrios_sin_estacion)



df['nombre'] = df['Estacion'].map(barrios_estaciones)
estaciones = df['Estacion'].unique()
selected_estaciones = st.multiselect("Select one or more stations", estaciones, default=[estaciones[6], estaciones[2]])
filtered_data = df[df['Estacion'].isin(selected_estaciones)]

variables = df.columns  # Variables disponibles en los datos
selected_variables = st.multiselect("Select the variables to display", variables[5:], default=["PM10", "NO2"])
filtered_data = filtered_data[['Fecha'] + selected_variables + ['Estacion', 'nombre', 'Dia de la semana', 'Mes', 'Year']]
filtered_data["Fecha"] = pd.to_datetime(filtered_data["Fecha"])


    

# Obtener la frecuencia seleccionada (por semanas o meses)
frequency = st.selectbox("Select frequency", ["Mes", "Semana", "Día"])

# Filtrar los datos por fecha seleccionada
start_date = st.date_input("Select start date", datetime(2018, 1, 1).date(), max_value = datetime.now())
end_date = st.date_input("Select end date", datetime.now(), min_value = datetime(2004, 1, 1).date())
filtered_data = filtered_data[(filtered_data['Fecha'].dt.date >= start_date) & (filtered_data['Fecha'].dt.date <= end_date)]


print(filtered_data)

# Agrupar los datos por estación, día, mes y año y calcular la media de las variables seleccionadas
if frequency == "Semana":
    grouped_data = filtered_data.groupby(['Estacion', pd.Grouper(key='Fecha', freq='W-MON')]).mean().reset_index()
elif frequency == "Mes":
    grouped_data = filtered_data.groupby(['Estacion', pd.Grouper(key='Fecha', freq='M')]).mean().reset_index()
else:
    grouped_data = filtered_data.groupby(['Estacion', pd.Grouper(key='Fecha', freq='D')]).mean().reset_index() # Sin agrupación, mantén los datos diarios

    
# Crear un mapa centrado en Valencia
m = folium.Map(location=[39.46975, -0.37739], zoom_start=12)

# Cargar los datos geoespaciales con información
# Supongamos que tienes un DataFrame llamado "data" con columnas "Area" y "Valor"
# data = pd.read_csv('datos.csv')
grouped_data_mapa = filtered_data.groupby(['nombre']).mean().reset_index()


for barrio_not, estacion in barrios_sin_estacion.items():
    if barrios_estaciones[barrios_sin_estacion[barrio_not]] in list(grouped_data_mapa['nombre']):
        fila_copiar = grouped_data_mapa.loc[grouped_data_mapa['nombre'] == barrios_estaciones[barrios_sin_estacion[barrio_not]]]
        dfc = pd.DataFrame([barrio_not] + list(fila_copiar.iloc[0][1:]), index = fila_copiar.columns).T
        grouped_data_mapa  = pd.concat([grouped_data_mapa, dfc], axis = 0)

# Agregar coropletas al mapa
folium.Choropleth(
    geo_data='distritos.json',  # Archivo GeoJSON con los datos geoespaciales de Valencia
    data=grouped_data_mapa,
    columns=['nombre', selected_variables[0]],  # Columnas con la información a mapear
    key_on='feature.properties.nombre',  # Campo que coincide entre el GeoJSON y los datos
    fill_color='YlOrRd',  # Esquema de colores (puedes elegir otro)
    fill_opacity=0.4,
    line_opacity=0.8 # Título de la leyenda
).add_to(m)

# Add markers to the map
for est, coords in df_estaciones_coords.items():
    if est in selected_estaciones:
        marker = folium.Marker(
            [coords[1], coords[0]],
             popup=f"{est}\n{selected_variables[0]}: {round(grouped_data_mapa.loc[grouped_data_mapa['nombre'] == barrios_estaciones[est], selected_variables[0]].values[0],3)}",
        )
        marker.add_to(m)  
        
st.header(f"Interactive map of Valencia for {selected_variables[0]}")
st.markdown("""The interactive map displays neighborhoods whose nearest station is one of the selected stations. Each station is also represented by an interactive icon on the map, indicating the average value of the first selected variable for the chosen date range.""")
    

# Mostrar el mapa en Streamlit
folium_static(m)

# Graficar distribuciones
if selected_variables:
    num_bins = st.slider('Select the number of bins:', min_value=5, max_value=50, value=10)
    st.header("Graphs of distributions by variable")
    fig, ax = plt.subplots(1, len(selected_variables), figsize=(10, 5))
    for i, variable in enumerate(selected_variables):
        try:
            sns.histplot(data = grouped_data, x = variable, bins = num_bins, hue = 'Estacion', kde = "True", alpha = 0.4, ax = ax[i])
        except:
            sns.histplot(data = grouped_data, x = variable, bins = num_bins, hue = 'Estacion', kde = "True", alpha = 0.4, ax = ax)
    st.pyplot(fig)

st.header("Correlation graphs by station")
# Graficar correlaciones
if len(selected_variables) > 1:
    fig, ax = plt.subplots(1, len(selected_estaciones), figsize=(15, 7))
    for i, station in enumerate(selected_estaciones):
        corr = grouped_data.loc[grouped_data['Estacion'] == station, selected_variables].corr()
        try:
            sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, ax= ax[i], cbar=False)
            ax[i].set_title(f'Correlation matrix of {station}')
        except:
            sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, ax= ax, cbar=False)
            ax.set_title(f'Correlation matrix of {station}')
    st.pyplot(fig)
else:
    st.write('Select at least 2 variables to plot the correlation matrix.')

# Create the combined line chart
plt.figure(figsize=(10, 6))

st.header("Line graphs by variable and station")

for variable in selected_variables:
    for station in selected_estaciones:
        station_data = grouped_data[grouped_data['Estacion'] == station]
        chart_data = station_data.set_index('Fecha')
        chart_data_sorted = chart_data.sort_index()  # Sort the data by index
        plt.plot(chart_data_sorted.index, chart_data_sorted[variable], label=f"{station} - {variable}")

plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Line graph for selected variables")
plt.legend()
st.pyplot(plt)

st.header("Violin chart by station")

grouped_data = filtered_data.copy()  # Sin agrupación, mantén los datos diarios
plt.figure(figsize=(15,10))
meses_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
              'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
grouped_data['Mes_Num'] = grouped_data['Mes'].map(meses_dict)
grouped_data_sorted = grouped_data.sort_values('Mes_Num')
grouped_data = grouped_data_sorted.drop('Mes_Num', axis=1)
print(grouped_data['Mes'].unique())
sns.violinplot(x = grouped_data['Mes'], y = grouped_data[selected_variables[0]], hue = grouped_data["Estacion"])
ax2 =sns.stripplot(x = grouped_data['Mes'], y = grouped_data[selected_variables[0]], color = "gray", size=3, hue = grouped_data["Estacion"])
plt.setp(ax2.collections, alpha=0.3)
st.pyplot(plt)

st.header("Principal Component Analysis")

st.markdown("""In this analysis we can observe the different relationships between features of our data,
            thus looking at the behavior of the gases of interest with respect to different environmental conditions.""")

from sklearn.impute import SimpleImputer
si = SimpleImputer()
filtered_data = df[df['Estacion'].isin(selected_estaciones)]
filtered_data = filtered_data[(filtered_data['Fecha'].dt.date >= start_date) & (filtered_data['Fecha'].dt.date <= end_date)]
# Agrupar los datos por estación, día, mes y año y calcular la media de las variables seleccionadas
if frequency == "Semana":
    grouped_data = filtered_data.groupby(['Estacion', pd.Grouper(key='Fecha', freq='W-MON')]).mean().reset_index()
elif frequency == "Mes":
    grouped_data = filtered_data.groupby(['Estacion', pd.Grouper(key='Fecha', freq='M')]).mean().reset_index()
else:
    grouped_data = filtered_data.groupby(['Estacion', pd.Grouper(key='Fecha', freq='D')]).mean().reset_index()   # Sin agrupación, mantén los datos diarios
filtered_data = grouped_data
# pd.concat([df.iloc[:,np.r_[5:27, 34]], meses_mapeo], axis = 1)
nombres = filtered_data.iloc[:,np.r_[5:18]].columns
data_pca = si.fit_transform(filtered_data.iloc[:,np.r_[5:18]])

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA()
sc = StandardScaler()

data_pca_esc = sc.fit_transform(data_pca)
scores = pca.fit_transform(data_pca_esc)
# Obtener las dos primeras componentes principales
component1 = scores[:, 0]
component2 = scores[:, 1]

# Obtener la varianza explicada por las dos primeras componentes
explained_variance = pca.explained_variance_ratio_[:2]

# Graficar el biplot
plt.figure(figsize=(12, 8))
sns.scatterplot(x=component1, y=component2, s=100, hue=filtered_data['Estacion'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Explained Variance: {:.2f}%'.format(sum(explained_variance) * 100))
plt.show()

cont = 0
for p1,p2 in zip(pca.components_[0], pca.components_[1]):
    if abs(p1) + abs(p2) >= 0:
        plt.arrow(0,0,10*p1,10*p2, head_width = 0.2)
        plt.text(10*p1,10*p2, s = nombres[cont], weight = "bold", fontsize = 12)
        #plt.text(22*p1,22*p2, s = f"({round(standardized_loadings[0][cont],2)}, {round(standardized_loadings[1][cont], 2)})", color = "red", weight = "bold")
    cont += 1
st.pyplot(plt)

st.header("Predictive model with LSTM network")

st.markdown("""To obtain a prediction of the first selected variable for each station, 
          we generated a neural network composed of an LSTM network with 64 neurons and a dense layer to fit the shape of our data.
          our data. """)


from sklearn.ensemble import RandomForestRegressor
si = SimpleImputer()
rfr = RandomForestRegressor(n_estimators = 100)
plt.figure(figsize=(12,7))
dic_importances = {} #
for estacion in selected_estaciones:
    
    limpio = df.loc[df['Estacion'] == estacion].iloc[:,np.r_[5:19]].dropna(axis=1, how='all')
    clean = pd.DataFrame(si.fit_transform(limpio), columns = limpio.columns)
    X,y = clean.drop([selected_variables[0]], axis = 1), clean[selected_variables[0]]
    rfr.fit(X,y)
    importances = list(zip(rfr.feature_importances_, X.columns))
    for imp,var in importances:
        if var in dic_importances:
            dic_importances[var].append(imp)
        else:
            dic_importances[var] = [imp]
            
    for key, value in dic_importances.items():
        if len(value) == 0:
            dic_importances[key] = [0] * max([len(i) for i in dic_importances.values()])
        elif len(value) < max([len(i) for i in dic_importances.values()]):
            dic_importances[key] = value + [0] * (max([len(i) for i in dic_importances.values()])-len(value))
        else:
            pass
        
    for i,variable in enumerate(selected_variables[1:2]): # Solo la siguiente variable
        preds = []
        minimo = X[variable].min()
        maximo = X[variable].max()
        for valor in np.arange(minimo, maximo,(maximo-minimo)/100):
            #print(valor)
            X.loc[:,variable] = [valor] * len(X[variable])
            preds_all = rfr.predict(X)
            preds.append(preds_all.mean())
        #st.write(f"{i},{variable}")
        plt.plot(np.arange(minimo, maximo,(maximo-minimo)/100), preds, label = estacion)

plt.legend()
plt.title(f"PDP, response {selected_variables[0]}")
plt.xlabel(selected_variables[1])
st.pyplot(plt)

plt.figure(figsize=(12,7))
df_importances = pd.DataFrame(dic_importances)
#st.write(df_importances)
df_importances = pd.melt(df_importances)
df_importances['estacion'] = selected_estaciones * len(dic_importances)
sns.barplot(x = df_importances['variable'],  y= df_importances['value'], hue = df_importances['estacion'])
plt.xticks(rotation=45)
st.pyplot(plt)

    
st.header("Station forecasts")
    
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
preds_5_steps = {}
def prepare_data_for_lstm(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    X = np.array(X)
    y = np.array(y)
    return X, y

for estacion in grouped_data['Estacion'].unique():
    agrupado_estacion = grouped_data.loc[grouped_data['Estacion'] == estacion].iloc[:,np.r_[5:18]]
    fecha =  grouped_data.loc[grouped_data['Estacion'] == estacion].loc[:,'Fecha']
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    agrupado_estacion=agrupado_estacion.dropna(axis=1, how='all')
    cols = agrupado_estacion.columns
    filtered_data = si.fit_transform(agrupado_estacion)
    scaled_data = scaler.fit_transform(filtered_data)
    n_steps = 7  # Number of time steps to consider for prediction
    X, y = prepare_data_for_lstm(scaled_data, n_steps)

    # Split the data into training and testing sets (80% training, 20% testing)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(n_steps, 
                                                       X_train.shape[2])))
    model.add(Dense(X_train.shape[2]))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    # Evaluate the model on the testing set
    loss = model.evaluate(X_test, y_test)
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    # Evaluate the model on the testing set
    loss = model.evaluate(X_test, y_test)
    # Prepare the last n_steps of data for prediction
    last_data = scaled_data[-n_steps:]
    last_data = last_data.reshape((1, n_steps, X_train.shape[2]))

    # Make predictions using the LSTM model
    predictions = model.predict(last_data)

    # Inverse transform the predictions to the original scale
    predicted_values = scaler.inverse_transform(predictions)
    
    preds_5_steps[estacion] = ([predicted_values],  cols, fecha)
   
    for variable in selected_variables:
        if variable in preds_5_steps[estacion][1]:
            plt.figure(figsize = (16,7))
            plt.plot(preds_5_steps[estacion][2][-20:], agrupado_estacion[variable][-20:])
            plt.scatter( preds_5_steps[estacion][2][-1:].values[0] + np.timedelta64(1, 'D'), preds_5_steps[estacion][0][0][0][list(preds_5_steps[estacion][1]).index(variable)], color = "red", s = 200)
            plt.xticks(preds_5_steps[estacion][2][-20:], rotation = "45")
            plt.ylabel(variable)
            plt.title(estacion)
            st.pyplot(plt)


# In[ ]:




