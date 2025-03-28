# %%
import requests
import pandas as pd
import numpy as np

start_date = int(pd.to_datetime('2025-03-01').timestamp())
end_date = int(pd.to_datetime('2025-03-03').timestamp())

response = requests.get(f'https://sensornet.nl/dataserver3/event/collection/nina_events/stream?conditions%5B0%5D%5B%5D=time&conditions%5B0%5D%5B%5D=%3E%3D&conditions%5B0%5D%5B%5D={start_date}&conditions%5B1%5D%5B%5D=time&conditions%5B1%5D%5B%5D=%3C&conditions%5B1%5D%5B%5D={end_date}&conditions%5B2%5D%5B%5D=label&conditions%5B2%5D%5B%5D=in&conditions%5B2%5D%5B2%5D%5B%5D=21&conditions%5B2%5D%5B2%5D%5B%5D=32&conditions%5B2%5D%5B2%5D%5B%5D=33&conditions%5B2%5D%5B2%5D%5B%5D=34&args%5B%5D=aalsmeer&args%5B%5D=schiphol&fields%5B%5D=time&fields%5B%5D=location_short&fields%5B%5D=location_long&fields%5B%5D=duration&fields%5B%5D=SEL&fields%5B%5D=SELd&fields%5B%5D=SELe&fields%5B%5D=SELn&fields%5B%5D=SELden&fields%5B%5D=SEL_dB&fields%5B%5D=lasmax_dB&fields%5B%5D=callsign&fields%5B%5D=type&fields%5B%5D=altitude&fields%5B%5D=distance&fields%5B%5D=winddirection&fields%5B%5D=windspeed&fields%5B%5D=label&fields%5B%5D=hex_s&fields%5B%5D=registration&fields%5B%5D=icao_type&fields%5B%5D=serial&fields%5B%5D=operator&fields%5B%5D=tags')

colnames = pd.DataFrame(response.json()['metadata'])
data = pd.DataFrame(response.json()['rows'])
data.columns = colnames.headers

data['time'] = pd.to_datetime(data['time'], unit = 's')

print(data['time'].min(),data['time'].max())

data.head()


# %%
print(data.columns.tolist())


# %% [markdown]
# we gaan de relevante kolommen naar 1 dataframe brengen: 

# %%
relevant_columns = ['time', "location_long", 'altitude', 'SEL_dB', 'distance', 'callsign', 'icao_type', 'operator']
df = data[relevant_columns].copy()


# %% [markdown]
# Opschonen - Missing values verwijderen: Omdat zowel altitude als SEL_dB essentieel zijn, droppen we rijen waar deze ontbreken

# %%
df = df.dropna(subset=['altitude', 'SEL_dB'])


# %% [markdown]
# Duplicaten verwijderen

# %%
df = df.drop_duplicates()


# %% [markdown]
# Reset index (netjes voor verwerking/plots)

# %%
df = df.reset_index(drop=True)


# %% [markdown]
# laten we kijken hoe groot onze kolom nog is: 

# %%
print(df.shape)
print(df.head())


# %% [markdown]
# Verkenning van de relatie tussen hoogte en geluid met behulp van een scatterplot. 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='altitude', y='SEL_dB', alpha=0.5)

plt.title('Relatie tussen Hoogte en Geluidsniveau (SEL_dB)')
plt.xlabel('Hoogte (feet)')
plt.ylabel('Geluidsniveau (dB)')
plt.grid(True)
plt.show()


# %% [markdown]
# trendlijn toevoegen om patroon beter te zien
# 

# %%
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='altitude', y='SEL_dB', scatter_kws={'alpha':0.4}, line_kws={'color':'red'})

plt.title('Relatie tussen Hoogte en Geluidsniveau (met trendlijn)')
plt.xlabel('Hoogte (feet)')
plt.ylabel('Geluidsniveau (dB)')
plt.grid(True)
plt.show()


# %% [markdown]
# Outliers detecteren en eventueel weghalen
# 

# %%
q_low = df['SEL_dB'].quantile(0.01)
q_hi  = df['SEL_dB'].quantile(0.99)
df_filtered = df[(df['SEL_dB'] >= q_low) & (df['SEL_dB'] <= q_hi)]

# Eventueel rare hoogtes filteren (>40.000 ft?)
df_filtered = df_filtered[(df_filtered['altitude'] >= 500) & (df_filtered['altitude'] <= 40000)]


# %% [markdown]
# Extra variabele meeplotten (kleur of grootte) Bijvoorbeeld de afstand meeplotten als kleur of grootte in scatter. Dit geeft inzicht of dichterbij zijnde vliegtuigen inderdaad meer geluid maken, los van hoogte.

# %%
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_filtered['altitude'], df_filtered['SEL_dB'], 
                      c=df_filtered['distance'], cmap='coolwarm', alpha=0.6)

plt.title('Hoogte vs Geluid met Afstand als kleur')
plt.xlabel('Hoogte (feet)')
plt.ylabel('Geluidsniveau (dB)')
cbar = plt.colorbar(scatter)
cbar.set_label('Afstand tot meetpunt (m)')
plt.show()



# %%
# Toon alle unieke stadnamen in de 'location_long' kolom
uniek= df_filtered['location_long'].unique()

# Print de unieke locaties
print("Unieke steden in 'location_long':", uniek)


# %%
df_filtered.head()

# %%
# Coördinaten mapping
locaties_coords = {
    'Uiterweg': (52.26224927007567, 4.7330575976587905),
    'Aalsmeerderweg': (52.280158469807105, 4.795345755018842),
    'Blaauwstraat': (52.2649466795041, 4.7740958913400995),
    'Hornweg': (52.27213650444824, 4.7935705607232775),
    'Kudelstaartseweg': (52.24558794567681, 4.755324955329966),
    'Darwinstraat': (52.23621816215143, 4.758856663004422),
    'Copierstraat': (52.22957895035498, 4.739337496658634),
}

# Voeg latitude en longitude toe aan je dataframe
df['latitude'] = df['location_long'].map(lambda x: locaties_coords[x][0])
df['longitude'] = df['location_long'].map(lambda x: locaties_coords[x][1])


# %%
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from branca.element import MacroElement
from jinja2 import Template
import plotly.express as px


st.set_page_config(layout="wide")


# === TABS ===
tab1, tab2, tab3, tab4, = st.tabs([
    " Heatmap",
    " Analyse Geluid per Locatie (Kaart met Radius)",
    " Vliegtuigtypes & Geluid",
    " Tijdlijn Geluidsbelasting"
    
])

df = df_filtered.copy()

if 'location_short' not in df.columns:
    df_full = data
    df = df_full[['time', 'location_long', 'location_short', 'altitude', 'SEL_dB', 
                  'distance', 'callsign', 'icao_type', 'operator', 'type']].dropna()

# === FILTEREN ===
df = df[(df['altitude'] >= 100) & (df['altitude'] <= 4000)]
df = df[(df['SEL_dB'] >= 50) & (df['SEL_dB'] <= 100)]

locaties_coords = {
    'Uiterweg': (52.26224927007567, 4.7330575976587905),
    'Aalsmeerderweg': (52.280158469807105, 4.795345755018842),
    'Blaauwstraat': (52.2649466795041, 4.7740958913400995),
    'Hornweg': (52.27213650444824, 4.7935705607232775),
    'Kudelstaartseweg': (52.24558794567681, 4.755324955329966),
    'Darwinstraat': (52.23621816215143, 4.758856663004422),
    'Copierstraat': (52.22957895035498, 4.739337496658634),
}
df['latitude'] = df['location_long'].map(lambda x: locaties_coords[x][0])
df['longitude'] = df['location_long'].map(lambda x: locaties_coords[x][1])

min_hoogte = int(df['altitude'].min())
max_hoogte = int(df['altitude'].max())
hoogte_range = tab1.slider("Selecteer een hoogtebereik (meters)", min_hoogte, max_hoogte, (min_hoogte, max_hoogte))

df = df[(df['altitude'] >= hoogte_range[0]) & (df['altitude'] <= hoogte_range[1])]

# ========== HEATMAP TAB ==========
with tab1:
    m = folium.Map(location=[52.26, 4.77], zoom_start=12)
    heat_data = [[row['latitude'], row['longitude'], row['SEL_dB']] for index, row in df.iterrows()]
    HeatMap(heat_data, radius=25, max_zoom=13).add_to(m)

    # Markers met afkortingen
    unique_locations = df[['location_long', 'location_short', 'latitude', 'longitude']].drop_duplicates()
    for index, row in unique_locations.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=10,
            popup=row['location_long'],
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7
        ).add_to(m)

        folium.map.Marker(
            [row['latitude'], row['longitude']],
            icon=folium.DivIcon(
                html=f"""<div style="font-size: 12pt; color: white; background-color: blue; 
                         border-radius: 50%; padding: 4px; text-align: center;">{row['location_short']}</div>"""
            )
        ).add_to(m)

    # Legenda
    afkortingen = unique_locations[['location_short', 'location_long']]
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 270px; z-index:9999; 
        background-color: rgba(255, 255, 255, 0.95);
        border: 2px solid #444; 
        border-radius: 8px; 
        padding: 12px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.4);
        font-size: 13px;
        color: #333;
        font-weight: 500;
        line-height: 1.4;
    ">
    <b style="font-size:14px; color:#222;">Legenda (afkortingen):</b><br>
    """
    for index, row in afkortingen.iterrows():
        legend_html += f"<b>{row['location_short']}</b> = {row['location_long']}<br>"
    legend_html += "</div> {% endmacro %}"

    class Legend(MacroElement):
        def __init__(self, legend_html):
            super().__init__()
            self._template = Template(legend_html)

    m.get_root().add_child(Legend(legend_html))

    st_folium(m, width=950, height=600)
    st.write(f"Aantal metingen na filtering: {len(df)}")

# ========== ANALYSE TAB MET KAART ==========
with tab2:
    st.subheader("Gemiddelde Geluidsbelasting (SEL_dB) per Locatie (Visualisatie op Kaart)")

    avg_per_location = df.groupby(['location_long', 'location_short']).agg({
        'SEL_dB': 'mean',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    avg_per_location.columns = ['Locatie', 'Afkorting', 'Gemiddelde_SEL_dB', 'lat', 'lon']

    m_avg = folium.Map(location=[52.26, 4.77], zoom_start=12)

    # Voeg de cirkels toe
    for index, row in avg_per_location.iterrows():
        folium.Circle(
            location=[row['lat'], row['lon']],
            radius=(row['Gemiddelde_SEL_dB'] - 50) * 15,
            color='red',
            fill=True,
            fill_opacity=0.5,
        ).add_to(m_avg)

        # Voeg de afkorting toe als visuele marker
        folium.map.Marker(
            [row['lat'], row['lon']],
            icon=folium.DivIcon(
                html=f"""<div style="font-size: 12pt; color: white; background-color: red; 
                         border-radius: 50%; padding: 4px; text-align: center;">{row['Afkorting']}</div>"""
            )
        ).add_to(m_avg)

        # Klikbare (onzichtbare) marker voor de popup met gemiddelde SEL_dB
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"{row['Locatie']}: Gemiddelde SEL_dB = {row['Gemiddelde_SEL_dB']:.2f}",
            icon=folium.Icon(color='white', icon_color='white', icon='info-sign')
        ).add_to(m_avg)

    # Legenda bouwen
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 270px; z-index:9999; 
        background-color: rgba(255, 255, 255, 0.95);
        border: 2px solid #444; 
        border-radius: 8px; 
        padding: 12px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.4);
        font-size: 13px;
        color: #333;
        font-weight: 500;
        line-height: 1.4;
    ">
    <b style="font-size:14px; color:#222;">Legenda (afkortingen):</b><br>
    """
    for index, row in avg_per_location.iterrows():
        legend_html += f"<b>{row['Afkorting']}</b> = {row['Locatie']}<br>"
    legend_html += "</div> {% endmacro %}"

    class Legend(MacroElement):
        def __init__(self, legend_html):
            super().__init__()
            self._template = Template(legend_html)

    m_avg.get_root().add_child(Legend(legend_html))

    st_folium(m_avg, width=950, height=600)

    st.write("""
    **Uitleg:** De grootte van de cirkel geeft de gemiddelde geluidsbelasting (SEL_dB) aan. 
    Hoe groter de cirkel, hoe meer geluid die locatie gemiddeld ervaart door overvliegende vliegtuigen.
    De afkortingen op de kaart verwijzen naar de locaties en zijn hieronder terug te vinden in de legenda.
    Klik op de locatie om het exacte gemiddelde te zien.
    """)


# ========== NIEUWE TAB VOOR VLIEGTUIG ANALYSE ==========
with tab3:
    st.subheader("Analyse: Gemiddelde Geluidsbelasting per Vliegtuigtype (met Hoogtefilter)")

    # Combinatie type en icao_type voor duidelijkheid
    df['vliegtuig_type'] = df['type'] + " (" + df['icao_type'] + ")"

    # Hoogtefilter slider (extra!)
    min_hoogte_type = int(df['altitude'].min())
    max_hoogte_type = int(df['altitude'].max())
    hoogte_range_type = st.slider(
        "Selecteer een hoogtebereik voor deze analyse (meters)", 
        min_hoogte_type, 
        max_hoogte_type, 
        (min_hoogte_type, max_hoogte_type)
    )

    # Filteren op gekozen hoogtebereik
    df_type_filtered = df[(df['altitude'] >= hoogte_range_type[0]) & (df['altitude'] <= hoogte_range_type[1])]

    # Filter op minimaal aantal metingen
    min_meting = st.slider("Minimaal aantal metingen per vliegtuigtype", 5, 50, 10)

    type_stats = df_type_filtered.groupby('vliegtuig_type').agg({
        'SEL_dB': ['mean', 'count']
    }).reset_index()
    type_stats.columns = ['vliegtuig_type', 'gemiddelde_SEL_dB', 'aantal_metingen']
    type_stats = type_stats[type_stats['aantal_metingen'] >= min_meting]
    type_stats = type_stats.sort_values(by='gemiddelde_SEL_dB', ascending=True)

    st.write(f"{len(type_stats)} vliegtuigtypes met minimaal {min_meting} metingen in het hoogtebereik {hoogte_range_type[0]}m - {hoogte_range_type[1]}m.")

    # Interactieve bar chart
    fig = px.bar(
        type_stats,
        x='gemiddelde_SEL_dB',
        y='vliegtuig_type',
        orientation='h',
        color='gemiddelde_SEL_dB',
        color_continuous_scale='RdYlGn_r',
        labels={'vliegtuig_type': 'Vliegtuigtype', 'gemiddelde_SEL_dB': 'Gemiddelde SEL_dB'},
        title='Gemiddelde Geluidsbelasting per Vliegtuigtype (laag = beter)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optionele top 5 weergeven
    if st.checkbox(" Toon de stilste en luidste 5 vliegtuigtypes"):
        st.markdown("###  Stilste 5 types")
        st.dataframe(type_stats.head(5).reset_index(drop=True).style.format({"gemiddelde_SEL_dB": "{:.2f}"}))

        st.markdown("###  Luidste 5 types")
        st.dataframe(type_stats.tail(5).reset_index(drop=True).style.format({"gemiddelde_SEL_dB": "{:.2f}"}))
    

# ========== TIJDLIJN TAB ========== 
with tab4:
    st.subheader(" Tijdlijn van Gemiddelde Geluidsbelasting (SEL_dB)")

    # Filters
    locatie_filter = st.selectbox(" Filter op locatie (optioneel)", ["Alle"] + sorted(df['location_long'].unique().tolist()))

    # Slimme vliegtuigfilter met minimum metingen (bijvoorbeeld 20)
    min_meting_vliegtuig = 20
    vliegtuig_counts = df['type'].value_counts()
    vliegtuig_types_voldoende = vliegtuig_counts[vliegtuig_counts >= min_meting_vliegtuig].index.tolist()
    
    # Optioneel: we tonen de count erbij voor duidelijkheid
    vliegtuig_options = ["Alle"] + [f"{vliegtuig} ({vliegtuig_counts[vliegtuig]})" for vliegtuig in vliegtuig_types_voldoende]

    vliegtuig_filter = st.selectbox(
        f" Filter op vliegtuigtype (min {min_meting_vliegtuig} metingen)", 
        vliegtuig_options
    )

    # Tijdresolutie
    tijd_resolutie = st.selectbox(" Kies tijdsresolutie", ["15min", "30min", "1H"], index=0)

    # Dataset kopiëren en filters toepassen
    df_tijd = df.copy()
    if locatie_filter != "Alle":
        df_tijd = df_tijd[df_tijd['location_long'] == locatie_filter]

    # Extract raw vliegtuig type zonder count als nodig
    if vliegtuig_filter != "Alle":
        vliegtuig_naam = vliegtuig_filter.split(' (')[0]
        df_tijd = df_tijd[df_tijd['type'] == vliegtuig_naam]

    # Tijd goed zetten
    df_tijd['time'] = pd.to_datetime(df_tijd['time'])

    # Groeperen per tijdsblok
    df_tijd['tijdblok'] = df_tijd['time'].dt.floor(tijd_resolutie)
    tijd_stats = df_tijd.groupby('tijdblok').agg({'SEL_dB': 'mean'}).reset_index()

    # Controle op data
    if not tijd_stats.empty:
        fig_tijd = px.line(
            tijd_stats,
            x='tijdblok',
            y='SEL_dB',
            title=f"Gemiddelde Geluidsbelasting over Tijd ({tijd_resolutie} blokken)",
            labels={'tijdblok': 'Tijd', 'SEL_dB': 'Gemiddelde SEL_dB (dB)'},
            markers=True
        )
        fig_tijd.update_traces(line=dict(color='blue'), marker=dict(size=6))
        st.plotly_chart(fig_tijd, use_container_width=True)
    else:
        st.warning("⚠️ Geen data beschikbaar voor deze selectie. Kies een andere locatie, vliegtuigtype of pas de filters aan.")




# %%

start_date = int(pd.to_datetime('2025-01-01').timestamp())
end_date = int(pd.to_datetime('2025-03-23').timestamp())

response = requests.get(f'https://sensornet.nl/dataserver3/event/collection/nina_events/stream?conditions%5B0%5D%5B%5D=time&conditions%5B0%5D%5B%5D=%3E%3D&conditions%5B0%5D%5B%5D={start_date}&conditions%5B1%5D%5B%5D=time&conditions%5B1%5D%5B%5D=%3C&conditions%5B1%5D%5B%5D={end_date}&conditions%5B2%5D%5B%5D=label&conditions%5B2%5D%5B%5D=in&conditions%5B2%5D%5B2%5D%5B%5D=21&conditions%5B2%5D%5B2%5D%5B%5D=32&conditions%5B2%5D%5B2%5D%5B%5D=33&conditions%5B2%5D%5B2%5D%5B%5D=34&args%5B%5D=aalsmeer&args%5B%5D=schiphol&fields%5B%5D=time&fields%5B%5D=location_short&fields%5B%5D=location_long&fields%5B%5D=duration&fields%5B%5D=SEL&fields%5B%5D=SELd&fields%5B%5D=SELe&fields%5B%5D=SELn&fields%5B%5D=SELden&fields%5B%5D=SEL_dB&fields%5B%5D=lasmax_dB&fields%5B%5D=callsign&fields%5B%5D=type&fields%5B%5D=altitude&fields%5B%5D=distance&fields%5B%5D=winddirection&fields%5B%5D=windspeed&fields%5B%5D=label&fields%5B%5D=hex_s&fields%5B%5D=registration&fields%5B%5D=icao_type&fields%5B%5D=serial&fields%5B%5D=operator&fields%5B%5D=tags')

colnames = pd.DataFrame(response.json()['metadata'])
data = pd.DataFrame(response.json()['rows'])
data.columns = colnames.headers

data['time'] = pd.to_datetime(data['time'], unit = 's')

print(data['time'].min(),data['time'].max())



# %%
knmi = pd.read_excel("KNMI_data_schiphol.xlsx", skiprows = 41)
knmi.head()

# %%
# 1. Datum omzetten in sensornet data
data['date'] = pd.to_datetime(data['time']).dt.date

# 2. Voeg stationinformatie toe (stel dat dit uit de 'location_short' komt)
data['station'] = data['location_long']

# 3. Gemiddelde SEL_dB per station per dag berekenen
agg_data = data.groupby(['date', 'station']).agg({'SEL_dB': 'mean'}).reset_index()

# 4. KNMI data datum omzetten naar datetime formaat
knmi['date'] = pd.to_datetime(knmi['YYYYMMDD'], format='%Y%m%d').dt.date

# 5. Merge de datasets op datum
merged_df = pd.merge(agg_data, knmi, on='date', how='inner')

# 6. Verwijder de YYYYMMDD kolom (deze kolom komt van KNMI)
merged_df = merged_df.drop(columns=['YYYYMMDD'])

# 7. Resultaat bekijken
print(merged_df.head())

# %%
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px

# Meetstations en coördinaten
meetstations = {
    'Uiterweg': (52.26224927007567, 4.7330575976587905),
    'Aalsmeerderweg': (52.280158469807105, 4.795345755018842),
    'Blaauwstraat': (52.2649466795041, 4.7740958913400995),
    'Hornweg': (52.27213650444824, 4.7935705607232775),
    'Kudelstaartseweg': (52.24558794567681, 4.755324955329966),
    'Darwinstraat': (52.23621816215143, 4.758856663004422),
    'Copierstraat': (52.22957895035498, 4.739337496658634)
}

# Cache de verwerking van de data
@st.cache_data
def process_data(df):
    df['RH'] = df['RH'].apply(lambda x: max(0, x)) 
    return df

merged_df = process_data(merged_df)

selected_date = st.date_input("Selecteer een datum", min_value=min(merged_df['date']), max_value=max(merged_df['date']))

# **Kaart genereren**
map = folium.Map(location=[52.26, 4.76], zoom_start=12)

for station, coords in meetstations.items():
    # Zoek de geluidswaarde voor de geselecteerde datum
    data = merged_df[(merged_df['station'] == station) & (merged_df['date'] == selected_date)]
    if not data.empty:
        sel_db = round(data["SEL_dB"].values[0], 2)
        popup_text = f"{station}<br><b>Geluidsniveau: {sel_db} dB</b>"
        kleur = "red"
    else:
        popup_text = f"{station}<br><b>Geen data beschikbaar</b>"
        kleur = "gray"

    # **Marker met popup**
    folium.Marker(
        location=coords,
        popup=folium.Popup(popup_text, max_width=250),
        icon=folium.Icon(color=kleur, icon="info-sign")
    ).add_to(map)

    # **Label direct op de kaart**
    folium.Marker(
        location=coords,
        icon=folium.DivIcon(
            html=f"""
                <div style="font-size: 10pt; 
                            font-weight: bold;
                            color: black; 
                            background-color: rgba(255, 255, 255, 0.8);
                            border-radius: 4px; 
                            padding: 3px; 
                            text-align: center;">
                    {station}
                </div>"""
        )
    ).add_to(map)

folium_static(map)

# **Checkbox voor scatterplot**
show_scatter = st.checkbox("Toon scatterplot", value=True)

if show_scatter:
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Kies variabele voor de X-as:", ["FG", "RH", "DDVEC"], 
                              format_func=lambda x: {"FG": "Windsnelheid", "RH": "Neerslag", "DDVEC": "Windrichting"}[x])
    with col2:
        selected_station = st.selectbox("Kies meetstation:", ["Alle"] + list(meetstations.keys()))

    # **Filteren van de data**
    filtered_df = merged_df if selected_station == "Alle" else merged_df[merged_df['station'] == selected_station]

    # **Interactieve scatterplot met vaste breedte**
    fig = px.scatter(
        filtered_df, x=x_axis, y="SEL_dB", 
        labels={"SEL_dB": "Geluidsniveau (dB)", "FG": "Windsnelheid (m/s)", "RH": "Neerslag (mm)", "DDVEC": "Windrichting (°)"},
        title=f"Correlatie tussen {x_axis} en Geluidsniveau ({selected_station})",
        hover_data=["station", "date"],
        width=600  # Hier wordt de breedte beperkt
    )

    st.plotly_chart(fig, use_container_width=False)


# %%
import requests
import pandas as pd
import numpy as np

start_date = int(pd.to_datetime('2025-03-01').timestamp())
end_date = int(pd.to_datetime('2025-03-3').timestamp())

response = requests.get(f'https://sensornet.nl/dataserver3/event/collection/nina_events/stream?conditions%5B0%5D%5B%5D=time&conditions%5B0%5D%5B%5D=%3E%3D&conditions%5B0%5D%5B%5D={start_date}&conditions%5B1%5D%5B%5D=time&conditions%5B1%5D%5B%5D=%3C&conditions%5B1%5D%5B%5D={end_date}&conditions%5B2%5D%5B%5D=label&conditions%5B2%5D%5B%5D=in&conditions%5B2%5D%5B2%5D%5B%5D=21&conditions%5B2%5D%5B2%5D%5B%5D=32&conditions%5B2%5D%5B2%5D%5B%5D=33&conditions%5B2%5D%5B2%5D%5B%5D=34&args%5B%5D=aalsmeer&args%5B%5D=schiphol&fields%5B%5D=time&fields%5B%5D=location_short&fields%5B%5D=location_long&fields%5B%5D=duration&fields%5B%5D=SEL&fields%5B%5D=SELd&fields%5B%5D=SELe&fields%5B%5D=SELn&fields%5B%5D=SELden&fields%5B%5D=SEL_dB&fields%5B%5D=lasmax_dB&fields%5B%5D=callsign&fields%5B%5D=type&fields%5B%5D=altitude&fields%5B%5D=distance&fields%5B%5D=winddirection&fields%5B%5D=windspeed&fields%5B%5D=label&fields%5B%5D=hex_s&fields%5B%5D=registration&fields%5B%5D=icao_type&fields%5B%5D=serial&fields%5B%5D=operator&fields%5B%5D=tags')

colnames = pd.DataFrame(response.json()['metadata'])
data = pd.DataFrame(response.json()['rows'])
data.columns = colnames.headers

data['time'] = pd.to_datetime(data['time'], unit = 's')

print(data['time'].min(),data['time'].max())
data.head(10)
data.info()

# Zorg dat je deze kolommen hebt: icao_type, type, SEL_dB
sensordata_clean = data.dropna(subset=['icao_type', 'SEL_dB', 'type'])

# Groepeer op ICAO-code én vliegtuignaam met merk
sound_per_type = (
    sensordata_clean
    .groupby(['icao_type', 'type'])['SEL_dB']
    .mean()
    .reset_index()
    .rename(columns={'type': 'aircraft_fullname', 'SEL_dB': 'avg_SEL_dB'})
)


# %%
import pandas as pd

# Laad alleen relevante kolommen ICAO-type, modelnaam, registratie, aantal stoelen
df = pd.read_excel("fleetnew.xlsx", usecols=[4, 5, 6, 13], header=None)

df = df.iloc[1:].copy()

df.columns = ['icao_type', 'aircraft_model', 'registration', 'seat_capacity']

# Zet seat_capacity om naar getallen
df['seat_capacity'] = pd.to_numeric(df['seat_capacity'], errors='coerce')

# Strip spaties en hoofdletters consistent maken
df['icao_type'] = df['icao_type'].astype(str).str.strip().str.upper()
df['registration'] = df['registration'].astype(str).str.strip().str.upper()

# Verwijder de foute eerste rij met tekst (die als data is ingelezen)
df = df[df['icao_type'] != 'AIRCRAFT ICAO CODE'].copy()

# Zet seat_capacity opnieuw om naar numeriek, voor de zekerheid
df['seat_capacity'] = pd.to_numeric(df['seat_capacity'], errors='coerce')

# Bekijk het resultaat
df.head()


# %%
# Groepeer fleetdata op ICAO-code + naam, neem gemiddelde capaciteit per type
fleet_summary = (
    df[df['seat_capacity'] > 0]
    .groupby(['icao_type', 'aircraft_model'])['seat_capacity']
    .mean()
    .reset_index()
)

# %%
# Merge met Sensornet-gegevens
merged = pd.merge(sound_per_type, fleet_summary, on='icao_type', how='inner')

# Bereken geluid per passagier
merged['sound_per_passenger'] = merged['avg_SEL_dB'] / merged['seat_capacity']


# %%
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Top 10 Stilste Vliegtuigtypes per Passagier")

# Cache de verwerking van de data
@st.cache_data
def process_data(merged):
    merged["icao_label"] = merged["icao_type"] + " (" + merged["aircraft_fullname"] + ")"
    return merged[
        merged["sound_per_passenger"].notna() &
        np.isfinite(merged["sound_per_passenger"]) &
        (merged["sound_per_passenger"] > 0)
    ].copy()

merged_clean = process_data(merged)

# Slider toevoegen om aantal vliegtuigen te kiezen
aantal_vliegtuigen = st.slider("Aantal vliegtuigen weergeven:", min_value=5, max_value=20, value=10)

# Sorteer en selecteer top vliegtuigen
top_vliegtuigen = (
    merged_clean.sort_values("sound_per_passenger")
    .drop_duplicates(subset=["icao_label"])
    .head(aantal_vliegtuigen)
    .copy()
)

# Check of er genoeg vliegtuigtypes zijn
if len(top_vliegtuigen) < aantal_vliegtuigen:
    st.warning(f"Er zijn slechts {len(top_vliegtuigen)} unieke vliegtuigtypes beschikbaar.")

# Plot maken
fig = px.bar(
    top_vliegtuigen,
    x="sound_per_passenger",
    y="icao_label",
    orientation="h",
    color="sound_per_passenger",
    color_continuous_scale="Blues",
    labels={
        "sound_per_passenger": "Gemiddeld geluid per passagier (dB)",
        "icao_label": "Vliegtuigtype"
    },
    title=f"Top {aantal_vliegtuigen} Stilste Vliegtuigtypes per Passagier"
)

# Layout aanpassen zodat stilste bovenaan staat
fig.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    height=600
)

# Toon grafiek
st.plotly_chart(fig, use_container_width=True)



# %%

st.sidebar.title("Welcome To Sidebar")

page= st.sidebar.radio("Go To", ["Geluidsmetingen rond Schiphol", "Kaart & Scatterplot", "Top 10 Stilste Vliegtuigen"])

# Logica om de juiste visualisatie weer te geven
if page == "Geluidsmetingen rond Schiphol": 
    st.subheader("Geluidsmetingen rond Schiphol")
    
elif page == "Kaart & Scatterplot": 
    st.subheader("Kaart & Scatterplot")
    
elif page == "Top 10 Stilste Vliegtuigen": 
    st.subheader("Top 10 Stilste Vliegtuigen")
    


