# Pedestrian Volume Data Visualization Dashboard (PEDAT)
# Author:   Amir Rafe (amir.rafe@usu.edu)
# File:     dash_beta.py
# Version:  1.0.10.beta  
# About:    A streamlit webapp to visualize pedestrian volum data in Utah

# Streamlit for web app functionality
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
from streamlit_keplergl import keplergl_static

# Data manipulation and analysis
import pandas as pd
import numpy as np
import json

# Visualization libraries
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.colors
import folium
from folium.plugins import Draw, MarkerCluster, Search, FastMarkerCluster
from keplergl import KeplerGl
import pydeck as pdk
from branca.colormap import linear

# Date and time handling
from datetime import datetime, date, timedelta
import pytz
import time

# Geospatial and geometry operations
from shapely.geometry import Polygon, Point

# PDF and image handling
from fpdf import FPDF
from PIL import Image
from reportlab.lib.pagesizes import landscape, letter, A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import imgkit
import tempfile
from io import BytesIO
from tempfile import NamedTemporaryFile

# Google Cloud services
from google.cloud import bigquery
from google.oauth2 import service_account

# Miscellaneous utilities
import os
import base64
import threading
import random
import copy
import kaleido

# Setting default scale for kaleido
pio.kaleido.scope.default_scale = 2


# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

# Define SQL query to retrieve the data from BigQuery
sql_query = """
SELECT *
FROM `decent-digit-387716.pedat_dataset.map_data`
"""
sql_query2 = """
SELECT *
FROM `decent-digit-387716.pedat_dataset.pedatutah`
WHERE ADDRESS IN UNNEST(@selected_signals)
"""
sql_query3 = """
SELECT *
FROM `decent-digit-387716.pedat_dataset.pedatdaily`
WHERE ADDRESS IN UNNEST(@selected_signals)
"""


# Define the title
title = 'Pedestrian Volume Data Visualization Dashboard'
text1 = 'This website provides data and visualizations of pedestrian volume at various locations in Utah. "Pedestrian volume" is an estimate of pedestrian crossing volume at an intersection, currently based on pedestrian push-button presses at traffic signals. See the "How to use" and "Notes" tabs on the left, or following the step-by-step instructions below.'
text2 = 'As of 10/31/2023, this website contains pedestrian volume data for 2,030 locations in Utah between 2018 and 2022.'

# Define the x and y axis labels
x_axis_label = 'TIME1'
y_axis_label = 'PED'

# Generating a randomized color map for unique signals
def create_color_map(unique_signals):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_signals)))
    colors = [matplotlib.colors.rgb2hex(c) for c in colors]
    random.shuffle(colors)  # Shuffle colors for variety
    color_map = dict(zip(unique_signals, colors))
    return color_map

@st.cache_resource
# Formatting numerical values into readable metrics (B, M, K)
def format_metric(value):
    # Check if the value is greater than or equal to 1 billion
    if value >= 1e9:
        return f'{round(value/1e9,1)} B'
    
    # Check if the value is greater than or equal to 1 million
    elif value >= 1e6:
        return f'{round(value/1e6,1)} M'
    
    # Check if the value is greater than or equal to 1 thousand
    elif value >= 1e3:
        return f'{round(value/1e3,1)} K'
    
    # Otherwise, return the value as is
    else:
        return str(value)

@st.cache_resource
# creating a time-series chart with custom aggregation and filtering options
def make_chart(df, signals, start_date, end_date, aggregation_method, location, Dash_selected, color_map, template='plotly'):
    if aggregation_method == 'Hour':
        groupby = ['SIGNAL','ADDRESS', pd.Grouper(key='TIME1', freq='1H')]
    elif aggregation_method == 'Day':
        groupby = ['SIGNAL','ADDRESS', pd.Grouper(key='TIME1', freq='1D')]
    elif aggregation_method == 'Week':
        groupby = ['SIGNAL','ADDRESS', pd.Grouper(key='TIME1', freq='1W')]
    elif aggregation_method == 'Month':
        groupby = ['SIGNAL','ADDRESS', pd.Grouper(key='TIME1', freq='1M')]
    elif aggregation_method == 'Year':
        groupby = ['SIGNAL','ADDRESS', pd.Grouper(key='TIME1', freq='1Y')]

    if location == 'All':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))]
    

    # Aggregate the data
    if location == 'All':
        df_agg = df_filtered.groupby(groupby).agg({'PED': 'sum'}).reset_index()
        df_agg['PED'] = df_agg['PED'].round(0)
        df_agg = df_agg.rename(columns={'SIGNAL': 'Signal ID', 'TIME1':'Timestamp' , 'PED':'Pedestrian'})

    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby(groupby).agg({'PED': 'sum'}).reset_index()
        df_agg['PED'] = df_agg['PED'].round(0)
        df_agg = df_agg.rename(columns={'SIGNAL': 'Signal ID' , 'TIME1':'Timestamp' , 'PED':'Pedestrian'})

    # Modify address for display, but this won't directly affect coloring or legend
    df_agg['Address'] = df_agg['ADDRESS'].str.replace(r'^\d+\s*--\s*', '', regex=True)

    # Create the line chart
    if aggregation_method == 'Hour':
        x_axis_label = '<b>Time<b>'
        fig = px.line(df_agg, x='Timestamp', y='Pedestrian', color='Signal ID', 
                  hover_data=['Pedestrian', 'Timestamp', 'Address'], 
                  color_discrete_map=color_map, template=template)
    else:
        x_axis_label = '<b>Date<b>'
        fig = px.line(df_agg, x='Timestamp', y='Pedestrian', color='Signal ID', 
                  hover_data=['Pedestrian', 'Timestamp', 'Address'], 
                  color_discrete_map=color_map, template=template)

    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text='<b>Pedestrian Volume<b>')
    fig.update_traces(line=dict(width=3))
    fig.update_layout(showlegend=True , legend_title_text='<b>Location<b>')


    # Set the time slider at the bottom
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    fig6 = copy.deepcopy(fig)
    fig6.update_layout(xaxis=dict(rangeselector=dict(visible=False ),rangeslider=dict(visible=False),type="date"))
    fig6.update_layout(
        showlegend=True,
        legend_title_text='<b>Location<b>',
        template='plotly',
        autosize=False,
        width=920,
        height=520
        )
    fig6.write_image("fig1.png")
    return fig


@st.cache_resource
# Aggregating and formatting data for table display based on selected criteria
def make_table(df, signals, start_date, end_date, aggregation_method, location,Dash_selected):
    if aggregation_method == 'Hour':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1H')]
    elif aggregation_method == 'Day':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1D')]
    elif aggregation_method == 'Week':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1W')]
    elif aggregation_method == 'Month':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1M')]
    elif aggregation_method == 'Year':
        groupby = ['ADDRESS', pd.Grouper(key='TIME1', freq='1Y')]

    if location == 'All':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))]
    
    # Aggregate the data
    if location == 'All':
        df_agg = df_filtered.groupby(groupby).agg({'PED': 'sum', 'CITY': 'first', 'SIGNAL': 'first' , 'LAT': 'first' , 'LNG': 'first' }).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby(groupby).agg({'PED': 'sum', 'P': 'first', 'CITY': 'first', 'SIGNAL': 'first' , 'LAT': 'first' , 'LNG': 'first'}).reset_index()

    df_agg['PED'] = df_agg['PED'].round(0)
    df_agg.rename(columns={'SIGNAL': 'Signal ID' , 'ADDRESS': 'Address' , 'TIME1':'Timestamp' , 'PED':'Pedestrian' , 'CITY':'City' , 'P': 'Phase' , 'LAT':'Latitude' , 'LNG': 'Longtitude' }, inplace=True)
    
    # Split the 'Address' column on '--' and take the second part if it exists
    df_agg['Address'] = df_agg['Address'].apply(lambda x: x.split('-- ')[1] if '--' in x else x)

    # Continue with the existing code to select columns and reset index...
    if 'Phase' in df_agg.columns:
        df_agg = df_agg[['Signal ID', 'Address' , 'Timestamp', 'Phase', 'Pedestrian', 'City' , 'Latitude' , 'Longtitude']]
    else:
        df_agg = df_agg[['Signal ID', 'Address' , 'Timestamp', 'Pedestrian', 'City' , 'Latitude' , 'Longtitude']]


    df_agg.reset_index(drop=True, inplace=True)  # remove index column
    df_agg['Timestamp'] = df_agg['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df_agg

@st.cache_resource
# Creating combined pie and treemap charts for pedestrian volume analysis
def make_pie_and_bar_chart(df, signals, start_date, end_date, location,Dash_selected , color_map):

    if location == 'All':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & (df['ADDRESS'].isin(signals))]
        # Modify address format in the original dataframe
        df_filtered['Address'] = df_filtered['ADDRESS'].str.replace(r'^\d+\s*--\s*', '', regex=True)
        # Create a mapping from SIGNAL to the first modified address encountered for each SIGNAL
        signal_to_address = df_filtered.groupby('SIGNAL')['Address'].first().to_dict()
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))]
        # Modify address format in the original dataframe
        df_filtered['Address'] = df_filtered['ADDRESS'].str.replace(r'^\d+\s*--\s*', '', regex=True)
        # Create a mapping from SIGNAL to the first modified address encountered for each SIGNAL
        signal_to_address = df_filtered.groupby('SIGNAL')['Address'].first().to_dict()

    # Aggregate the data
    if location == 'All':
        df_agg = df_filtered.groupby(['ADDRESS', 'SIGNAL']).agg({'PED': 'sum','CITY': 'first'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered['P'] == int(filter_val)].groupby(['ADDRESS', 'SIGNAL']).agg({'PED': 'sum' , 'CITY': 'first'}).reset_index()

    # Aggregate the data by signal and sum the pedestrian counts
    df_agg1 = df_agg.groupby('SIGNAL').agg({'PED': 'sum'}).reset_index()
    df_agg1 = df_agg1.rename(columns={'SIGNAL': 'Signal ID'})
    df_agg1['Signal'] = df_agg1['Signal ID'].astype(str)

    # Round the pedestrian count to two decimal places
    df_agg1['PED'] = df_agg1['PED'].round(0)
    df_agg1['Address'] = df_agg1['Signal ID'].map(signal_to_address)


    # Create the pie chart
    colors = [color_map.get(x, '#000000') for x in df_agg1['Signal ID']]
    fig_pie = go.Figure(data=[go.Pie(labels=df_agg1['Signal ID'], values=df_agg1['PED'], name='Signal ID', 
                                 marker=dict(colors=colors))])

    # Create the treemap figure
    marker_colors = df_agg1['Signal ID'].apply(lambda x: color_map.get(x, '#000000'))
    fig_treemap = go.Figure(data=go.Treemap(
        labels=df_agg1['Signal ID'],
        parents=['']*len(df_agg1),
        values=df_agg1['PED'], 
        name='Signal ID',
        marker=dict(colors=marker_colors),
        customdata=df_agg1['Address'],
        hovertemplate='<b>Signal ID:</b> %{label}<br><b>Pedestrian:</b> %{value}<br><b>Address:</b> %{customdata}<extra></extra>',
    ))

    fig_treemap.update_layout(title='Pedestrian Volume by location', showlegend=False)

    # Combine the pie, bar, and treemap charts
    fig_combined = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'treemap'}]])
    fig_combined.add_trace(fig_pie.data[0], row=1, col=1)
    fig_combined.add_trace(fig_treemap.data[0], row=1, col=2)
    fig_combined.update_layout(showlegend=True ,  legend_title_text='<b>Location<b>')
    fig_combined.update_layout(template='plotly')
    fig4 = copy.deepcopy(fig_pie)
    fig4.update_layout(
        showlegend=True,
        legend=dict(title='<b>Location<b>'),
        template='plotly'
    )
    fig4.write_image("fig2.png")
    
    return fig_combined , df_agg1

@st.cache_resource
# Generating hourly average pedestrian volume bar chart with date and location filtering
def make_bar_chart(df, signals, start_date, end_date, location, Dash_selected):
    # Convert "TIME1" to datetime and extract the hour and date information
    df['Hour'] = pd.to_datetime(df['TIME1']).dt.hour
    df['Date'] = pd.to_datetime(df['TIME1']).dt.date
    
    if location == 'All':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & 
                         (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & 
                         (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & 
                         (df['TIME1'] <= end_date) & 
                         (df['ADDRESS'].isin(signals))]

    # Aggregate the data
    if location == 'All':
        df_agg = df_filtered.groupby(['Date', 'Hour']).agg({'PED': 'sum'}).reset_index()
    else:
        df_agg = df_filtered[df_filtered[col] == int(filter_val)].groupby(['Date', 'Hour']).agg({'PED': 'sum'}).reset_index()

    # Calculate the hourly average by dividing the sum of PED by number of days
    df_hourly_avg = df_agg.groupby('Hour').agg({'PED': 'mean'}).reset_index()

    # Round the pedestrian count to 0 decimal places
    df_hourly_avg['PED'] = df_hourly_avg['PED'].round(0)
    
    # Define the data and layout for the bar chart
    data = [go.Bar(x=df_hourly_avg['Hour'], y=df_hourly_avg['PED'], text=df_hourly_avg['PED'], texttemplate='%{y:,.0f}', textposition='auto')]
    layout = go.Layout(xaxis_title='<b>Hour of the day<b>', yaxis_title='<b>Pedestrian Volume<b>', xaxis=dict(tickmode='linear', dtick=1))

    # Create the bar chart
    fig_bar = go.Figure(data=data, layout=layout)
    fig_bar.update_yaxes(tickformat=".0f")
    fig_bar.update_layout(template='plotly')
    fig2 = copy.deepcopy(fig_bar)
    fig2.update_layout(autosize=False, width=920, height=520)
    fig2.update_layout(template='plotly')
    fig2.write_image("fig3.png")
    
    return fig_bar, df_hourly_avg

@st.cache_resource
# Creating a bar chart to visualize average pedestrian volume by day of the week
def make_bar_chart2(df, signals, start_date, end_date, location, Dash_selected):
    if location == 'All':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Convert "TIME1" to datetime and extract the day of the week information
    df['Day_of_Week'] = pd.to_datetime(df['TIME1']).dt.dayofweek
    df['Date'] = pd.to_datetime(df['TIME1']).dt.date

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) &
                         (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) &
                         (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) &
                         (df['TIME1'] <= end_date) &
                         (df['ADDRESS'].isin(signals))]

    # Filter based on the location
    if location != 'All':
        df_filtered = df_filtered[df_filtered[col] == int(filter_val)]

    # Group by date and day of the week, then sum the PED values
    df_agg = df_filtered.groupby(['Date', 'Day_of_Week']).agg({'PED': 'sum'}).reset_index()

    # Group by day of the week and calculate the average pedestrian counts
    df_agg2 = df_agg.groupby('Day_of_Week').agg({'PED': 'mean'}).reset_index()

    # Round the pedestrian count to 0 decimal places
    df_agg2['PED'] = df_agg2['PED'].round(0)
    
    # Create the bar chart
    fig_bar = go.Figure(data=[go.Bar(x=df_agg2['Day_of_Week'], y=df_agg2['PED'], showlegend=False ,  texttemplate='%{y:,.0f}', textposition='auto')])

    # Set the x-axis tick labels to the full names of the days of the week
    fig_bar.update_layout(xaxis_title='<b>Day of the week<b>', yaxis_title='<b>Pedestrian Volume<b>', showlegend=False)
    fig_bar.update_xaxes(tickmode='array', tickvals=[0, 1, 2, 3, 4, 5, 6], ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    fig_bar.update_yaxes(tickformat=".0f")
    fig_bar.update_layout(template='plotly')
    fig_bar.update_layout(template='plotly')
    fig3 = copy.deepcopy(fig_bar)
    fig3.update_layout(autosize=False, width=920, height=520)
    fig3.update_layout(template='plotly')
    fig3.write_image("fig4.png")

    return fig_bar, df_agg2

@st.cache_resource
# Generating a monthly average pedestrian volume bar chart with date and location filtering
def make_bar_chart3(df, signals, start_date, end_date, location, Dash_selected):
    if location == 'All':
        filter_val = 'all'
        col = 'PED'
    else:
        filter_val = location.split()[-1]
        col = 'P'

    # Filter the dataframe by the selected signals and date range
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df[(df['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & 
                         (df['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC')) & 
                         (df['ADDRESS'].isin(signals))]
    else:
        df_filtered = df[(df['TIME1'] >= start_date) & 
                         (df['TIME1'] <= end_date) & 
                         (df['ADDRESS'].isin(signals))]
    
    # Filter based on the location
    if location != 'All':
        df_filtered = df_filtered[df_filtered[col] == int(filter_val)]

    # Resample to daily frequency and sum the PED counts
    df_daily = df_filtered.resample('D', on='TIME1').agg({'PED': 'sum'}).reset_index()

    # Extract month and year from the "TIME1" column
    df_daily['Month'] = pd.to_datetime(df_daily['TIME1']).dt.month
    df_daily['Year'] = pd.to_datetime(df_daily['TIME1']).dt.year

    # Group by month and year, and calculate the average of PED
    df_agg = df_daily.groupby(['Month']).agg({'PED': 'mean'}).reset_index()

    # Group by month and calculate the average pedestrian counts
    df_agg2 = df_daily.groupby('Month').agg({'PED': 'mean'}).reset_index()

    # Round the pedestrian count to 0 decimal places
    df_agg2['PED'] = df_agg2['PED'].round(0)
    
    # Create the bar chart
    fig_bar = go.Figure(data=[go.Bar(x=df_agg2['Month'], y=df_agg2['PED'], showlegend=False, texttemplate='%{y:,.0f}', textposition='auto')])

    # Set the x-axis tick labels to the full names of the months
    fig_bar.update_layout(xaxis_title='<b>Month of the year<b>', yaxis_title='<b>Pedestrian Volume<b>', showlegend=False)
    fig_bar.update_xaxes(tickmode='array', 
                         tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
                         ticktext=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    fig_bar.update_yaxes(tickformat=".0f")
    fig_bar.update_layout(template='plotly')
    fig8 = copy.deepcopy(fig_bar)
    fig8.update_layout(autosize=False, width=920, height=520)
    fig8.update_layout(template='plotly')
    fig8.write_image("fig5.png")

    return fig_bar,df_agg2

@st.cache_resource
# Creating a bar chart for average daily pedestrian volume per signal with color mapping and date/location filters
def make_bar_chart4(df, signals, start_date, end_date, location, Dash_selected, color_map):
    # Filter based on location
    if location == 'All':
        df_filtered = df[df['ADDRESS'].isin(signals)]
    else:
        filter_val = location.split()[-1]
        df_filtered = df[(df['P'] == int(filter_val)) & (df['ADDRESS'].isin(signals))]
        
    # Filter based on date
    if Dash_selected == 'Recent data (last 1 year)':
        df_filtered = df_filtered[(df_filtered['TIME1'] >= pd.to_datetime(start_date).tz_localize('UTC')) & 
                                  (df_filtered['TIME1'] <= pd.to_datetime(end_date).tz_localize('UTC'))]
    else:
        df_filtered = df_filtered[(df_filtered['TIME1'] >= start_date) & (df_filtered['TIME1'] <= end_date)]

    # Aggregate by SIGNAL and Date, summing up the hourly PED values
    df_daily_sum = df_filtered.groupby(['SIGNAL', pd.Grouper(key='TIME1', freq='1D')]).agg({'PED': 'sum'}).reset_index()

    # Calculate the average daily PED for each SIGNAL
    df_agg = df_daily_sum.groupby('SIGNAL').agg({'PED': 'mean'}).reset_index()

    # Create the bar chart
    fig_bar = go.Figure()

    # Only include unique SIGNALs present in the filtered DataFrame, converted to strings
    unique_signals = df_agg['SIGNAL'].astype(int).unique()
    
    # Get colors for each signal based on the color map
    bar_colors = [color_map[signal] for signal in unique_signals]

    # Create the bar chart with colors
    fig_bar.add_trace(go.Bar(x=unique_signals, y=df_agg['PED'], marker_color=bar_colors,
                             texttemplate='%{y:,.0f}', textposition='auto'))
    
    fig_bar.update_xaxes(tickmode='array', 
                         tickvals=unique_signals,
                         type='category')

    fig_bar.update_layout(xaxis_title='<b>Location<b>',
                          yaxis_title='<b>Pedestrian Volume<b>')
    

    fig_bar.update_yaxes(tickformat=".0f")
    fig_bar.update_layout(template='plotly')
    fig18 = copy.deepcopy(fig_bar)
    fig18.update_layout(autosize=False, width=920, height=520)
    fig18.update_layout(template='plotly')
    fig18.write_image("fig7.png")

    return fig_bar, df_agg

    
@st.experimental_fragment
def make_map2(df, signals, aggregation_method, location_selected, Dash_selected):
       
    # Check if the 'TIME1' datetime objects are already timezone-aware
    if df['TIME1'].dt.tz is not None:
        # If they are, convert them to UTC (if they aren't already)
        df['TIME1'] = df['TIME1'].dt.tz_convert('UTC')
    else:
        # If they are timezone-naive, assume they are UTC and localize accordingly
        df['TIME1'] = pd.to_datetime(df['TIME1']).dt.tz_localize('UTC')

    # Now you have 'TIME1' as timezone-aware in UTC, you can strip the timezone if needed
    df['TIME1'] = df['TIME1'].dt.tz_localize(None)

    # Determine the date format based on dashboard selection
    if Dash_selected == 'Recent data (last 1 year)':
        # Moment.js format for date and time
        date_format = "YYYY-MM-DD"  

        # Streamlit time slider for selecting the date range
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=df['TIME1'].min().to_pydatetime(),
            max_value=df['TIME1'].max().to_pydatetime(),
            value=(df['TIME1'].min().to_pydatetime(), df['TIME1'].max().to_pydatetime()),
            format=date_format
        )

        # Streamlit slider for selecting the hour range
        start_hour, end_hour = st.slider(
            "Select Hour Range",
            min_value=0,
            max_value=23,
            value=(0, 23)
        )

        # Filter by date, hour, selected signals, and location
        mask = (df['TIME1'].dt.date >= start_date.date()) & (df['TIME1'].dt.date <= end_date.date()) & (df['TIME1'].dt.hour >= start_hour) & (df['TIME1'].dt.hour < end_hour) & (df['ADDRESS'].isin(signals))
        if location_selected == 'All':
            mask &= df['P'] >= 0  # include all values of P for intersections
        else:
            if location_selected.startswith('Phase'):
                phase_num = int(location_selected.split()[1])
                mask &= df['P'] == phase_num
            else:
                mask &= df['ADDRESS'] == location_selected
        df_filtered = df.loc[mask]
    else:
        start_date, end_date = st.slider(
        "Select Date Range",
        min_value=df['TIME1'].min().to_pydatetime(),
        max_value=df['TIME1'].max().to_pydatetime(),
        value=(df['TIME1'].min().to_pydatetime(), df['TIME1'].max().to_pydatetime()),
        format= "YYYY-MM-DD")
        mask = (df['TIME1'] >= start_date) & (df['TIME1'] <= end_date) & (df['ADDRESS'].isin(signals))
        df_filtered = df.loc[mask]
    
    # Define and apply aggregation methods
    agg_functions = {
        'Hour': 'sum',
        'Day': 'sum',
        'Week': 'sum',
        'Month': 'sum',
        'Year': 'sum'
    }
    aggregation_function = agg_functions[aggregation_method]

    # Aggregate data by location
    df_filtered.rename(columns={'LNG': 'LON'}, inplace=True)
    df_agg = df_filtered.groupby(['LAT', 'LON', 'ADDRESS' , 'SIGNAL']).agg({'PED': aggregation_function}).reset_index()
    

    # Create color map
    unique_signals = df['SIGNAL'].unique().tolist()
    mean_lat = df['LAT'].mean()
    mean_lng = df['LNG'].mean()
    # Create a Folium map with specified width and height
    m = folium.Map(location=[mean_lat, mean_lng],  zoom_start=13, tiles=None)
    # Add custom tile layers
    pedat_tiles = folium.TileLayer(
        tiles='https://api.mapbox.com/styles/v1/bashasvari/clhgx1yir00h901q1ecbt9165/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1IjoiYmFzaGFzdmFyaSIsImEiOiJjbGVmaTdtMmIwcXkzM3Jxam9hb2pwZ3BoIn0.JmYank8e3bmQ7RmRiVdTIg',
        attr='PEDAT map',
        name='PEDAT',
        overlay=False,
        control=True
    )
    satellite = folium.TileLayer(
        tiles='https://api.mapbox.com/styles/v1/bashasvari/cluvp5mkm000i01og0rbcgwmf/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1IjoiYmFzaGFzdmFyaSIsImEiOiJjbGVmaTdtMmIwcXkzM3Jxam9hb2pwZ3BoIn0.JmYank8e3bmQ7RmRiVdTIg',
        attr='Satellite',
        name='Satellite Map',
        overlay=False,
        control=True
    )

    # Add the PEDAT layer and show it by default
    pedat_tiles.add_to(m)
    satellite.add_to(m)

    # Adding other tile layers but not showing them by default
    folium.TileLayer('OpenStreetMap', name='Open Street Map', overlay=False, control=True).add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark Matter', overlay=False, control=True).add_to(m)

    # Create a color scale based on PED values
    max_ped = df_agg['PED'].max()
    min_ped = df_agg['PED'].min()
    colormap = linear.YlOrRd_09.scale(min_ped, max_ped)
    colormap.caption = 'Pedestrian Count'
    colormap.add_to(m)
    
    # Constants for scaling 
    base_radius = 4  # This is the base size for circles
    scale_factor = 0.04  # This factor will scale the transformation to appropriate map units
    # Square root scaling
    df_agg['Scaled_RADIUS'] = np.sqrt(df_agg['PED']) * scale_factor

    # Adding circles with scaled radii
    for idx, row in df_agg.iterrows():
        folium.CircleMarker(
            location=[row['LAT'], row['LON']],
            radius=max(base_radius, row['Scaled_RADIUS']),  # Ensure a minimum size for visibility
            popup=folium.Popup(f"<div style='width:250px;'><strong>Address:</strong><br>{row['ADDRESS']}<br><strong>Pedestrian Count:</strong><br>{round(row['PED'])}</div>", max_width=250),
            color=colormap(row['PED']),
            fill=True,
            fill_opacity=0.7,
            fill_color=colormap(row['PED'])
        ).add_to(m)
    # Use streamlit_folium for displaying the map with width and height
    #sw = df[['LAT', 'LNG']].min().values.tolist()
    #ne = df[['LAT', 'LNG']].max().values.tolist()
    #m.fit_bounds([sw, ne])
    folium.LayerControl().add_to(m)
    st_folium(m, width='80%', height=600)

    

# Define the Streamlit app
def main():
    # Set the app title
    st.set_page_config(page_title='PEDAT Dashboard' , page_icon="📈" , layout="wide"  )
    # Add a title to the sidebar
    st.title("Pedestrian Volume in Utah")
    st.markdown(text1)
    st.markdown(text2)
    
    st.markdown("""
            <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                }
            </style>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        .css-6qob1r {
        margin-top: -75px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Display UDOT logo in the sidebar
    udot_path = 'images/UDOT.png' 
    st.sidebar.image(udot_path , width=265)
    st.sidebar.write("")
    # Display USU in the sidebar
    logo_path = 'images/logo-1.png'  
    st.sidebar.image(logo_path , width=240)

    st.sidebar.markdown(f'[**Singleton Transportation Lab**](https://engineering.usu.edu/cee/research/labs/patrick-singleton/index)')

    expander2 = st.sidebar.expander("**How to use**")
    with expander2:
        expander2.write('''
                First, use the map to select or search for specific location(s) with available data. Second, a particular type of data (recent or historical). Third, select parameters (dates, location and time units). Fourth, view the results as averages, figures, a map, data, or a report. 
                
        ''')
        expander2.write(f'[**PEDAT User Guide**](https://usu-my.sharepoint.com/:b:/g/personal/a02347157_aggies_usu_edu/Echmn2nMMtdItMHwkwxQ6C8BEIagmAfCggPTA2DOrMExuA?e=0QHiQh)')
    
    st.sidebar.markdown("[Step 1: Select location(s)](#step-1-select-data-type-and-location-s)")
    st.subheader('Step 1: Select data type and location(s)')
    st.markdown(
            """<style>
        div[class*="stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 16px;
        }
            </style>
            """, unsafe_allow_html=True)
  
    # Read the data
    df3 = pd.read_json('data/updated_mapdata.json')
    selected_data = df3
    
    # Check if the list is empty before accessing it
    mean_lat = selected_data['LAT'].mean()
    mean_lng = selected_data['LON'].mean()
    df3 = selected_data
    a = df3['ADDRESS'].tolist()
    default_address = [a[1]]
    icon_image = 'images/ts_small.png'
    icon_size = (7, 14)

    # Create a custom TileLayer
    pedat_tiles = folium.TileLayer(
        tiles='https://api.mapbox.com/styles/v1/bashasvari/clhgx1yir00h901q1ecbt9165/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1IjoiYmFzaGFzdmFyaSIsImEiOiJjbGVmaTdtMmIwcXkzM3Jxam9hb2pwZ3BoIn0.JmYank8e3bmQ7RmRiVdTIg',
        attr='PEDAT map',
        name='PEDAT',
        overlay=False,
        control=True
    )
    Satellite  = folium.TileLayer(
        tiles='https://api.mapbox.com/styles/v1/bashasvari/cluvp5mkm000i01og0rbcgwmf/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1IjoiYmFzaGFzdmFyaSIsImEiOiJjbGVmaTdtMmIwcXkzM3Jxam9hb2pwZ3BoIn0.JmYank8e3bmQ7RmRiVdTIg',
        attr='Satellite',
        name='Satellite Map',
        overlay=False,
        control=True
    )
    # Create the map with PEDAT as the default visible layer
    m = folium.Map(location=[mean_lat, mean_lng], zoom_start=8, tiles=None)

    # Add the PEDAT layer and show it by default
    pedat_tiles.add_to(m)
    Satellite.add_to(m)
    # Adding other tile layers but not showing them by default
    folium.TileLayer('OpenStreetMap', name='Open Street Map', overlay=False, control=True).add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark Matter', overlay=False, control=True).add_to(m)

    # Convert dataFrame to GeoJSON format
    geo_data = df3[["LAT", "LON", "ADDRESS"]].copy()

    def feature_from_row(row):
        return {
            "type": "Feature",
            "properties": {"address": row["ADDRESS"]},
            "geometry": {
                "type": "Point",
                "coordinates": [row["LON"], row["LAT"]]
            }
        }

    geo_json = {
        "type": "FeatureCollection",
        "features": geo_data.apply(feature_from_row, axis=1).tolist()
    }

    # Create the invisible GeoJson layer for search functionality
    geo_layer = folium.GeoJson(geo_json ,
    marker=folium.CircleMarker(radius=0.00000000000000000001, color=None ,opacity=7), control=False).add_to(m)
    search = Search(
        layer=geo_layer,
        geom_type="Point",
        placeholder="Search for an address/signals",
        collapsed=True,
        search_label="address"
    ).add_to(m)

    
    df3['Address'] = df3['ADDRESS'].str.replace(r'^\d+\s*--\s*', '', regex=True)
    # Add custom icons
    callback = """
    function (row) {
        var icon, marker, popupContent;
        icon = L.AwesomeMarkers.icon({
            icon: 'fa-traffic-light', 
            
            prefix: 'fa', 
            
        });
        marker = L.marker(new L.LatLng(row[0], row[1]), {icon: icon});
        popupContent = '<b>Signal ID:</b> ' + row[3]  + '<br>' + '<b>Address:</b> ' + row[2];
        marker.bindPopup(popupContent, {maxWidth: 300, minWidth: 150});
        return marker;
    };
    """

    # Prepare your data list, with latitude, longitude, and popup content (address)
    data = list(zip(df3['LAT'], df3['LON'], df3['Address'] , df3['SIGNAL']))

    # Use FastMarkerCluster with the callback
    FastMarkerCluster(data=data, callback=callback , control=False).add_to(m)

    Draw(
        export=False,
        filename="my_data.geojson",
        position="topleft",
        draw_options={
            'circle': False,
            'circlemarker': False,
            'marker': False,
            'polyline': False
        }
    ).add_to(m)

    address = []

    # Fit the map bounds to show all points
    sw = df3[['LAT', 'LON']].min().values.tolist()
    ne = df3[['LAT', 'LON']].max().values.tolist()
    m.fit_bounds([sw, ne]) 

    if 'selected_addresses' not in st.session_state:
        st.session_state.selected_addresses = []

    # Render the map using st_folium
    folium.LayerControl().add_to(m)
    s = st_folium(m, width='80%', height=600, returned_objects=["last_object_clicked", "last_active_drawing"])
    
    # Function to add or update the selected addresses based on map interaction
    def add_selected_address(lat, lng, polygon=None):
        if polygon is not None:
            for index, row in df3.iterrows():
                point = Point(row['LON'], row['LAT'])
                if polygon.contains(point) and row['ADDRESS'] not in st.session_state.selected_addresses:
                    st.session_state.selected_addresses.append(row['ADDRESS'])
        else:
            filtered_df = df3[(df3['LAT'] == lat) & (df3['LON'] == lng)]
            for index, row in filtered_df.iterrows():
                if row['ADDRESS'] not in st.session_state.selected_addresses:
                    st.session_state.selected_addresses.append(row['ADDRESS'])

    # Check if a location or a polygon has been selected on the map
    if s is not None:
        if "last_object_clicked" in s and s["last_object_clicked"] is not None:
            json_obj = s["last_object_clicked"]
            add_selected_address(json_obj["lat"], json_obj["lng"])
        elif "last_active_drawing" in s and s["last_active_drawing"] is not None:
            polygon_coords = s["last_active_drawing"]["geometry"]["coordinates"]
            polygon = Polygon(polygon_coords[0])
            add_selected_address(None, None, polygon)

    selected_signals = []

    # Ensure 'selected_signals' is initialized in session_state
    if 'selected_signals' not in st.session_state:
        st.session_state.selected_signals = []

    # Form for finalizing the selection of addresses
    with st.form("selected_locations_form"):
        if not st.session_state.selected_addresses:
            st.write('')
        else:
            st.write("**Review and select the locations**")

        # List to store the state of each checkbox
        checkbox_states = []

        # Dynamically create a checkbox for each selected address
        for address in st.session_state.selected_addresses:
            checkbox_state = st.checkbox(address, key=address, value=True)
            checkbox_states.append(checkbox_state)

        # Conditionally display the dropdown if any address is selected
        if any(checkbox_states):
            dash = ['Recent data (last 1 year)', 'Historical data (last 5 years)']
            Dash_selected = st.selectbox('**Select data type**', options=dash)
        else:
            # Prompt user if no address is checked
            st.warning("Please select at least one address to proceed.")

        # Submit button for the form
        submitted = st.form_submit_button("Submit")

        if submitted:
            # Filter the selected addresses based on checkboxes that are ticked
            selected_signals = [address for address, checked in zip(st.session_state.selected_addresses, checkbox_states) if checked]
            st.session_state.selected_signals = selected_signals
            
    if st.session_state.selected_signals:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(
                    "selected_signals", "STRING", st.session_state.selected_signals
                )
            ]
        )

        # Use an appropriate SQL query based on the data type selected
        if Dash_selected == 'Recent data (last 1 year)':
            df = client.query(sql_query2, job_config=job_config).to_dataframe()
        else:
            df = client.query(sql_query3, job_config=job_config).to_dataframe()
            df['TIME1'] = pd.to_datetime(df['TIME1'])

        # Extract unique signals and create color maps
        unique_signals = df['SIGNAL'].unique().tolist()
        color_map = create_color_map(unique_signals)

        # Assuming df3 is defined elsewhere and contains 'ADDRESS'
        all_addresses = df3['ADDRESS'].tolist()

        # Filter the list of all_addresses to include only the selected signals
        addresses_to_keep = set(st.session_state.selected_signals).intersection(set(all_addresses))
        all_addresses = list(addresses_to_keep)
        st.markdown(
            """<style>
        div[class*="stColumn"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 32px;
        }
            </style>
            """, unsafe_allow_html=True)

        st.sidebar.markdown("[Step 2: Select parameters](#step-2-select-parameters)")
        st.subheader('Step 2: Select parameters')
        form = st.form("Select parameters")
        
        # Add a calendar widget to select a date range
        start_date = form.date_input('**Start date**', df['TIME1'].min())
        end_date = form.date_input('**End date**', df['TIME1'].max())

        # Add a slider for selecting the aggregation method
        if Dash_selected == 'Recent data (last 1 year)':
            locations = ['All'] + ['Phase ' + str(int(i)) for i in sorted(df[df['ADDRESS'].isin(all_addresses)]['P'].dropna().unique().tolist())]
            #locations = ['All'] + ['Phase ' + str(int(i)) for i in sorted(df['P'].dropna().unique().tolist())]
            location_selected = form.selectbox('**Location unit**', options=locations)
            aggregation_methods = ['Hour', 'Day', 'Week', 'Month', 'Year']
            aggregation_method_selected = form.selectbox('**Time unit**', options=aggregation_methods)      
        else:
            aggregation_methods = ['Day', 'Week', 'Month', 'Year']
            aggregation_method_selected = form.selectbox('**Time unit**', options=aggregation_methods)
            location = ['All']
            location_selected = location[0]
        
        st.markdown(
            """<style>
        div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 16px;
        }
            </style>
            """, unsafe_allow_html=True)

        form.form_submit_button("Submit")

        # Convert the date objects to datetime objects
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        dt_str = start_date.strftime("%b %d, %Y")
        dt_str2 = end_date.strftime("%b %d, %Y")
        start_date2 = pd.Timestamp(start_date).tz_localize('UTC')
        end_date2 = pd.Timestamp(end_date).tz_localize('UTC')
        # Format the metric values
        # Filter the DataFrame based on the selected date range and location
        if Dash_selected == 'Recent data (last 1 year)':
            mask = (df['TIME1'] >= start_date2) & (df['TIME1'] <= end_date2)
        else: 
            mask = (df['TIME1'] >= start_datetime) & (df['TIME1'] <= end_datetime)
            
        if location_selected == 'All':
            filtered_df = df.loc[mask]
        else:
            phase_number = int(location_selected.replace('Phase ', ''))
            mask &= (df['P'] == phase_number)
            filtered_df = df.loc[mask]

        # Calculate the sum of pedestrians based on the filtered DataFrame
        total_pedestrians = filtered_df['PED'].sum()
        num_signals = len(df['ADDRESS'].unique())
        total_pedestrians_formatted = format_metric(total_pedestrians)
        num_signals_formatted = format_metric(num_signals)
        st.sidebar.markdown("[Metrics](#metrics)")
        st.subheader('**Metrics**')
        col1, col2 , col3 , col4= st.columns(4)

        # Display the metric boxes
        col1.metric("**Total pedestrians**", total_pedestrians_formatted)
        col2.metric("**Selected locations**", num_signals_formatted)
        col3.metric("**Start date**" , dt_str)
        col4.metric("**End date**" , dt_str2)


        selected_signals = st.session_state.selected_signals or default_address

        # Averages section
        st.sidebar.markdown("[Averages](#averages)")
        st.subheader('**Averages**')
        with st.expander("Expand"):
            if Dash_selected == 'Recent data (last 1 year)':
                st.subheader('Average daily pedestrian volume, by location')
                fig16, df_agg16 = make_bar_chart4(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected, color_map)
                cv16 = df_agg16.to_csv(index=True)
                st.download_button(
                    label="📥 Download data",
                    data=cv16,
                    file_name="barchart_daily_location.csv",
                    mime='text/csv',
                )
                st.plotly_chart(fig16, use_container_width=True )

                st.subheader('Average hourly pedestrian volume, by hour-of-day, total of all locations')
                fig2, df_agg3 = make_bar_chart(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected)
                cv3 = df_agg3.to_csv(index=True)
                st.download_button(
                    label="📥 Download data",
                    data=cv3,
                    file_name="barcharthourly.csv",
                    mime='text/csv',
                )
                st.plotly_chart(fig2, use_container_width=True )

                st.subheader('Average daily pedestrian volume, by day-of-week, total of all locations')
                fig3 , df_agg4= make_bar_chart2(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected)
                cv4 = df_agg4.to_csv(index=True)
                st.download_button(
                    label="📥 Download data",
                    data=cv4,
                    file_name="barchartdaily.csv",
                    mime='text/csv',
                )
                st.plotly_chart(fig3, use_container_width=True )

                st.subheader('Average daily pedestrian volume, by month-of-year, total of all locations')
                fig8 , df_agg5= make_bar_chart3(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected)
                cv5 = df_agg5.to_csv(index=True)
                st.download_button(
                    label="📥 Download data",
                    data=cv5,
                    file_name="barchartmonthly.csv",
                    mime='text/csv',
                )
                st.plotly_chart(fig8, use_container_width=True )

            else:
                st.subheader('Average daily pedestrian volume, by location')
                fig16, df_agg16 = make_bar_chart4(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected, color_map)
                cv16 = df_agg16.to_csv(index=True)
                st.download_button(
                    label="📥 Download data",
                    data=cv16,
                    file_name="barchart_daily_location.csv",
                    mime='text/csv',
                )
                st.plotly_chart(fig16, use_container_width=True )
            
                st.subheader('Average daily pedestrian volume, by day-of-week, total of all locations')
                fig3 , df_agg4= make_bar_chart2(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected)
                cv4 = df_agg4.to_csv(index=True)
                st.download_button(
                    label="📥 Download data",
                    data=cv4,
                    file_name="barchartdaily.csv",
                    mime='text/csv',
                )
                st.plotly_chart(fig3, use_container_width=True )

                st.subheader('Average daily pedestrian volume, by month-of-year, total of all locations')
                fig8 , df_agg5= make_bar_chart3(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected)
                cv5 = df_agg5.to_csv(index=True)
                st.download_button(
                    label="📥 Download data",
                    data=cv5,
                    file_name="barchartmonthly.csv",
                    mime='text/csv',
                )
                st.plotly_chart(fig8, use_container_width=True )

        # Figures section
        st.sidebar.markdown("[Figures](#figures)")
        st.subheader('**Figures**')
        with st.expander("Expand"):
            if Dash_selected == 'Recent data (last 1 year)':

                st.subheader('Total pedestrian volume, by location')
                fig4 , df_agg1 = make_pie_and_bar_chart(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected ,color_map)
                cv2 = df_agg1.to_csv(index=True)
                st.download_button(
                    label="📥 Download data",
                    data=cv2,
                    file_name="piebarchart.csv",
                    mime='text/csv',
                )
                st.plotly_chart(fig4, use_container_width=True )

                table = make_table(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, Dash_selected)
                pivot_table = table.pivot_table(values='Pedestrian', index='Timestamp', columns='Signal ID', aggfunc='sum')
                cv1 = pivot_table.to_csv(index=True)
                selected_method_lower = aggregation_method_selected.lower()
                st.subheader(f'Time series of pedestrian volume, by {selected_method_lower}, by location')
                st.download_button(
                    label="📥 Download data",
                    data=cv1,
                    file_name="TimeSeries.csv",
                    mime='text/csv',
                )
                fig1 = make_chart(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, Dash_selected, color_map,template='plotly')
                st.plotly_chart(fig1, use_container_width=True )
                table['Signal ID'] = table['Signal ID'].astype(str)
                table['Pedestrian'] = table['Pedestrian'].astype(str)
                table['Signal ID'] = table['Signal ID'].str.replace(',', '.')
                table['Pedestrian'] = table['Pedestrian'].str.replace(',', '.')
                table['Signal ID'] = pd.to_numeric(table['Signal ID'], errors='coerce')
                table['Pedestrian'] = pd.to_numeric(table['Pedestrian'] , errors='coerce')
                grouped = table.groupby('Signal ID')['Pedestrian'].describe().round(0)
                missing_counts = table['Pedestrian'].isna().groupby(table['Signal ID']).sum()
                grouped['Missing Count'] = missing_counts
                DS = grouped.to_csv(index=True)

                # Box Plot
                st.subheader(f'Box plot of pedestrian volume, by {selected_method_lower}, by location')
                signal_ids = table['Signal ID'].unique() 
                fig = go.Figure()
                for signal_id, group in table.groupby('Signal ID'):
                    if signal_id in signal_ids:
                        color = color_map.get(signal_id, '#000000')  # Default to black if signal_id not found
                        fig.add_trace(go.Box(y=group['Pedestrian'], name=f'{signal_id}', 
                                            marker=dict(color=color)))
                # Save the box_plot_data DataFrame to a CSV file
                st.download_button(
                    label="📥 Download data",
                    data=DS,
                    file_name="box_plot_data.csv",
                    mime='text/csv',
                )
                fig.update_layout(yaxis_title='<b>Pedestrian Volume<b>', xaxis_title='<b>Location<b>')
                fig.update_layout(xaxis=dict(title='<b>Location<b>', type='category', tickmode='array', tickvals=signal_ids,
                                            ticktext=[str(signal_id) for signal_id in signal_ids]))
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                fig7 = copy.deepcopy(fig)
                fig7.update_layout(autosize=False, width=920, height=520 , showlegend=False)
                fig7.update_layout(template='plotly')
                fig7.write_image("fig6.png")
            else:
                st.subheader('Total pedestrian volume, by location')
                # Add a pie chart to show pedestrian volume by signal
                fig4 , df_agg1 = make_pie_and_bar_chart(df, selected_signals, start_datetime, end_datetime, location_selected, Dash_selected ,color_map)
                cv2 = df_agg1.to_csv(index=True)
                st.download_button(
                    label="📥 Download data",
                    data=cv2,
                    file_name="piebarchart.csv",
                    mime='text/csv',
                )
                st.plotly_chart(fig4, use_container_width=True )
                table = make_table(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, Dash_selected)
                pivot_table = table.pivot_table(values='Pedestrian', index='Timestamp', columns='Signal ID', aggfunc='sum')
                cv1 = pivot_table.to_csv(index=True)
                selected_method_lower = aggregation_method_selected.lower()
                st.subheader(f'Time series of pedestrian volume, by {selected_method_lower}, by location')
                st.download_button(
                    label="📥 Download data",
                    data=cv1,
                    file_name="TimeSeries.csv",
                    mime='text/csv',
                )
                fig1 = make_chart(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, Dash_selected, color_map,template='plotly')
                st.plotly_chart(fig1, use_container_width=True )
                table['Signal ID'] = table['Signal ID'].astype(str)
                table['Pedestrian'] = table['Pedestrian'].astype(str)
                table['Signal ID'] = table['Signal ID'].str.replace(',', '.')
                table['Pedestrian'] = table['Pedestrian'].str.replace(',', '.')
                table['Signal ID'] = pd.to_numeric(table['Signal ID'], errors='coerce')
                table['Pedestrian'] = pd.to_numeric(table['Pedestrian'] , errors='coerce')
                grouped = table.groupby('Signal ID')['Pedestrian'].describe().round(0)
                missing_counts = table['Pedestrian'].isna().groupby(table['Signal ID']).sum()
                grouped['Missing Count'] = missing_counts
                DS = grouped.to_csv(index=True)
                
                # Box Plot
                st.subheader(f'Box plot of pedestrian volume, by {selected_method_lower}, by location')
                signal_ids = table['Signal ID'].unique() 
                fig = go.Figure()
                for signal_id, group in table.groupby('Signal ID'):
                    if signal_id in signal_ids:
                        color = color_map.get(signal_id, '#000000')  # Default to black if signal_id not found
                        fig.add_trace(go.Box(y=group['Pedestrian'], name=f'{signal_id}', 
                                            marker=dict(color=color)))

                # Save the box_plot_data DataFrame to a CSV file
                st.download_button(
                    label="📥 Download data",
                    data=DS,
                    file_name="box_plot_data.csv",
                    mime='text/csv',
                )
                fig.update_layout(yaxis_title='<b>Pedestrian Volume<b>', xaxis_title='<b>Location<b>')
                fig.update_layout(xaxis=dict(title='<b>Location<b>', type='category', tickmode='array', tickvals=signal_ids,
                                            ticktext=[str(signal_id) for signal_id in signal_ids]))
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                fig7 = copy.deepcopy(fig)
                fig7.update_layout(autosize=False, width=920, height=520 , showlegend=False)
                fig7.update_layout(template='plotly')
                fig7.write_image("fig6.png")

        # Map section
        st.sidebar.markdown("[Map](#map)")
        st.subheader('**Map**')  
        with st.expander("Expand"):
            make_map2(filtered_df, selected_signals ,aggregation_method_selected , location_selected, Dash_selected)
            
            #keplergl_static(map_2)
            
        
        # Data section
        st.sidebar.markdown("[Data](#data)")
        st.subheader('**Data**')
        with st.expander("Expand"): 
            # Filter your data based on the selected date range
            st.subheader(f'Data, by {selected_method_lower}, by location')        
            # Display the filtered data in a table
            table = make_table(df, selected_signals, start_datetime, end_datetime, aggregation_method_selected, location_selected, Dash_selected)
            cc = table.to_csv(index=False)
            st.download_button(
                label="📥 Download",
                data=cc,
                file_name="RawData.csv",
                mime='text/csv',
            )
            st.dataframe(table , use_container_width=True)
            # CSS to inject contained in a string
            hide_dataframe_row_index = """
                        <style>
                        .row_heading.level0 {display:none}
                        .blank {display:none}
                        </style>
                        """

            # Descriptive  statistics
            st.subheader(f'Descriptive  statistics, by {selected_method_lower}, by location')
            st.download_button(
            label="📥 Download",
            data=DS,
            file_name="Descriptive Stat.csv",
            mime='text/csv',)
            st.dataframe(grouped , use_container_width=True)
        
        # Report section
        class PDF(FPDF):

            def __init__(self):
                super().__init__(orientation='L')
                self.page_width = 8.5 * 72  # Letter page width in points (1 inch = 72 points)
                self.page_height = 11 * 72  # Letter page height in points
                self.l_margin = 0.5 * 72    # Left margin in points
                self.r_margin = 0.5 * 72    # Right margin in points
                #self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True) # Add DejaVu font

            def header(self):
                # Select Arial bold 15
                self.set_font('Arial', 'I', 8)
                # Move to the right
                self.cell(self.l_margin)
                # Calculate the X position of the center of the available page width
                center_x = self.w / 2
                # Calculate the X position of the center of the title
                title_x = center_x - (self.get_string_width('Pedestrian Volume Data Report') / 2)
                # Line break
                self.ln(20)
                # Check if we are on the first page
                if self.page_no() == 1:
                    # Add the logo to the first page
                    self.image('images/logo.png', x=10, y=10, w=33/2)

            def footer(self):
                # Position at 1.5 cm from bottom
                self.set_y(-15)
                # Arial italic 8
                self.set_font('Arial', 'I', 8)
                # Title
                self.cell(0, 10, 'Pedestrian Volume Report', 0, 0, 'L')
                # Ensure the footer is placed at 1.5 cm from the bottom
                self.set_y(-15)
                # Set the font for the footer: Arial italic, 8
                self.set_font('Arial', 'I', 8)

                # Footer content
                # Set the timezone to 'America/Denver' for Utah
                user_timezone = pytz.timezone('America/Denver')
                
                # Get the current time in the Utah timezone
                now = datetime.now(user_timezone)
                
                # Format the date and time strings
                date_str = now.strftime('%Y-%m-%d')
                time_str = now.strftime('%H:%M:%S')
                
                # Create the footer string
                footer_str = 'Report generated on {} at {}'.format(date_str, time_str)

                # Add the formatted date and time to the footer, centered
                self.cell(0, 10, footer_str, 0, 0, 'C')

                # Add the page number
                self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'R')


            def chapter_title(self, title):
                # Arial 12
                self.set_font('Arial', '', 12)
                # Background color
                self.set_fill_color(200, 220, 255)
                # Title
                self.cell(0, 6, 'Chapter %d : %s' % (num, label), 0, 1, 'L', 1)
                # Line break
                self.ln(4)

            def chapter_body(self, body):
                # Read text file
                with open(name, 'rb') as fh:
                    txt = fh.read().decode('latin-1')
                # Times 12
                self.set_font('Times', '', 12)
                # Output justified text
                self.multi_cell(0, 5, txt)
                # Line break
                self.ln()
                # Mention in italics
                self.set_font('', 'I')
                self.cell(0, 5, '(end of excerpt)')


        def generate_report(selected_signals, start_datetime, end_datetime, location_selected, aggregation_method_selected, Dash_selected):
                # Generate plots
                pdf = PDF()

                # Add a page
                pdf.add_page()
                

                # Add the logo to the first page after the first add_page() call
                pdf.image('images/logo.png', x=10, y=10, w=33/2)

                # Set title
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 15, 'Pedestrian Volume in Utah', ln=True, align='C')

                # Add a sample text
                pdf.set_font('Arial', '', 12)
                pdf.multi_cell(0, 6, "This report provides data and visualizations of pedestrian volume at various locations in Utah. Pedestrian volume is an estimate of pedestrian crossing volume at an intersection, currently based on pedestrian push-button presses at traffic signals.")


                # Add selected signals
                pdf.set_font('Arial', 'B', 14)
                pdf.ln(5) 
                pdf.cell(1, 10, 'Selected location(s):', ln=True, align='L')
                pdf.set_font('Arial', '', 12)
                for signal in selected_signals:
                    pdf.cell(1, 6, signal, ln=True, align='L')

                # Add duration
                pdf.set_font('Arial', 'B', 14)
                pdf.ln(5)
                pdf.cell(0, 10, 'Selected parameters:', ln=True)
                pdf.set_font('Arial', '', 12)
                start_date = start_datetime.strftime('%Y-%m-%d') 
                end_date = end_datetime.strftime('%Y-%m-%d')
                pdf.cell(0, 6, f'Start date: {start_date}', ln=True)
                pdf.cell(0, 6, f'End date: {end_date}', ln=True)
                # Display the selected location unit
                pdf.cell(0, 6, f'Location unit: {location_selected}', ln=True)
                # Display the selected time unit
                pdf.cell(0, 6, f'Time unit: {aggregation_method_selected}', ln=True)
                
                # Define figure indices based on condition
                figure_indices = [7, 3, 4, 5, 2, 1, 6] if Dash_selected == 'Recent data (last 1 year)' else [7, 4, 5, 2, 1, 6]

                # Define subtitles based on the condition
                if Dash_selected == 'Recent data (last 1 year)':
                    subtitles = [
                        'Average daily pedestrian volume, by location',
                        'Average hourly pedestrian volume, by hour-of-day, total of all locations',
                        'Average daily pedestrian volume, by day-of-week, total of all locations',
                        'Average daily pedestrian volume, by month-of-year, total of all locations',
                        'Total pedestrian volume, by location',
                        f'Time series of pedestrian volume, by {selected_method_lower}, by location',
                        f'Box plot of pedestrian volume, by {selected_method_lower}, by location',
                    ]
                else:
                    subtitles = [
                        'Average daily pedestrian volume, by location',
                        'Average daily pedestrian volume, by day-of-week, total of all locations',
                        'Average daily pedestrian volume, by month-of-year, total of all locations',
                        'Total pedestrian volume, by location',
                        f'Time series of pedestrian volume, by {selected_method_lower}, by location',
                        f'Box plot of pedestrian volume, by {selected_method_lower}, by location',
                    ]

                # Iterate through figures and subtitles
                for i, subtitle in zip(figure_indices, subtitles):
                    # Add a page for each figure
                    pdf.add_page()
                    
                    # Set subtitle for the image
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, subtitle, ln=True)
                    
                    # Calculate the y position for the image, considering the space taken by the subtitle
                    y_position = pdf.get_y()
                    
                    # Add image to the page, fit to available width
                    image_path = f'fig{i}.png'
                    pdf.image(image_path, x=36, y=y_position, w=233)

                # Save the PDF to a BytesIO object
                pdf_buffer = BytesIO()
                pdf.output(pdf_buffer, "F")
                pdf_bytes = pdf_buffer.getvalue()

                # Create a download link for the PDF
                b64_pdf = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:file/pdf;base64,{b64_pdf}" download="report.pdf">Click here to download the PDF report</a>'
                st.markdown(href, unsafe_allow_html=True)

                pdf_buffer.close()

                files_to_remove = ["fig1.png", "fig2.png", "fig4.png", "fig5.png" ,  "fig6.png" , "fig7.png"]

                if os.path.exists("fig3.png"):
                    os.remove("fig3.png")

                for file in files_to_remove:
                    if os.path.exists(file):
                        os.remove(file)

        st.sidebar.markdown("[Report](#report)")
        st.subheader('Report')
        with st.expander("Expand"):
            if st.button('Generate PDF report'):
                    generate_report(selected_signals, start_datetime, end_datetime, location_selected, aggregation_method_selected, Dash_selected)

        st.sidebar.markdown(
            """<style>
        div[class*="stDate"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 16px;
        }
            </style>
            """, unsafe_allow_html=True)
        st.markdown(
            """<style>
        div[class*="stExpander"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 16px;
        }
            </style>
            """, unsafe_allow_html=True)

        expander = st.sidebar.expander("**Notes**")
        with expander:
                expander.write('''
                        "Pedestrian volume" is an estimate of pedestrian volume, specifically the estimated number of pedestrian crossings at an intersection. These estimated pedestrian volumes are based on pedestrian push-button data, obtained via high-resolution traffic signal controller log data from the Utah Department of Transportation's [Automated Traffic Signal Performance Measures System (ATSPM)](https://udottraffic.utah.gov/atspm/) system. [Research](https://rosap.ntl.bts.gov/view/dot/54924) conducted by the Singleton Transportation Lab at Utah State University has validated the use of pedestrian traffic signal data as a reasonably-accurate estimate of pedestrian volumes in Utah. This website was developed by the [Singleton Transportation Lab](https://engineering.usu.edu/cee/research/labs/patrick-singleton/index) in coordination and funded by the Utah Department of Transportation. 
                ''')
        hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
        st.markdown(hide_menu_style, unsafe_allow_html=True)

    
if __name__ == '__main__':
    main()
