#!/usr/bin/env python3
"""
HM Land Registry Price Paid Data Dashboard
Clean, working implementation

Requirements:
    pip install streamlit pandas plotly requests
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from datetime import datetime

st.set_page_config(
    page_title="UK Property Dashboard",
    page_icon="🏠",
    layout="wide"
)

# Constants
PPD_COLUMNS = [
    'transaction_id', 'price', 'date_of_transfer', 'postcode', 'property_type',
    'old_new', 'duration', 'paon', 'saon', 'street', 'locality',
    'town_city', 'district', 'county', 'ppd_category_type', 'record_status'
]

PROPERTY_TYPE_MAP = {
    'D': 'Detached', 'S': 'Semi-Detached', 'T': 'Terraced',
    'F': 'Flats/Maisonettes', 'O': 'Other'
}

@st.cache_data
def load_data():
    """Load and clean PPD data"""
    data_path = os.path.join('data_up', 'data_up.csv')
    
    if not os.path.exists(data_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(data_path, names=PPD_COLUMNS, parse_dates=['date_of_transfer'])
        
        # Clean data
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price', 'date_of_transfer'])
        df = df[df['price'] > 0]
        
        # Clean postcode
        df['postcode'] = df['postcode'].str.upper().str.strip()
        df = df[df['postcode'].str.len() >= 5]
        
        # Extract postcode area
        df['postcode_area'] = df['postcode'].str.extract(r'^([A-Z]{1,2}\d{1,2})')
        
        # Map property types
        df['property_type_desc'] = df['property_type'].map(PROPERTY_TYPE_MAP)
        
        # Create address
        df['full_address'] = (
            df['paon'].fillna('').astype(str) + ' ' +
            df['street'].fillna('').astype(str) + ' ' +
            df['town_city'].fillna('').astype(str)
        ).str.strip()
        
        # Add time columns
        df['year'] = df['date_of_transfer'].dt.year
        df['month'] = df['date_of_transfer'].dt.month
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def filter_data(df, location, property_type, min_price, max_price, date_from, date_to):
    """Filter data based on criteria"""
    filtered = df.copy()
    
    if location:
        mask = (
            filtered['postcode'].str.contains(location, case=False, na=False) |
            filtered['town_city'].str.contains(location, case=False, na=False) |
            filtered['county'].str.contains(location, case=False, na=False) |
            filtered['district'].str.contains(location, case=False, na=False)
        )
        filtered = filtered[mask]
    
    if property_type != 'All':
        filtered = filtered[filtered['property_type'] == property_type]
    
    if min_price > 0:
        filtered = filtered[filtered['price'] >= min_price]
    
    if max_price > 0:
        filtered = filtered[filtered['price'] <= max_price]
    
    filtered = filtered[
        (filtered['date_of_transfer'] >= pd.Timestamp(date_from)) &
        (filtered['date_of_transfer'] <= pd.Timestamp(date_to))
    ]
    
    return filtered

def create_histogram(df):
    """Create price histogram"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['price'],
        nbinsx=30,
        marker_color='lightblue',
        opacity=0.7
    ))
    
    mean_price = df['price'].mean()
    fig.add_vline(x=mean_price, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: £{mean_price:,.0f}")
    
    fig.update_layout(
        title="Price Distribution",
        xaxis_title="Price (£)",
        yaxis_title="Count",
        height=400
    )
    
    return fig

@st.cache_data
def get_postcode_coords(postcode_area):
    """Get coordinates for postcode area"""
    try:
        response = requests.get(f"https://api.postcodes.io/postcodes/{postcode_area}1AA", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data['result']['latitude'], data['result']['longitude']
    except:
        pass
    return 54.5, -2.0  # UK center fallback

def create_map(df, location):
    """Create property distribution map"""
    if df.empty:
        return go.Figure()
    
    # Aggregate by postcode area
    area_stats = df.groupby('postcode_area').agg({
        'price': ['mean', 'count']
    }).round(0)
    
    area_stats.columns = ['mean_price', 'count']
    area_stats = area_stats.reset_index()
    
    # Get coordinates
    coords = []
    for area in area_stats['postcode_area']:
        lat, lon = get_postcode_coords(area)
        coords.append({'postcode_area': area, 'lat': lat, 'lon': lon})
    
    coords_df = pd.DataFrame(coords)
    area_stats = area_stats.merge(coords_df, on='postcode_area')
    
    # Create map
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=area_stats['lat'],
        lon=area_stats['lon'],
        mode='markers',
        marker=dict(
            size=area_stats['count'] / area_stats['count'].max() * 30 + 10,
            color=area_stats['count'],
            colorscale='Viridis',
            opacity=0.8
        ),
        text=area_stats['postcode_area'],
        hovertemplate='%{text}<br>Properties: %{marker.color}<br>Avg: £%{customdata:,.0f}<extra></extra>',
        customdata=area_stats['mean_price']
    ))
    
    center_lat = area_stats['lat'].mean()
    center_lon = area_stats['lon'].mean()
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=8
        ),
        title=f"Property Distribution: {location}",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def main():
    st.title("UK Property Dashboard")
    
    # Check data
    if not os.path.exists('data_up/data_up.csv'):
        st.error("Place PPD CSV file at: data_up/data_up.csv")
        return
    
    # Load data
    df = load_data()
    if df.empty:
        st.error("No data loaded")
        return
    
    st.success(f"Loaded {len(df):,} properties")
    
    # Sidebar filters
    with st.sidebar:
        st.header("Search Filters")
        
        location = st.text_input("Location", placeholder="London, SW1A, etc.")
        
        property_type = st.selectbox(
            "Property Type",
            ['All'] + list(PROPERTY_TYPE_MAP.keys()),
            format_func=lambda x: 'All' if x == 'All' else f"{x} - {PROPERTY_TYPE_MAP[x]}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min Price", min_value=0, value=0, step=10000)
        with col2:
            max_price = st.number_input("Max Price", min_value=0, value=0, step=10000)
        
        min_date = df['date_of_transfer'].min().date()
        max_date = df['date_of_transfer'].max().date()
        
        date_from = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
        date_to = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)
        
        search_btn = st.button("Search", type="primary")
        
        if search_btn and not location:
            st.error("Enter location")
    
    # Results
    if search_btn and location:
        filtered_df = filter_data(df, location, property_type, min_price, max_price, date_from, date_to)
        
        if filtered_df.empty:
            st.warning("No properties found")
            return
        
        st.success(f"Found {len(filtered_df):,} properties")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Properties", "Analytics", "Map", "Trends"])
        
        with tab1:
            st.subheader("Property Listings")
            
            display_df = filtered_df[['price', 'date_of_transfer', 'full_address', 'property_type_desc', 'postcode']].copy()
            display_df['price'] = display_df['price'].apply(lambda x: f"£{x:,}")
            display_df['date_of_transfer'] = display_df['date_of_transfer'].dt.strftime('%Y-%m-%d')
            display_df.columns = ['Price', 'Date', 'Address', 'Type', 'Postcode']
            
            st.dataframe(display_df, use_container_width=True, height=400)
        
        with tab2:
            st.subheader("Price Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Properties", f"{len(filtered_df):,}")
            with col2:
                st.metric("Mean Price", f"£{filtered_df['price'].mean():,.0f}")
            with col3:
                st.metric("Median Price", f"£{filtered_df['price'].median():,.0f}")
            with col4:
                st.metric("Max Price", f"£{filtered_df['price'].max():,.0f}")
            
            fig_hist = create_histogram(filtered_df)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            if not filtered_df['property_type_desc'].isna().all():
                type_counts = filtered_df['property_type_desc'].value_counts()
                fig_pie = px.pie(values=type_counts.values, names=type_counts.index, title="Property Types")
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab3:
            st.subheader("Geographic Distribution")
            
            fig_map = create_map(filtered_df, location)
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Area summary
            area_summary = filtered_df.groupby('postcode_area').agg({
                'price': ['mean', 'count']
            }).round(0)
            area_summary.columns = ['Avg Price', 'Count']
            area_summary = area_summary.sort_values('Count', ascending=False)
            
            st.subheader("Postcode Areas")
            st.dataframe(area_summary.head(20), use_container_width=True)
        
        with tab4:
            st.subheader("Market Trends")
            
            monthly_data = filtered_df.groupby(filtered_df['date_of_transfer'].dt.to_period('M')).agg({
                'price': ['mean', 'count']
            }).round(0)
            monthly_data.columns = ['avg_price', 'count']
            monthly_data.index = monthly_data.index.astype(str)
            monthly_data = monthly_data.reset_index()
            
            if len(monthly_data) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly_data['date_of_transfer'],
                    y=monthly_data['avg_price'],
                    mode='lines+markers',
                    name='Avg Price',
                    yaxis='y'
                ))
                fig.add_trace(go.Bar(
                    x=monthly_data['date_of_transfer'],
                    y=monthly_data['count'],
                    name='Count',
                    opacity=0.6,
                    yaxis='y2'
                ))
                fig.update_layout(
                    title="Monthly Trends",
                    yaxis=dict(title="Price", side="left"),
                    yaxis2=dict(title="Count", side="right", overlaying="y"),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            yearly_stats = filtered_df.groupby('year').agg({
                'price': ['mean', 'median', 'count']
            }).round(0)
            yearly_stats.columns = ['Mean', 'Median', 'Count']
            
            st.subheader("Yearly Summary")
            st.dataframe(yearly_stats, use_container_width=True)

if __name__ == "__main__":
    main()