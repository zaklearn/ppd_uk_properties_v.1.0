#!/usr/bin/env python3
"""
HM Land Registry Price Paid Data (PPD) Dashboard
Automatically loads CSV from data_up folder and provides interactive property analysis

Requirements:
    pip install streamlit pandas plotly
    
Setup:
    Create folder: data_up/
    Place PPD CSV file as: data_up/data_up.csv
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from typing import Dict, List

st.set_page_config(
    page_title="PPD Property Dashboard",
    page_icon="🏠",
    layout="wide"
)

# PPD Column definitions (no headers in CSV files)
PPD_COLUMNS = [
    'Transaction_ID',
    'Price',
    'Date_of_Transfer', 
    'Postcode',
    'Property_Type',
    'Old_New',
    'Duration',
    'PAON',
    'SAON', 
    'Street',
    'Locality',
    'Town_City',
    'District',
    'County',
    'PPD_Category_Type',
    'Record_Status'
]

PROPERTY_TYPE_MAP = {
    'D': 'Detached',
    'S': 'Semi-Detached', 
    'T': 'Terraced',
    'F': 'Flats/Maisonettes',
    'O': 'Other'
}

OLD_NEW_MAP = {'Y': 'New Build', 'N': 'Existing'}
DURATION_MAP = {'F': 'Freehold', 'L': 'Leasehold'}
CATEGORY_MAP = {'A': 'Standard', 'B': 'Additional'}

@st.cache_data
def load_ppd_data() -> pd.DataFrame:
    """Load PPD data from data_up folder"""
    data_path = os.path.join('data_up', 'data_up.csv')
    
    if not os.path.exists(data_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(data_path, names=PPD_COLUMNS, parse_dates=['Date_of_Transfer'])
        
        # Map coded values to descriptions
        df['Property_Type_Desc'] = df['Property_Type'].map(PROPERTY_TYPE_MAP)
        df['Old_New_Desc'] = df['Old_New'].map(OLD_NEW_MAP) 
        df['Duration_Desc'] = df['Duration'].map(DURATION_MAP)
        df['Category_Desc'] = df['PPD_Category_Type'].map(CATEGORY_MAP)
        
        # Create full address
        df['Full_Address'] = (
            df['PAON'].fillna('') + ' ' + 
            df['SAON'].fillna('') + ' ' +
            df['Street'].fillna('') + ', ' +
            df['Locality'].fillna('') + ', ' +
            df['Town_City'].fillna('') + ', ' +
            df['County'].fillna('')
        ).str.replace(r'\s+', ' ', regex=True).str.strip(', ')
        
        # Add derived columns
        df['Year'] = df['Date_of_Transfer'].dt.year
        df['Month'] = df['Date_of_Transfer'].dt.month
        df['Price_Band'] = pd.cut(df['Price'], 
                                 bins=[0, 200000, 400000, 700000, 1500000, float('inf')],
                                 labels=['Under £200k', '£200k-400k', '£400k-700k', '£700k-1.5M', 'Over £1.5M'])
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def filter_data(df: pd.DataFrame, location: str, property_type: str, 
                min_price: int, max_price: int, date_range: tuple) -> pd.DataFrame:
    """Apply filters to PPD data"""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    # Location filter
    if location:
        location_mask = (
            filtered['Postcode'].str.contains(location, case=False, na=False) |
            filtered['Town_City'].str.contains(location, case=False, na=False) |
            filtered['County'].str.contains(location, case=False, na=False) |
            filtered['District'].str.contains(location, case=False, na=False) |
            filtered['Street'].str.contains(location, case=False, na=False)
        )
        filtered = filtered[location_mask]
    
    # Property type filter
    if property_type != 'All':
        filtered = filtered[filtered['Property_Type'] == property_type]
    
    # Price filter
    if min_price > 0:
        filtered = filtered[filtered['Price'] >= min_price]
    if max_price > 0:
        filtered = filtered[filtered['Price'] <= max_price]
    
    # Date filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered['Date_of_Transfer'] >= pd.Timestamp(start_date)) &
            (filtered['Date_of_Transfer'] <= pd.Timestamp(end_date))
        ]
    
    return filtered

def calculate_stats(df: pd.DataFrame) -> Dict:
    """Calculate summary statistics"""
    if df.empty:
        return {}
    
    return {
        'total_transactions': len(df),
        'mean_price': df['Price'].mean(),
        'median_price': df['Price'].median(),
        'min_price': df['Price'].min(),
        'max_price': df['Price'].max(),
        'std_price': df['Price'].std(),
        'date_range': f"{df['Date_of_Transfer'].min().date()} to {df['Date_of_Transfer'].max().date()}"
    }

def create_price_chart(df: pd.DataFrame) -> go.Figure:
    """Create price distribution chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['Price'],
        nbinsx=50,
        name='Price Distribution',
        marker_color='lightblue'
    ))
    
    # Add mean/median lines
    mean_price = df['Price'].mean()
    median_price = df['Price'].median()
    
    fig.add_vline(x=mean_price, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: £{mean_price:,.0f}")
    fig.add_vline(x=median_price, line_dash="dash", line_color="green",
                  annotation_text=f"Median: £{median_price:,.0f}")
    
    fig.update_layout(
        title="Property Price Distribution",
        xaxis_title="Price (£)",
        yaxis_title="Number of Properties",
        height=400
    )
    
    return fig

def create_time_series(df: pd.DataFrame) -> go.Figure:
    """Create time series of prices and volumes"""
    monthly_data = df.groupby(df['Date_of_Transfer'].dt.to_period('M')).agg({
        'Price': ['mean', 'median', 'count']
    }).round(2)
    
    monthly_data.columns = ['mean_price', 'median_price', 'transaction_count']
    monthly_data.index = monthly_data.index.astype(str)
    monthly_data = monthly_data.reset_index()
    
    fig = go.Figure()
    
    # Price trends
    fig.add_trace(go.Scatter(
        x=monthly_data['Date_of_Transfer'],
        y=monthly_data['mean_price'],
        mode='lines',
        name='Mean Price',
        line=dict(color='blue'),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_data['Date_of_Transfer'],
        y=monthly_data['median_price'],
        mode='lines',
        name='Median Price', 
        line=dict(color='green'),
        yaxis='y'
    ))
    
    # Transaction volume
    fig.add_trace(go.Bar(
        x=monthly_data['Date_of_Transfer'],
        y=monthly_data['transaction_count'],
        name='Transaction Count',
        marker_color='orange',
        opacity=0.3,
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Property Market Trends Over Time",
        xaxis_title="Date",
        yaxis=dict(title="Price (£)", side="left"),
        yaxis2=dict(title="Transaction Count", side="right", overlaying="y"),
        height=500
    )
    
    return fig

def display_properties_table(df: pd.DataFrame):
    """Display interactive properties table"""
    st.markdown("### 🏠 Property Listings")
    
    if df.empty:
        st.warning("No properties found matching your criteria")
        return
    
    # Select relevant columns for display
    display_columns = [
        'Price', 'Date_of_Transfer', 'Full_Address', 'Property_Type_Desc', 
        'Old_New_Desc', 'Duration_Desc', 'Postcode'
    ]
    
    # Format the data for better display
    display_df = df[display_columns].copy()
    display_df['Price'] = display_df['Price'].apply(lambda x: f"£{x:,.0f}")
    display_df['Date_of_Transfer'] = display_df['Date_of_Transfer'].dt.strftime('%Y-%m-%d')
    
    # Rename columns for better readability
    display_df.columns = [
        'Price', 'Sale Date', 'Address', 'Property Type', 
        'Age', 'Tenure', 'Postcode'
    ]
    
    # Display with pagination
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )
    
    st.info(f"Showing {len(df):,} properties matching your filters")

def main():
    st.title("🏠 Price Paid Data Property Dashboard")
    st.markdown("Analyzing UK property transactions from HM Land Registry")
    
    # Check if data file exists
    data_path = os.path.join('data_up', 'data_up.csv')
    if not os.path.exists('data_up'):
        st.error("❌ 'data_up' folder not found. Please create it in the application root directory.")
        st.markdown("**Setup Instructions:**")
        st.code("""
1. Create folder: data_up/
2. Download PPD CSV from: landregistry.data.gov.uk
3. Rename file to: data_up.csv
4. Place in: data_up/data_up.csv
        """)
        return
    
    if not os.path.exists(data_path):
        st.error("❌ data_up.csv not found in data_up folder")
        st.markdown("**Required:** Place PPD CSV file as `data_up/data_up.csv`")
        return
    
    # Load data
    with st.spinner("Loading property data..."):
        df = load_ppd_data()
    
    if df.empty:
        st.error("Could not load data from data_up.csv")
        return
    
    original_count = len(df)
    st.success(f"Loaded {original_count:,} property transactions")
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔧 Property Filters")
        
        location = st.text_input(
            "📍 Location",
            placeholder="London, Manchester, SW1A, etc.",
            help="Search by postcode, town, county, or street name"
        )
        
        property_type = st.selectbox(
            "🏠 Property Type",
            ['All'] + list(PROPERTY_TYPE_MAP.keys()),
            format_func=lambda x: 'All Types' if x == 'All' else f"{x} - {PROPERTY_TYPE_MAP.get(x, x)}"
        )
        
        st.markdown("**💰 Price Range**")
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min (£)", min_value=0, value=0, step=10000)
        with col2:
            max_price = st.number_input("Max (£)", min_value=0, value=0, step=10000)
        
        # Date range filter
        min_date = df['Date_of_Transfer'].min().date()
        max_date = df['Date_of_Transfer'].max().date()
        
        date_range = st.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        st.markdown("---")
        st.markdown("**Data Info:**")
        st.info(f"📊 Total Records: {original_count:,}")
        st.info(f"📅 Date Range: {min_date} to {max_date}")
    
    # Apply filters automatically
    filtered_df = filter_data(df, location, property_type, min_price, max_price, date_range)
    
    if filtered_df.empty:
        st.warning("❌ No properties found matching your filter criteria")
        st.markdown("**Try:**")
        st.markdown("- Broadening your location search")
        st.markdown("- Adjusting price range")
        st.markdown("- Changing property type to 'All'")
        st.markdown("- Expanding date range")
        return
    
    # Show filter results
    if len(filtered_df) != original_count:
        st.info(f"🔍 Showing {len(filtered_df):,} properties (filtered from {original_count:,} total)")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Properties", "📊 Overview", "📈 Analytics", "🗺️ Geographic", "📊 Export"
    ])
    
    with tab1:
        display_properties_table(filtered_df)
    
    with tab2:
        stats = calculate_stats(filtered_df)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{stats['total_transactions']:,}")
        with col2:
            st.metric("Mean Price", f"£{stats['mean_price']:,.0f}")
        with col3:
            st.metric("Median Price", f"£{stats['median_price']:,.0f}")
        with col4:
            price_range = stats['max_price'] - stats['min_price']
            st.metric("Price Range", f"£{price_range:,.0f}")
        
        # Price distribution
        fig_price = create_price_chart(filtered_df)
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Property type breakdown
        if not filtered_df['Property_Type_Desc'].isna().all():
            type_counts = filtered_df['Property_Type_Desc'].value_counts()
            fig_types = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Property Types Distribution"
            )
            st.plotly_chart(fig_types, use_container_width=True)
    
    with tab3:
        # Time series analysis
        if len(filtered_df) > 1:
            fig_time = create_time_series(filtered_df)
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Yearly breakdown
        yearly_stats = filtered_df.groupby('Year').agg({
            'Price': ['mean', 'median', 'count'],
            'Property_Type_Desc': lambda x: x.mode().iloc[0] if not x.empty else 'Mixed'
        }).round(0)
        yearly_stats.columns = ['Mean Price', 'Median Price', 'Transactions', 'Top Property Type']
        
        st.markdown("### 📅 Yearly Summary")
        st.dataframe(yearly_stats, use_container_width=True)
        
        # Monthly trends
        monthly_avg = filtered_df.groupby(['Year', 'Month'])['Price'].mean().reset_index()
        monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(day=1))
        
        if len(monthly_avg) > 1:
            fig_monthly = px.line(
                monthly_avg, 
                x='Date', 
                y='Price',
                title="Monthly Average Prices"
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab4:
        # Geographic analysis
        st.markdown("### 🗺️ Geographic Distribution")
        
        # Top locations by transaction volume
        location_stats = filtered_df.groupby('Town_City').agg({
            'Price': ['mean', 'count'],
            'Transaction_ID': 'count'
        }).round(0)
        location_stats.columns = ['Mean Price', 'Price Count', 'Total Transactions']
        location_stats = location_stats.sort_values('Total Transactions', ascending=False).head(20)
        
        st.markdown("#### 🏙️ Top Cities/Towns")
        st.dataframe(location_stats, use_container_width=True)
        
        # County analysis
        if not filtered_df['County'].isna().all():
            county_stats = filtered_df.groupby('County')['Price'].agg(['mean', 'count']).round(0)
            county_stats.columns = ['Mean Price', 'Transactions']
            county_stats = county_stats.sort_values('Transactions', ascending=False)
            
            fig_county = px.bar(
                x=county_stats.index[:15],
                y=county_stats['Transactions'][:15],
                title="Top 15 Counties by Transaction Volume",
                labels={'x': 'County', 'y': 'Number of Transactions'}
            )
            st.plotly_chart(fig_county, use_container_width=True)
        
        # Postcode area analysis
        if location:
            postcode_stats = filtered_df.groupby('Postcode')['Price'].agg(['mean', 'count']).round(0)
            postcode_stats.columns = ['Mean Price', 'Transactions']
            postcode_stats = postcode_stats.sort_values('Mean Price', ascending=False).head(20)
            
            st.markdown("#### 📮 Top Postcodes by Price")
            st.dataframe(postcode_stats, use_container_width=True)
    
    with tab5:
        # Export options
        st.markdown("### 📥 Export Filtered Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📊 Download Filtered CSV",
                data=csv,
                file_name=f"ppd_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary report
            summary = {
                'analysis_date': datetime.now().isoformat(),
                'total_records': len(filtered_df),
                'original_records': original_count,
                'statistics': stats,
                'filters_applied': {
                    'location': location if location else None,
                    'property_type': property_type if property_type != 'All' else None,
                    'min_price': min_price if min_price > 0 else None,
                    'max_price': max_price if max_price > 0 else None,
                    'date_range': f"{date_range[0]} to {date_range[1]}" if date_range and len(date_range) == 2 else None
                }
            }
            
            import json
            summary_json = json.dumps(summary, indent=2, default=str)
            st.download_button(
                "📄 Download Summary Report",
                data=summary_json,
                file_name=f"ppd_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
        
        # Display current filters
        st.markdown("#### 🔧 Current Filters")
        filter_info = []
        if location:
            filter_info.append(f"📍 Location: {location}")
        if property_type != 'All':
            filter_info.append(f"🏠 Type: {PROPERTY_TYPE_MAP.get(property_type, property_type)}")
        if min_price > 0:
            filter_info.append(f"💰 Min Price: £{min_price:,}")
        if max_price > 0:
            filter_info.append(f"💰 Max Price: £{max_price:,}")
        if date_range and len(date_range) == 2:
            filter_info.append(f"📅 Date: {date_range[0]} to {date_range[1]}")
        
        if filter_info:
            for info in filter_info:
                st.text(info)
        else:
            st.text("No filters applied - showing all data")

if __name__ == "__main__":
    main()
