import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import os
import requests
from io import BytesIO
import base64
import tempfile

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="CO2 Emissions Analytics Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .insight-box {
        background: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .stTab {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to download CSV from Google Drive
@st.cache_data
def download_csv_from_gdrive(file_id):
    """Download CSV file from Google Drive using file ID"""
    try:
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        with st.spinner("Downloading CSV from Google Drive..."):
            response = requests.get(url)
            
            if response.status_code == 200:
                # Read CSV from the downloaded content
                df = pd.read_csv(BytesIO(response.content))
                st.success(f"✅ CSV downloaded successfully! {len(df):,} records loaded.")
                return df
            else:
                st.error(f"❌ Failed to download CSV. Status code: {response.status_code}")
                return None
                
    except Exception as e:
        st.error(f"❌ Error downloading CSV: {str(e)}")
        return None

# Function to generate HTML report (alternative to PDF)
def generate_html_report(df, metrics, plots_data=None):
    """Generate a comprehensive HTML report of the CO2 emissions analysis"""
    try:
        # HTML template with styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CO2 Emissions Analytics Report</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                    color: #333;
                }}
                .header {{
                    text-align: center;
                    color: #1f77b4;
                    border-bottom: 3px solid #1f77b4;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border-left: 5px solid #007bff;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .table th, .table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                .table th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                }}
                .recommendation {{
                    background: #e8f5e8;
                    border-left: 5px solid #28a745;
                    padding: 15px;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌍 CO2 Emissions Analytics Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Executive Summary
        html_content += """
            <div class="section">
                <h2>📊 Executive Summary</h2>
        """
        
        if metrics:
            html_content += f"""
                <p>This comprehensive report analyzes CO2 emissions data covering 
                <strong>{metrics.get('total_countries', 'N/A')} countries</strong> with 
                <strong>{metrics.get('total_records', 0):,} total records</strong> 
                spanning the years <strong>{metrics.get('year_range', 'N/A')}</strong>. 
                The dataset demonstrates <strong>{metrics.get('data_completeness', 0):.1f}% data completeness</strong>.</p>
            </div>
            
            <div class="section">
                <h2>🔢 Key Metrics</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>{metrics.get('total_countries', 'N/A')}</h3>
                        <p>Total Countries</p>
                    </div>
                    <div class="metric-card">
                        <h3>{metrics.get('total_records', 0):,}</h3>
                        <p>Total Records</p>
                    </div>
                    <div class="metric-card">
                        <h3>{metrics.get('year_range', 'N/A')}</h3>
                        <p>Year Range</p>
                    </div>
                    <div class="metric-card">
                        <h3>{metrics.get('data_completeness', 0):.1f}%</h3>
                        <p>Data Completeness</p>
                    </div>
                </div>
            """
            
            if 'avg_co2_emission' in metrics:
                html_content += f"""
                <h3>CO2 Emissions Insights</h3>
                <ul>
                    <li><strong>Average CO2 Emission:</strong> {metrics.get('avg_co2_emission', 0):.2f}</li>
                    <li><strong>Maximum CO2 Emission:</strong> {metrics.get('max_co2_emission', 0):.2f}</li>
                    <li><strong>Countries with Limited Data:</strong> {metrics.get('countries_with_limited_data', 0)}</li>
                </ul>
                """
        
        html_content += "</div>"
        
        # Data Quality Section
        html_content += """
            <div class="section">
                <h2>🔍 Data Quality Analysis</h2>
        """
        
        if df is not None and not df.empty:
            missing_data = df.isnull().sum()
            total_missing = missing_data.sum()
            total_cells = len(df) * len(df.columns)
            completeness = ((total_cells - total_missing) / total_cells * 100)
            
            html_content += f"""
                <h3>Overall Data Quality</h3>
                <ul>
                    <li><strong>Total Missing Values:</strong> {total_missing:,}</li>
                    <li><strong>Overall Completeness:</strong> {completeness:.2f}%</li>
                    <li><strong>Total Records:</strong> {len(df):,}</li>
                    <li><strong>Total Columns:</strong> {len(df.columns)}</li>
                </ul>
            """
            
            # Top columns with missing data
            missing_cols = missing_data[missing_data > 0].sort_values(ascending=False)
            if len(missing_cols) > 0:
                html_content += "<h3>Columns with Missing Data</h3><table class='table'>"
                html_content += "<tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>"
                
                for col, count in missing_cols.head(10).items():
                    percentage = (count / len(df)) * 100
                    html_content += f"<tr><td>{col}</td><td>{count:,}</td><td>{percentage:.1f}%</td></tr>"
                
                html_content += "</table>"
        
        html_content += "</div>"
        
        # Dataset Preview
        if df is not None and not df.empty:
            html_content += """
                <div class="section">
                    <h2>📋 Dataset Preview</h2>
                    <p>First 10 rows of the dataset:</p>
                    <table class='table'>
            """
            
            # Column headers
            html_content += "<tr>"
            for col in df.columns[:8]:  # Show first 8 columns
                html_content += f"<th>{col}</th>"
            if len(df.columns) > 8:
                html_content += "<th>...</th>"
            html_content += "</tr>"
            
            # Data rows
            for idx in range(min(10, len(df))):
                html_content += "<tr>"
                for col in df.columns[:8]:
                    value = str(df.iloc[idx][col])
                    if len(value) > 20:
                        value = value[:17] + "..."
                    html_content += f"<td>{value}</td>"
                if len(df.columns) > 8:
                    html_content += "<td>...</td>"
                html_content += "</tr>"
            
            html_content += "</table></div>"
        
        # Recommendations
        html_content += """
            <div class="section">
                <h2>💡 Recommendations</h2>
                
                <div class="recommendation">
                    <h3>🔧 Data Quality Improvements</h3>
                    <ul>
                        <li>Address missing data in key columns to improve analysis accuracy</li>
                        <li>Implement data validation procedures for future data collection</li>
                        <li>Consider data imputation techniques for critical missing values</li>
                    </ul>
                </div>
                
                <div class="recommendation">
                    <h3>📈 Analysis Focus Areas</h3>
                    <ul>
                        <li>Prioritize countries with comprehensive data for trend analysis</li>
                        <li>Focus on recent years for policy-relevant insights</li>
                        <li>Investigate outliers and anomalies in the emission data</li>
                    </ul>
                </div>
                
                <div class="recommendation">
                    <h3>📊 Monitoring and Reporting</h3>
                    <ul>
                        <li>Establish regular reporting cycles for emission tracking</li>
                        <li>Develop key performance indicators (KPIs) for emission reduction</li>
                        <li>Create automated alerts for significant emission changes</li>
                    </ul>
                </div>
            </div>
        """
        
        # Footer
        html_content += f"""
            <div class="footer">
                <p>🌍 CO2 Emissions Analytics Report | Generated by Streamlit Dashboard</p>
                <p>Report contains {len(df):,} records | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        st.error(f"Error generating HTML report: {str(e)}")
        return None

# Function to convert HTML to downloadable file
def create_downloadable_report(html_content, filename):
    """Create a downloadable HTML report"""
    return html_content.encode('utf-8')
@st.cache_data
def load_data(file_path):
    """Load and cache the CO2 emissions data with comprehensive error handling"""
    try:
        if not os.path.exists(file_path):
            st.error(f"❌ File not found: {file_path}")
            st.info("Please make sure the file exists in the correct location.")
            return None
        
        df = pd.read_csv(file_path)
        
        if df.empty:
            st.error("❌ The CSV file is empty.")
            return None
        
        st.success(f"✅ Data loaded successfully! {len(df):,} records from {df['country'].nunique() if 'country' in df.columns else 'N/A'} countries")
        return df
        
    except pd.errors.EmptyDataError:
        st.error("❌ The CSV file is empty or corrupted.")
        return None
    except pd.errors.ParserError as e:
        st.error(f"❌ Error parsing CSV file: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {str(e)}")
        return None

# Safe metric calculation
def safe_calculate_metrics(df):
    """Calculate metrics with error handling"""
    try:
        metrics = {}
        
        # Basic metrics
        metrics['total_countries'] = df['country'].nunique() if 'country' in df.columns else 0
        metrics['total_records'] = len(df)
        
        # Year range
        if 'year' in df.columns and not df['year'].isna().all():
            metrics['year_range'] = f"{int(df['year'].min())} - {int(df['year'].max())}"
            metrics['min_year'] = int(df['year'].min())
            metrics['max_year'] = int(df['year'].max())
        else:
            metrics['year_range'] = "N/A"
            metrics['min_year'] = 2000
            metrics['max_year'] = 2023
        
        # Data completeness
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        metrics['data_completeness'] = ((total_cells - missing_cells) / total_cells) * 100
        
        # Country analysis
        if 'country' in df.columns:
            country_counts = df['country'].value_counts()
            metrics['countries_with_limited_data'] = len(country_counts[country_counts < 50])
            metrics['top_country'] = country_counts.index[0] if len(country_counts) > 0 else "N/A"
            metrics['top_country_records'] = country_counts.iloc[0] if len(country_counts) > 0 else 0
        
        # CO2 insights
        if 'co2' in df.columns:
            co2_data = df['co2'].dropna()
            if len(co2_data) > 0:
                metrics['total_co2_records'] = len(co2_data)
                metrics['avg_co2_emission'] = co2_data.mean()
                metrics['max_co2_emission'] = co2_data.max()
            else:
                metrics['avg_co2_emission'] = 0
                metrics['max_co2_emission'] = 0
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}

def safe_country_analysis(df):
    """Perform country analysis with error handling"""
    try:
        if 'country' not in df.columns:
            return pd.DataFrame()
        
        # Basic aggregation
        agg_dict = {'year': ['count', 'min', 'max']}
        
        # Add CO2 columns if they exist
        if 'co2' in df.columns:
            agg_dict['co2'] = ['mean', 'sum']
        if 'co2_per_capita' in df.columns:
            agg_dict['co2_per_capita'] = 'mean'
        if 'population' in df.columns:
            agg_dict['population'] = 'mean'
        
        country_stats = df.groupby('country').agg(agg_dict).round(2)
        
        # Flatten column names
        new_columns = []
        for col in country_stats.columns:
            if isinstance(col, tuple):
                new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(str(col))
        
        country_stats.columns = new_columns
        return country_stats.reset_index()
        
    except Exception as e:
        st.error(f"Error in country analysis: {str(e)}")
        return pd.DataFrame()

def create_safe_visualizations(df):
    """Create visualizations with comprehensive error handling"""
    plots = {}
    
    try:
        # 1. Records per country
        if 'country' in df.columns:
            country_counts = df['country'].value_counts().head(15).reset_index()
            country_counts.columns = ['Country', 'Count']
            
            fig = px.bar(
                country_counts, 
                x='Count', 
                y='Country',
                orientation='h',
                title='Top 15 Countries by Record Count',
                color='Count',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500, showlegend=False)
            plots['country_records'] = fig
        
        # 2. Time series if year and co2 exist
        if all(col in df.columns for col in ['year', 'co2']):
            yearly_data = df.groupby('year')['co2'].sum().reset_index()
            if len(yearly_data) > 0:
                fig = px.line(
                    yearly_data, 
                    x='year', 
                    y='co2',
                    title='Total CO2 Emissions Over Time',
                    markers=True
                )
                fig.update_traces(line=dict(width=3))
                fig.update_layout(height=400)
                plots['time_series'] = fig
        
        # 3. CO2 distribution
        if 'co2' in df.columns:
            co2_data = df['co2'].dropna()
            if len(co2_data) > 0:
                fig = px.histogram(
                    df, 
                    x='co2', 
                    title='CO2 Emissions Distribution',
                    nbins=50
                )
                fig.update_layout(height=400)
                plots['co2_distribution'] = fig
        
        # 4. Missing data analysis
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title='Missing Data by Column',
                labels={'x': 'Missing Values', 'y': 'Column'}
            )
            fig.update_layout(height=max(400, len(missing_data) * 25))
            plots['missing_data'] = fig
        
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
    
    return plots

def main():
    """Main application function with comprehensive error handling"""
    
    # Header
    st.markdown('<h1 class="main-header">🌍 CO2 Emissions Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # File path input
    st.sidebar.title("🔧 Configuration")
    
    # Google Drive download section
    st.sidebar.markdown("### 📁 Data Source")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Local File", "Google Drive"]
    )
    
    if data_source == "Google Drive":
        st.sidebar.markdown("#### Download from Google Drive")
        gdrive_file_id = st.sidebar.text_input(
            "Google Drive File ID:",
            help="Enter the file ID from your Google Drive shareable link"
        )
        
        if st.sidebar.button("📥 Download CSV from Google Drive"):
            if gdrive_file_id:
                df = download_csv_from_gdrive(gdrive_file_id)
                if df is not None:
                    st.session_state['df'] = df
                    st.sidebar.success("CSV downloaded and loaded!")
            else:
                st.sidebar.error("Please enter a Google Drive file ID")
        
        # Instructions for Google Drive
        with st.sidebar.expander("ℹ️ How to get Google Drive File ID"):
            st.markdown("""
            1. Upload your CSV to Google Drive
            2. Right-click the file → Share
            3. Change permissions to "Anyone with the link"
            4. Copy the shareable link
            5. Extract the file ID from the URL:
               `https://drive.google.com/file/d/FILE_ID_HERE/view`
            6. Paste the FILE_ID_HERE part above
            """)
    
    else:  # Local File
        file_path = st.sidebar.text_input(
            "CSV File Path:", 
            value=r"final_cleaned_emissions_data.csv",
            help="Enter the path to your CSV file"
        )
    
    # Report Generation Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Report Generation")
    
    # Check if data is available
    data_available = False
    report_df = None
    
    if data_source == "Google Drive" and 'df' in st.session_state:
        data_available = True
        report_df = st.session_state['df']
    elif data_source == "Local File" and file_path:
        try:
            report_df = load_data(file_path)
            if report_df is not None:
                data_available = True
        except:
            pass
    
    if data_available and report_df is not None:
        st.sidebar.success("✅ Data loaded - Ready to generate report!")
        
        if st.sidebar.button("📊 Generate HTML Report", type="primary"):
            with st.spinner("Generating comprehensive HTML report..."):
                # Calculate metrics for the report
                report_metrics = safe_calculate_metrics(report_df)
                
                # Generate HTML report
                html_content = generate_html_report(report_df, report_metrics)
                
                if html_content:
                    # Create download button
                    report_bytes = create_downloadable_report(html_content, "CO2_Report.html")
                    
                    st.sidebar.download_button(
                        label="📥 Download HTML Report",
                        data=report_bytes,
                        file_name=f"CO2_Emissions_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                    
                    st.sidebar.success("✅ HTML report generated successfully!")
                    
                    # Show preview option
                    if st.sidebar.button("👁️ Preview Report"):
                        st.markdown("---")
                        st.markdown("## 📄 Report Preview")
                        st.components.v1.html(html_content, height=600, scrolling=True)
                else:
                    st.sidebar.error("❌ Failed to generate report")
    else:
        st.sidebar.info("📊 Load data first to generate report")
        
    # Additional report options
    with st.sidebar.expander("ℹ️ Report Features"):
        st.markdown("""
        **The HTML report includes:**
        - 📊 Executive summary with key statistics
        - 🔢 Key metrics and insights  
        - 🔍 Data quality analysis
        - 📋 Dataset preview
        - 💡 Professional recommendations
        - 🎨 Professional styling and formatting
        
        **Benefits:**
        - ✅ No external dependencies required
        - ✅ Can be opened in any web browser
        - ✅ Easy to share and print
        - ✅ Professional appearance
        """)
    
    if not file_path and data_source == "Local File":
        st.warning("⚠️ Please specify a file path in the sidebar.")
        return  # Exit early if no file path
    
    # Load data based on source
    if data_source == "Google Drive":
        # Check if data is already loaded in session state
        if 'df' in st.session_state:
            df = st.session_state['df']
        else:
            st.info("📁 Please download your CSV file from Google Drive using the sidebar.")
            return
    else:
        # Load from local file
        with st.spinner("Loading data..."):
            df = load_data(file_path)
    
    # Check if data loading failed
    if df is None:
        st.info("📁 Make sure your CSV file is in the same directory as this script, or provide the full path.")
        return  # Exit early if data loading failed
    
    # Ensure df is not None before proceeding
    if df is None or len(df) == 0:
        st.error("❌ No data available to analyze.")
        return
    
    # Display basic info about the dataset
    st.markdown("## 📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Rows:** {len(df):,}")
    with col2:
        st.info(f"**Columns:** {len(df.columns)}")
    with col3:
        st.info(f"**Countries:** {df['country'].nunique() if 'country' in df.columns else 'N/A'}")
    
    # Calculate metrics
    with st.spinner("Calculating metrics..."):
        metrics = safe_calculate_metrics(df)
    
    # Sidebar filters
    st.sidebar.markdown("### 📊 Data Filters")
    
    # Year filter
    if 'year' in df.columns and not df['year'].isna().all():
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=metrics.get('min_year', 2000),
            max_value=metrics.get('max_year', 2023),
            value=(metrics.get('min_year', 2000), metrics.get('max_year', 2023))
        )
        df_filtered = df[df['year'].between(year_range[0], year_range[1])]
    else:
        df_filtered = df.copy()
        st.sidebar.info("No year data available for filtering")
    
    # Country filter
    if 'country' in df.columns:
        countries = st.sidebar.multiselect(
            "Select Countries (optional)",
            options=sorted(df['country'].unique()),
            default=[]
        )
        
        if countries:
            df_filtered = df_filtered[df_filtered['country'].isin(countries)]
    
    # Main content tabs
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", 
        "🏆 Country Analysis", 
        "📊 Visualizations",
        "🔍 Data Quality"
    ])
    
    with tab1:
        st.markdown("## 📈 Key Metrics")
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🌍 Countries", metrics.get('total_countries', 0))
            
            with col2:
                st.metric("📊 Records", f"{metrics.get('total_records', 0):,}")
            
            with col3:
                st.metric("📅 Years", metrics.get('year_range', 'N/A'))
            
            with col4:
                st.metric("✅ Completeness", f"{metrics.get('data_completeness', 0):.1f}%")
        
        # Dataset preview
        st.markdown("### 📋 Dataset Preview")
        if not df_filtered.empty:
            st.dataframe(df_filtered.head(10), use_container_width=True)
        else:
            st.info("No data available with current filters.")
        
        # Basic statistics
        st.markdown("### 📊 Basic Statistics")
        numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0 and not df_filtered.empty:
            st.dataframe(df_filtered[numeric_columns].describe(), use_container_width=True)
        else:
            st.info("No numeric columns found for statistical analysis.")
    
    with tab2:
        st.markdown("## 🏆 Country Analysis")
        
        if 'country' in df_filtered.columns and not df_filtered.empty:
            # Country record counts
            country_counts = df_filtered['country'].value_counts().reset_index()
            country_counts.columns = ['Country', 'Records']
            
            # Show top countries
            st.markdown("### 📊 Top 20 Countries by Records")
            st.dataframe(country_counts.head(20), use_container_width=True)
            
            # Countries with limited data
            limited_countries = country_counts[country_counts['Records'] < 50]
            
            if len(limited_countries) > 0:
                st.markdown("### ⚠️ Countries with Limited Data (<50 records)")
                st.dataframe(limited_countries, use_container_width=True)
            else:
                st.success("✅ All countries have sufficient data (≥50 records)")
            
            # Detailed country analysis
            country_analysis = safe_country_analysis(df_filtered)
            if not country_analysis.empty:
                st.markdown("### 🔍 Detailed Country Statistics")
                st.dataframe(country_analysis, use_container_width=True)
        else:
            st.warning("⚠️ No country data available with current filters.")
    
    with tab3:
        st.markdown("## 📊 Data Visualizations")
        
        if not df_filtered.empty:
            plots = create_safe_visualizations(df_filtered)
            
            if 'country_records' in plots:
                st.plotly_chart(plots['country_records'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'time_series' in plots:
                    st.plotly_chart(plots['time_series'], use_container_width=True)
            
            with col2:
                if 'co2_distribution' in plots:
                    st.plotly_chart(plots['co2_distribution'], use_container_width=True)
        else:
            st.info("No data available for visualization with current filters.")
    
    with tab4:
        st.markdown("## 🔍 Data Quality Analysis")
        
        if not df_filtered.empty:
            # Missing data visualization
            plots = create_safe_visualizations(df_filtered)
            if 'missing_data' in plots:
                st.plotly_chart(plots['missing_data'], use_container_width=True)
            
            # Data quality summary
            st.markdown("### 📋 Data Quality Summary")
            
            total_cells = len(df_filtered) * len(df_filtered.columns)
            missing_cells = df_filtered.isnull().sum().sum()
            
            quality_metrics = {
                'Total Records': len(df_filtered),
                'Total Columns': len(df_filtered.columns),
                'Total Cells': total_cells,
                'Missing Cells': missing_cells,
                'Data Completeness': f"{((total_cells - missing_cells) / total_cells * 100):.2f}%"
            }
            
            quality_df = pd.DataFrame(list(quality_metrics.items()), 
                                    columns=['Metric', 'Value'])
            st.dataframe(quality_df, use_container_width=True, hide_index=True)
            
            # Column-wise missing data
            missing_by_column = df_filtered.isnull().sum().sort_values(ascending=False)
            missing_by_column = missing_by_column[missing_by_column > 0]
            
            if len(missing_by_column) > 0:
                st.markdown("### 📊 Missing Data by Column")
                missing_df = pd.DataFrame({
                    'Column': missing_by_column.index,
                    'Missing_Count': missing_by_column.values,
                    'Missing_Percentage': (missing_by_column.values / len(df_filtered) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No missing data found!")
        else:
            st.info("No data available for quality analysis with current filters.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌍 CO2 Emissions Analytics Dashboard | Built with Streamlit & Plotly</p>
        <p>Analyzing {len(df_filtered):,} records | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
