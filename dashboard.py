import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Cargo Robotics Performance Dashboard",
    page_icon="🚚",
    layout="wide"
)

# --- Simple Logo Display ---
logo_path = "cargo_logo.png"  # Just use one logo
if os.path.exists(logo_path):
    st.image(logo_path, width=400)

# --- Custom CSS for a Polished Look & Responsiveness ---
st.markdown("""
<style>
    /* Increase base font size for the entire app */
    html, body, [class*="st-"] {
        font-size: 18px;
    }
    
    /* Responsive columns for KPI and Error tables */
    @media (max-width: 1200px) {
        div[data-testid="column"] {
            flex-wrap: wrap;
            flex: 1 1 100%; /* Allow columns to take full width and wrap */
            min-width: 300px; /* Prevent columns from becoming too narrow */
        }
    }

    /* Increase font size for tables generated from pandas .to_html() */
    table {
        font-size: 1.1rem !important; /* Make table text larger */
        width: 100%;
    }
    th {
        font-size: 1.2rem !important; /* Make table headers even larger */
        text-align: left;
        font-weight: bold;
    }
    /* Increase font size for native Streamlit DataFrames and their content */
    .stDataFrame, .stDataFrame [data-testid="stTable"] {
        font-size: 1.1rem !important;
    }
    /* Increase font size for widget labels (selectbox, etc.) */
    .st-emotion-cache-1y4p8pa {
        font-size: 1.2rem !important;
    }
    /* Increase font size for expander header */
    .st-emotion-cache-pwan1i {
        font-size: 1.2rem !important;
    }
    /* Increase font size for expander content */
    .st-emotion-cache-1hver84 {
        font-size: 1.1rem !important;
    }
    /* Style for the main section dividers */
    hr {
        height: 2px !important;
        border: none !important;
        background: linear-gradient(to right, #ff4b4b, #ffa421, #f9f871) !important; /* Red -> Orange -> Yellow */
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data(file_path):
    """Loads all sheets from an Excel file into a dictionary of DataFrames."""
    try:
        xls = pd.ExcelFile(file_path)
        data = {}
        for sheet_name in xls.sheet_names:
            # Read summary sheets normally, but read daily sheets with the first row as the header
            if sheet_name in ['Master Summary', 'Error Summary']:
                data[sheet_name] = pd.read_excel(xls, sheet_name)
            else:
                data[sheet_name] = pd.read_excel(xls, sheet_name, header=0)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please run the report updater first.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the Excel file: {e}", icon="🔥")
        return None

# --- Reusable Dashboard Function ---
def create_dashboard_view(summary_df, error_summary_df, all_data, key_prefix):
    """Creates a full dashboard view for a given dataset."""
    if summary_df is None or summary_df.empty:
        st.info("No data available for this view.", icon="📊")
        return

    # --- Helper function for calculating change ---
    def get_change(current, previous, is_time=False):
        """Calculates the percentage change and formats it for st.metric."""
        if pd.isna(current) or pd.isna(previous) or previous == 0:
            return None  # Return None for st.metric to display no delta
        
        change = ((current - previous) / previous) * 100
        return f"{change:.2f}%"

    # --- Prepare DataFrames ---
    summary_df = summary_df.copy()
    error_summary_df = error_summary_df.copy() if error_summary_df is not None else pd.DataFrame()

    # Rename 'Day' to 'Log' to match user request
    if 'Day' in summary_df.columns:
        summary_df.rename(columns={'Day': 'Log'}, inplace=True)

    # Convert relevant columns to numeric, REMOVE throughput
    numeric_cols = [
        'Pickup Avg (s)', 'Placement Avg (s)', 'Stow Avg (s)', 
        'Retrieve Avg (s)', 'Read Label Avg (s)', 
        'Packages Picked Up', 'Pickup Attempts', 'Packages Placed', 'Placement Attempts',
        'Packages Retrieved', 'Retrieval Attempts', 'Total Errors', 
    ]
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')

    # --- Master Summary Section (No longer in a tab) ---
    st.header("Overall Performance", divider='orange')

    # --- KPI Tracker & Error Analysis ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("KPI Tracker")
        if not summary_df.empty and len(summary_df) >= 2:
            last_day_stats = summary_df.iloc[-1]
            prev_day_stats = summary_df.iloc[-2]

            # --- Helpers for formatting changes ---
            def get_change(current, previous, is_time=False):
                if pd.isna(current) or pd.isna(previous) or previous == 0: return "N/A"
                change = ((current - previous) / previous) * 100
                color = "green" if (change < 0 if is_time else change > 0) else "red"
                return f'<span style="color:{color};">{change:+.2f}%</span>'

            def get_rate_change(current, previous):
                if pd.isna(current) or pd.isna(previous): return "N/A"
                change = current - previous
                color = "green" if change >= 0 else "red"
                return f'<span style="color:{color};">{change:+.2f} pts</span>'

            # --- Calculate Overall Averages & Rates ---
            if 'Packages Picked Up' in summary_df.columns and 'Pickup Attempts' in summary_df.columns:
                pickup_rate_overall = (summary_df['Packages Picked Up'].sum() / summary_df['Pickup Attempts'].sum() * 100) if summary_df['Pickup Attempts'].sum() > 0 else 0
                last_pickup_rate = (last_day_stats['Packages Picked Up'] / last_day_stats['Pickup Attempts'] * 100) if last_day_stats['Pickup Attempts'] > 0 else 0
                prev_pickup_rate = (prev_day_stats['Packages Picked Up'] / prev_day_stats['Pickup Attempts'] * 100) if prev_day_stats['Pickup Attempts'] > 0 else 0
            
            if 'Packages Placed' in summary_df.columns and 'Placement Attempts' in summary_df.columns:
                placement_rate_overall = (summary_df['Packages Placed'].sum() / summary_df['Placement Attempts'].sum() * 100) if summary_df['Placement Attempts'].sum() > 0 else 0
                last_placement_rate = (last_day_stats['Packages Placed'] / last_day_stats['Placement Attempts'] * 100) if last_day_stats['Placement Attempts'] > 0 else 0
                prev_placement_rate = (prev_day_stats['Packages Placed'] / prev_day_stats['Placement Attempts'] * 100) if prev_day_stats['Placement Attempts'] > 0 else 0

            retrieve_rate_overall = (summary_df['Packages Retrieved'].sum() / summary_df['Retrieval Attempts'].sum() * 100) if summary_df['Retrieval Attempts'].sum() > 0 else 0
            last_retrieval_rate = (last_day_stats['Packages Retrieved'] / last_day_stats['Retrieval Attempts'] * 100) if last_day_stats['Retrieval Attempts'] > 0 else 0
            prev_retrieval_rate = (prev_day_stats['Packages Retrieved'] / prev_day_stats['Retrieval Attempts'] * 100) if prev_day_stats['Retrieval Attempts'] > 0 else 0

            kpi_data = {
                'Metric': [
                    'Pickup Avg (s)', 'Placement Avg (s)', 'Stow Avg (s)',
                    'Retrieve Avg (s)', 'Read Label Avg (s)', 
                    'Pickup Success Rate', 'Placement Success Rate', 'Retrieval Success Rate'
                ],
                'Overall Average': [
                    f"{summary_df['Pickup Avg (s)'].mean():.2f}" if 'Pickup Avg (s)' in summary_df.columns else "N/A",
                    f"{summary_df['Placement Avg (s)'].mean():.2f}" if 'Placement Avg (s)' in summary_df.columns else "N/A",
                    f"{summary_df['Stow Avg (s)'].mean():.2f}",
                    f"{summary_df['Retrieve Avg (s)'].mean():.2f}",
                    f"{summary_df['Read Label Avg (s)'].mean():.2f}",
                    f"{pickup_rate_overall:.2f}%" if 'Packages Picked Up' in summary_df.columns else "N/A",
                    f"{placement_rate_overall:.2f}%" if 'Packages Placed' in summary_df.columns else "N/A",
                    f"{retrieve_rate_overall:.2f}%"
                ],
                'Most Recent Day': [
                    f"{last_day_stats['Pickup Avg (s)']:.2f}" if 'Pickup Avg (s)' in last_day_stats else "N/A",
                    f"{last_day_stats['Placement Avg (s)']:.2f}" if 'Placement Avg (s)' in last_day_stats else "N/A",
                    f"{last_day_stats['Stow Avg (s)']:.2f}",
                    f"{last_day_stats['Retrieve Avg (s)']:.2f}",
                    f"{last_day_stats['Read Label Avg (s)']:.2f}",
                    f"{last_pickup_rate:.2f}%" if 'Packages Picked Up' in last_day_stats else "N/A",
                    f"{last_placement_rate:.2f}%" if 'Packages Placed' in last_day_stats else "N/A",
                    f"{last_retrieval_rate:.2f}%"
                ],
                'Change from Prev. Day': [
                    get_change(last_day_stats['Pickup Avg (s)'], prev_day_stats['Pickup Avg (s)'], is_time=True) if 'Pickup Avg (s)' in last_day_stats else "N/A",
                    get_change(last_day_stats['Placement Avg (s)'], prev_day_stats['Placement Avg (s)'], is_time=True) if 'Placement Avg (s)' in last_day_stats else "N/A",
                    get_change(last_day_stats['Stow Avg (s)'], prev_day_stats['Stow Avg (s)'], is_time=True),
                    get_change(last_day_stats['Retrieve Avg (s)'], prev_day_stats['Retrieve Avg (s)'], is_time=True),
                    get_change(last_day_stats['Read Label Avg (s)'], prev_day_stats['Read Label Avg (s)'], is_time=True),
                    get_rate_change(last_pickup_rate, prev_pickup_rate) if 'Packages Picked Up' in last_day_stats else "N/A",
                    get_rate_change(last_placement_rate, prev_placement_rate) if 'Packages Placed' in last_day_stats else "N/A",
                    get_rate_change(last_retrieval_rate, prev_retrieval_rate)
                ]
            }
            kpi_df = pd.DataFrame(kpi_data)
            st.markdown(kpi_df.to_html(escape=False, index=False, na_rep="N/A"), unsafe_allow_html=True)

        elif not summary_df.empty: # Handle case with only one day of data
            last_day_stats = summary_df.iloc[-1]
            # FIXED: Use new column names
            pickup_rate = (last_day_stats['Packages Picked Up'] / last_day_stats['Pickup Attempts'] * 100) if last_day_stats['Pickup Attempts'] > 0 else 0
            placement_rate = (last_day_stats['Packages Placed'] / last_day_stats['Placement Attempts'] * 100) if last_day_stats['Placement Attempts'] > 0 else 0
            retrieval_rate = (last_day_stats['Packages Retrieved'] / last_day_stats['Retrieval Attempts'] * 100) if last_day_stats['Retrieval Attempts'] > 0 else 0
            
            # For a single day, Overall Average is the same as Most Recent Day
            most_recent_values = [
                f"{last_day_stats['Pickup Avg (s)']:.2f}" if 'Pickup Avg (s)' in last_day_stats else "N/A",
                f"{last_day_stats['Placement Avg (s)']:.2f}" if 'Placement Avg (s)' in last_day_stats else "N/A",
                f"{last_day_stats['Stow Avg (s)']:.2f}" if pd.notna(last_day_stats['Stow Avg (s)']) else "N/A",
                f"{last_day_stats['Retrieve Avg (s)']:.2f}" if pd.notna(last_day_stats['Retrieve Avg (s)']) else "N/A",
                f"{last_day_stats['Read Label Avg (s)']:.2f}" if pd.notna(last_day_stats['Read Label Avg (s)']) else "N/A",
                f"{pickup_rate:.2f}%",
                f"{placement_rate:.2f}%",
                f"{retrieval_rate:.2f}%"
            ]
            
            kpi_data = {
                'Metric': [
                    'Pickup Avg (s)', 'Placement Avg (s)', 'Stow Avg (s)',
                    'Retrieve Avg (s)', 'Read Label Avg (s)', 
                    'Pickup Success Rate', 'Placement Success Rate', 'Retrieval Success Rate'
                ],
                'Overall Average': most_recent_values,
                'Most Recent Day': most_recent_values
            }
            kpi_df = pd.DataFrame(kpi_data)
            st.markdown(kpi_df.to_html(escape=False, index=False, na_rep="N/A"), unsafe_allow_html=True)
        else:
            st.info("No KPI data available.")

    with col2:
        st.subheader("Error Analysis")
        if not error_summary_df.empty and len(summary_df) >= 2:
            error_categories = [col for col in error_summary_df.columns if col not in ['Day', 'Date']]
            # Ensure error counts are numeric, filling any non-numeric with 0
            for col in error_categories:
                error_summary_df[col] = pd.to_numeric(error_summary_df[col], errors='coerce').fillna(0)

            total_errors_by_category = error_summary_df[error_categories].sum()
            total_errors_overall = total_errors_by_category.sum()

            last_day_errors = error_summary_df.iloc[-1]
            prev_day_errors = error_summary_df.iloc[-2]
            
            last_day_total = last_day_errors[error_categories].sum()
            prev_day_total = prev_day_errors[error_categories].sum()

            error_analysis_data = []
            for cat in error_categories:
                if total_errors_by_category[cat] == 0: continue
                
                overall_pct = (total_errors_by_category[cat] / total_errors_overall * 100) if total_errors_overall > 0 else 0
                last_day_pct = (last_day_errors.get(cat, 0) / last_day_total * 100) if last_day_total > 0 else 0
                prev_day_pct = (prev_day_errors.get(cat, 0) / prev_day_total * 100) if prev_day_total > 0 else 0
                
                change_in_pct = last_day_pct - prev_day_pct
                color = "red" if change_in_pct > 0 else "green"
                change_str = f'<span style="color:{color};">{change_in_pct:+.2f} pts</span>'

                error_analysis_data.append({
                    'Error Category': cat, 'Overall % of Total Errors': overall_pct,
                    'Most Recent Day %': last_day_pct, 'Change from Prev. Day': change_str
                })
            
            sorted_error_data = sorted(error_analysis_data, key=lambda x: x['Overall % of Total Errors'], reverse=True)
            final_error_data = sorted_error_data[:7]
            if len(sorted_error_data) > 7:
                others_data = sorted_error_data[7:]
                final_error_data.append({
                    'Error Category': 'Others',
                    'Overall % of Total Errors': sum(d['Overall % of Total Errors'] for d in others_data),
                    'Most Recent Day %': sum(d['Most Recent Day %'] for d in others_data),
                    'Change from Prev. Day': 'N/A'
                })
            
            for item in final_error_data:
                item['Overall % of Total Errors'] = f"{item['Overall % of Total Errors']:.2f}%"
                item['Most Recent Day %'] = f"{item['Most Recent Day %']:.2f}%"
            
            error_df = pd.DataFrame(final_error_data)
            st.markdown(error_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        elif not error_summary_df.empty: # Handle case with only one day of error data
            error_categories = [col for col in error_summary_df.columns if col not in ['Day', 'Date']]
            for col in error_categories:
                error_summary_df[col] = pd.to_numeric(error_summary_df[col], errors='coerce').fillna(0)

            last_day_errors = error_summary_df.iloc[-1]
            total_errors_last_day = last_day_errors[error_categories].sum()

            error_analysis_data = []
            if total_errors_last_day > 0:
                for cat in error_categories:
                    count = last_day_errors.get(cat, 0)
                    if count > 0:
                        error_analysis_data.append({
                            'Error Category': cat,
                            '% of Total Errors': (count / total_errors_last_day * 100)
                        })
                
                sorted_error_data = sorted(error_analysis_data, key=lambda x: x['% of Total Errors'], reverse=True)
                
                for item in sorted_error_data:
                    item['% of Total Errors'] = f"{item['% of Total Errors']:.2f}%"
                
                error_df = pd.DataFrame(sorted_error_data)
                st.markdown(error_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("No errors recorded for this day.")
        else:
            st.info("Not enough data for error analysis.")

    # --- Trend Charts ---
    st.header("Performance Trends", divider='orange')
    if len(summary_df) > 1:
        # Ensure Log column is integer for axis ticks
        if 'Log' in summary_df.columns:
            summary_df['Log'] = summary_df['Log'].astype(int)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4>Package Pickup Operations</h4>", unsafe_allow_html=True)
            if 'Packages Picked Up' in summary_df.columns and 'Pickup Attempts' in summary_df.columns:
                fig_pickup = go.Figure()
                fig_pickup.add_trace(go.Bar(
                    name='Packages Picked Up', 
                    x=summary_df['Log'], 
                    y=summary_df['Packages Picked Up'], 
                    marker_color='#34a853'
                ))
                fig_pickup.add_trace(go.Bar(
                    name='Pickup Attempts', 
                    x=summary_df['Log'], 
                    y=summary_df['Pickup Attempts'], 
                    marker_color='#fbbc05'
                ))
                fig_pickup.update_layout(
                    barmode='group', 
                    yaxis_title='Count', 
                    title='Package Pickup: Success vs Attempts'
                )
                fig_pickup.update_xaxes(dtick=1, title_text='Log')
                st.plotly_chart(fig_pickup, use_container_width=True)
            else:
                st.info("Pickup data not available")

            st.markdown("<h4>Package Placement Operations</h4>", unsafe_allow_html=True)
            if 'Packages Placed' in summary_df.columns and 'Placement Attempts' in summary_df.columns:
                fig_placement = go.Figure()
                fig_placement.add_trace(go.Bar(
                    name='Packages Placed', 
                    x=summary_df['Log'], 
                    y=summary_df['Packages Placed'], 
                    marker_color='#ff6d01'
                ))
                fig_placement.add_trace(go.Bar(
                    name='Placement Attempts', 
                    x=summary_df['Log'], 
                    y=summary_df['Placement Attempts'], 
                    marker_color='#ff9500'
                ))
                fig_placement.update_layout(
                    barmode='group', 
                    yaxis_title='Count', 
                    title='Package Placement: Success vs Attempts'
                )
                fig_placement.update_xaxes(dtick=1, title_text='Log')
                st.plotly_chart(fig_placement, use_container_width=True)
            else:
                st.info("Placement data not available")

        with col2:
            st.markdown("<h4>Average Process Times</h4>", unsafe_allow_html=True)
            time_columns = []
            if 'Pickup Avg (s)' in summary_df.columns:
                time_columns.append('Pickup Avg (s)')
            if 'Placement Avg (s)' in summary_df.columns:
                time_columns.append('Placement Avg (s)')
            time_columns.extend(['Stow Avg (s)', 'Retrieve Avg (s)', 'Read Label Avg (s)'])
            
            fig = px.line(summary_df, x='Log', y=time_columns, markers=True)
            fig.update_xaxes(dtick=1) # Set x-axis ticks to integers
            fig.update_yaxes(title_text='Time (s)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<h4>Packages Retrieved vs. Attempts</h4>", unsafe_allow_html=True)
            fig4 = go.Figure(data=[
                go.Bar(name='Packages Retrieved', x=summary_df['Log'], y=summary_df['Packages Retrieved']),
                go.Bar(name='Retrieval Attempts', x=summary_df['Log'], y=summary_df['Retrieval Attempts'])
            ])
            fig4.update_layout(barmode='group', yaxis_title='Count')
            fig4.update_xaxes(dtick=1, title_text='Log')
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Need at least two days of data to show trend charts.")

    # --- Day-by-Day Table ---
    st.header("Daily Performance", divider='orange')
    st.subheader("Day-by-Day Performance")

    # Create a display-ready version of the dataframe to ensure 'N/A' is shown correctly.
    display_df = summary_df.copy()

    # Define which columns need specific decimal formatting.
    cols_to_format = [
        'Pickup Avg (s)', 'Placement Avg (s)', 'Stow Avg (s)', 'Retrieve Avg (s)', 'Read Label Avg (s)', 
    ]

    # Apply formatting: 2 decimal places for numbers, and 'N/A' for any missing values.
    for col in cols_to_format:
        if col in display_df.columns:
            # The summary_df was already converted to numeric, so we can safely format.
            display_df[col] = display_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')

    # Display the fully formatted dataframe.
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Daily Detail Section ---
    st.subheader("Daily Detailed Report")
    day_sheets = sorted([s for s in all_data.keys() if s.startswith('Day ')], key=lambda x: int(x.split(' ')[1]))

    if day_sheets:
        selected_day = st.selectbox(
            "Select a day to inspect:",
            options=day_sheets,
            key=f"{key_prefix}_day_select" # Use the key_prefix
        )

        if selected_day:
            day_df = all_data[selected_day]

            # --- Find the end of the main event logs (before summary stats) ---
            summary_start_indices = day_df.index[day_df.iloc[:, 0].astype(str).str.contains('Average', na=False)]
            event_log_end_row = summary_start_indices.min() - 2 if not summary_start_indices.empty else len(day_df)

            # --- Extract Event Logs for all 4 types ---
            pickup_events = day_df.iloc[:event_log_end_row, 0:3].dropna(how='all')      # Columns A-C: Pickup
            placement_events = day_df.iloc[:event_log_end_row, 3:6].dropna(how='all')   # Columns D-F: Placement  
            retrieve_events = day_df.iloc[:event_log_end_row, 6:9].dropna(how='all')    # Columns G-I: Retrieve
            read_label_events = day_df.iloc[:event_log_end_row, 9:12].dropna(how='all') # Columns J-L: Read Label

            # --- Standardize Column Headers for all 4 event types ---
            clean_headers = ['Timestamp', 'Event Label', 'Time (s)']
            
            # Handle Pickup Events
            if pickup_events.empty:
                pickup_events = pd.DataFrame([['N/A', 'N/A', 'N/A']], columns=clean_headers)
            else:
                pickup_events.columns = clean_headers

            # Handle Placement Events
            if placement_events.empty:
                placement_events = pd.DataFrame([['N/A', 'N/A', 'N/A']], columns=clean_headers)
            else:
                placement_events.columns = clean_headers

            # Handle Retrieve Events
            if retrieve_events.empty:
                retrieve_events = pd.DataFrame([['N/A', 'N/A', 'N/A']], columns=clean_headers)
            else:
                retrieve_events.columns = clean_headers

            # Handle Read Label Events
            if read_label_events.empty:
                read_label_events = pd.DataFrame([['N/A', 'N/A', 'N/A']], columns=clean_headers)
            else:
                if read_label_events.shape[1] == 3:
                    read_label_events.columns = clean_headers
                else:
                    read_label_events = pd.DataFrame([['N/A', 'N/A', 'N/A']], columns=clean_headers)

            # Find the furthest down row for error positioning
            last_pickup_idx = pickup_events.index[-1] if not pickup_events.empty else -1
            last_placement_idx = placement_events.index[-1] if not placement_events.empty else -1
            last_retrieve_idx = retrieve_events.index[-1] if not retrieve_events.empty else -1
            last_read_label_idx = read_label_events.index[-1] if not read_label_events.empty else -1
            
            furthest_down_row = max(last_pickup_idx, last_placement_idx, last_retrieve_idx, last_read_label_idx)

            # --- Display Summaries and Event Logs in 4 Columns ---
            st.subheader("Event Logs & Summaries")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("<h4>Pickup Summary</h4>", unsafe_allow_html=True)
                if not summary_start_indices.empty:
                    summary_start_row = summary_start_indices.min()
                    pickup_summary_df = day_df.iloc[summary_start_row:summary_start_row+3, 0:2].dropna(how='all', axis=1).reset_index(drop=True)
                    if pickup_summary_df.shape[1] == 2:
                        pickup_summary_df.columns = ["Metric", "Time (s)"]
                    else:
                        pickup_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                else:
                    pickup_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                st.dataframe(pickup_summary_df, use_container_width=True, hide_index=True)
                
                st.markdown("<h4>Pickup Events</h4>", unsafe_allow_html=True)
                pickup_df_display = pickup_events.reset_index(drop=True)
                pickup_df_display.index = pickup_df_display.index + 1
                pickup_df_display.index.name = '#'
                st.dataframe(pickup_df_display, use_container_width=True)

            with col2:
                st.markdown("<h4>Placement Summary</h4>", unsafe_allow_html=True)
                if not summary_start_indices.empty:
                    summary_start_row = summary_start_indices.min()
                    placement_summary_df = day_df.iloc[summary_start_row:summary_start_row+3, 3:5].dropna(how='all', axis=1).reset_index(drop=True)
                    if placement_summary_df.shape[1] == 2:
                        placement_summary_df.columns = ["Metric", "Time (s)"]
                    else:
                        placement_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                else:
                    placement_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                st.dataframe(placement_summary_df, use_container_width=True, hide_index=True)

                st.markdown("<h4>Placement Events</h4>", unsafe_allow_html=True)
                placement_df_display = placement_events.reset_index(drop=True)
                placement_df_display.index = placement_df_display.index + 1
                placement_df_display.index.name = '#'
                st.dataframe(placement_df_display, use_container_width=True)

            with col3:
                st.markdown("<h4>Retrieve Summary</h4>", unsafe_allow_html=True)
                if not summary_start_indices.empty:
                    summary_start_row = summary_start_indices.min()
                    retrieve_summary_df = day_df.iloc[summary_start_row:summary_start_row+3, 6:8].dropna(how='all', axis=1).reset_index(drop=True)
                    if retrieve_summary_df.shape[1] == 2:
                        retrieve_summary_df.columns = ["Metric", "Time (s)"]
                    else:
                        retrieve_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                else:
                    retrieve_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                st.dataframe(retrieve_summary_df, use_container_width=True, hide_index=True)

                st.markdown("<h4>Retrieve Events</h4>", unsafe_allow_html=True)
                retrieve_df_display = retrieve_events.reset_index(drop=True)
                retrieve_df_display.index = retrieve_df_display.index + 1
                retrieve_df_display.index.name = '#'
                st.dataframe(retrieve_df_display, use_container_width=True)

            with col4:
                st.markdown("<h4>Read Label Summary</h4>", unsafe_allow_html=True)
                if not summary_start_indices.empty:
                    summary_start_row = summary_start_indices.min()
                    read_label_summary_df = day_df.iloc[summary_start_row:summary_start_row+3, 9:11].dropna(how='all', axis=1).reset_index(drop=True)
                    if read_label_summary_df.shape[1] == 2:
                        read_label_summary_df.columns = ["Metric", "Time (s)"]
                    else:
                        read_label_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                else:
                    read_label_summary_df = pd.DataFrame({'Metric': ['Average', 'Min', 'Max'], 'Time (s)': ['N/A', 'N/A', 'N/A']})
                st.dataframe(read_label_summary_df, use_container_width=True, hide_index=True)

                st.markdown("<h4>Read Label Events</h4>", unsafe_allow_html=True)
                read_label_df_display = read_label_events.reset_index(drop=True)
                read_label_df_display.index = read_label_df_display.index + 1
                read_label_df_display.index.name = '#'
                st.dataframe(read_label_df_display, use_container_width=True)

            # --- Display Failures Section (Full Width) ---
            st.subheader("Failures")
            st.markdown("<h4>Failures Logged</h4>", unsafe_allow_html=True)
            if furthest_down_row != -1:
                error_start_row = furthest_down_row + 7 # Start 7 rows down
                
                # Extract the block of potential error data
                error_block = day_df.iloc[error_start_row:]
                
                # Find the first completely blank row to mark the end of the table
                end_row_indices = error_block.index[error_block.isnull().all(axis=1)]
                error_end_row = end_row_indices.min() if not end_row_indices.empty else len(day_df)
                
                # Dynamically find the start of the error table instead of assuming it's at column 0
                potential_error_df = day_df.iloc[error_start_row:error_end_row]
                
                # Find the first column that is not entirely empty
                first_valid_col = 0
                for col in range(potential_error_df.shape[1]):
                    if not potential_error_df.iloc[:, col].isnull().all():
                        first_valid_col = col
                        break
                
                # Select the 3 columns of the error table from the dynamic start
                error_df = potential_error_df.iloc[:, first_valid_col:first_valid_col+3].dropna(how='all')

                if not error_df.empty and error_df.shape[1] == 3:
                    # Check if the first row looks like headers before assigning
                    if not pd.api.types.is_numeric_dtype(error_df.iloc[0, 0]) and isinstance(error_df.iloc[0, 0], str):
                        error_df.columns = error_df.iloc[0].astype(str)
                        error_df = error_df[1:].reset_index(drop=True)
                    
                    error_df.index = error_df.index + 1
                    error_df.index.name = '#'
                    st.dataframe(error_df, use_container_width=True)
                else:
                    st.write("No failure data found in expected location.")
            else:
                st.write("Could not determine where to look for failure data.")

            # --- Raw Log File Display Section ---
            st.subheader("Complete Log File")

            if selected_day:
                # Extract the day number from the selected day (e.g., "Day 1" -> "1")
                day_number = selected_day.split(' ')[1]
                
                # Look for corresponding log files in the appropriate archive folder
                import glob
                import json
                
                # Determine which archive folder to use based on the key_prefix
                if key_prefix == "benchmark":
                    archive_folder = "archive_benchmark"
                else:
                    archive_folder = "archive"
                
                # Search for log files that might correspond to this day
                log_files = glob.glob(f"{archive_folder}/logs_*.json")
                
                if log_files:
                    # Sort log files by date to match with day numbers
                    log_files.sort()
                    
                    # Try to match the day number with the log file index
                    try:
                        day_idx = int(day_number) - 1  # Convert to 0-based index
                        if 0 <= day_idx < len(log_files):
                            selected_log_file = log_files[day_idx]
                            
                            st.markdown(f"**📄 Log File: `{selected_log_file}`**")
                            
                            try:
                                with open(selected_log_file, 'r') as f:
                                    log_content = f.read()
                                
                                # Display file info
                                import os
                                file_size = os.path.getsize(selected_log_file)
                                log_lines = log_content.strip().split('\n')
                                st.info(f"📁 **File:** {selected_log_file} | **Size:** {file_size:,} bytes | **Lines:** {len(log_lines):,}")
                                
                                # Download button
                                st.download_button(
                                    label=f"⬇️ Download {os.path.basename(selected_log_file)}",
                                    data=log_content,
                                    file_name=os.path.basename(selected_log_file),
                                    mime="application/json",
                                    key=f"{key_prefix}_download_{day_number}"
                                )
                                
                                # Checkbox to show/hide the log content
                                show_raw_log = st.checkbox(
                                    f"Show complete log file content for {selected_day}",
                                    key=f"{key_prefix}_show_log_{day_number}"
                                )
                                
                                if show_raw_log:
                                    st.markdown("**📋 Complete Log File (Raw Text):**")
                                    st.text_area(
                                        "Log Content",
                                        value=log_content,
                                        height=400,
                                        key=f"{key_prefix}_log_content_{day_number}"
                                    )
                                
                            except FileNotFoundError:
                                st.error(f"❌ Log file {selected_log_file} not found.")
                            except Exception as e:
                                st.error(f"❌ Error reading log file: {e}")
                        else:
                            st.warning(f"⚠️ No log file found for {selected_day}. Available log files: {len(log_files)}")
                            
                            # Show available log files for debugging
                            with st.expander("🔍 Available log files"):
                                for i, log_file in enumerate(log_files):
                                    st.write(f"Day {i+1}: `{log_file}`")
                    
                    except ValueError:
                        st.error("❌ Could not parse day number from selection.")
                else:
                    st.warning(f"⚠️ No log files found in the {archive_folder} folder.")
                    
                    # Also check if there are any log files in the logs folder
                    current_log_files = glob.glob("logs/*.json")
                    if current_log_files:
                        st.info(f"ℹ️ Found {len(current_log_files)} log files in the logs folder that haven't been processed yet.")

            else:
                st.info("👆 Select a day above to view the corresponding log file.")

    else:
        st.info("No daily detail sheets found in the report to display.")

# --- Main Application ---
# Load data from BOTH report files
road_test_data = load_data('master_report.xlsx')
benchmark_data = load_data('benchmark_report.xlsx')

# Display the logo and the title underneath
st.title("Performance Dashboard")

# --- Main Tabs ---
tab1, tab2 = st.tabs(["Road Tests", "In-House Benchmarks"])

with tab1:
    if road_test_data:
        summary_df = road_test_data.get('Master Summary')
        error_summary_df = road_test_data.get('Error Summary')
        create_dashboard_view(summary_df, error_summary_df, road_test_data, "road_test")
    else:
        st.info("No Road Test data found. Please process logs to generate the `master_report.xlsx` file.", icon="🛣️")

with tab2:
    if benchmark_data:
        benchmark_summary_df = benchmark_data.get('Master Summary')
        # Benchmarks will have their own error summary within their own report
        error_summary_df = benchmark_data.get('Error Summary') 
        create_dashboard_view(benchmark_summary_df, error_summary_df, benchmark_data, "benchmark")
    else:
        st.info("No Benchmark data found. Please process logs to generate the `benchmark_report.xlsx` file.", icon="🔬")
