import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_monthly_revenue(row, cloud_type, ondemand_ratio, owner_split, platform_fee):
    """Calculate monthly revenue based on pricing and usage parameters"""
    if cloud_type == 'Community':
        spot_price = row['Community Spot']
        ondemand_price = row['Community OnDemand']
    else:
        spot_price = row['Secure Spot']
        ondemand_price = row['Secure OnDemand']

    # Calculate weighted average hourly price
    # Note: spot_ratio is now (1 - ondemand_ratio)
    avg_hourly_price = (spot_price * (1 - ondemand_ratio) + ondemand_price * ondemand_ratio)

    # Calculate monthly revenue (24 hours * 30 days = 720 hours per month)
    monthly_revenue = avg_hourly_price * 720

    # Apply platform fee and owner split
    monthly_revenue = monthly_revenue * (1 - platform_fee/100) * (owner_split/100)

    return monthly_revenue

def calculate_financial_metrics(purchase_price, monthly_revenue):
    """Calculate ROI metrics"""
    annual_revenue = monthly_revenue * 12
    roi = (annual_revenue / purchase_price) * 100
    payback_period = purchase_price / monthly_revenue if monthly_revenue > 0 else float('inf')

    return {
        'Monthly Revenue': monthly_revenue,
        'Annual Revenue': annual_revenue,
        'ROI (%)': roi,
        'Payback Period (months)': payback_period
    }

def main():
    st.title('GPU ROI Calculator')

    try:
        # Load CSV directly from the same folder
        df = pd.read_csv('price.csv')

        # Sidebar controls
        st.sidebar.header('Parameters')

        # Cloud Type Selection (moved before GPU selection)
        cloud_type = st.sidebar.radio(
            'Select Cloud Type',
            ['Community', 'Secure'],
            index=1  # Default to 'Secure' (index 1)
        )

        # GPU Category and Selection
        st.sidebar.subheader('GPU Selection')

        # First select the GPU category
        gpu_category = st.sidebar.selectbox(
            'GPU Category',
            ['Data Center', 'Consumer', 'Workstation'],
            index=0  # Default to Data Center
        )

        # Filter GPUs based on category
        filtered_gpus = df[df['Card Type'] == gpu_category]

        # Set default GPU for each category
        default_gpus = {
            'Data Center': 'NVIDIA H100 NVL',
            'Consumer': 'NVIDIA GeForce RTX 4090',
            'Workstation': 'NVIDIA RTX 6000 Ada Generation'
        }

        # GPU Selection
        selected_gpu = st.sidebar.selectbox(
            'GPU Model',
            filtered_gpus['GPU Type'].unique(),
            index=filtered_gpus['GPU Type'].unique().tolist().index(default_gpus[gpu_category])
                if default_gpus[gpu_category] in filtered_gpus['GPU Type'].unique().tolist()
                else 0
        )

        # Owner Split Slider
        owner_split = st.sidebar.slider(
            'Owner Split (%)',
            min_value=0,
            max_value=100,
            value=70,
            step=10
        )

        # On-Demand vs Spot Slider (inverted from previous version)
        ondemand_ratio = st.sidebar.slider(
            'On-Demand Usage Ratio (%)',
            min_value=0,
            max_value=100,
            value=80,  # Default now set to 80%
            step=10
        ) / 100

        # Display Spot ratio for reference
        st.sidebar.info(f'Spot Usage Ratio: {100 - ondemand_ratio * 100:.0f}%')

        # Add platform fee slider in sidebar
        platform_fee = st.sidebar.slider(
            'Platform Fee (%)',
            min_value=0,
            max_value=100,
            value=24 if cloud_type == 'Community' else 20,  # Default based on cloud type
            step=1
        )

        # Get selected GPU data
        gpu_data = df[df['GPU Type'] == selected_gpu].iloc[0]
        default_purchase_price = float(gpu_data['Price NFT'])  # Convert to float

        # Add purchase price override
        st.sidebar.subheader('Purchase Price')
        use_custom_price = st.sidebar.checkbox('Override Purchase Price', False)
        custom_price = st.sidebar.number_input(
            'Custom Purchase Price ($)',
            min_value=0.0,
            value=float(default_purchase_price),  # Ensure float type
            step=1000.0,
            disabled=not use_custom_price
        )

        # Use custom price if override is enabled, otherwise use default
        purchase_price = custom_price if use_custom_price else default_purchase_price

        # Calculate metrics
        monthly_revenue = calculate_monthly_revenue(
            gpu_data,
            cloud_type,
            ondemand_ratio,
            owner_split,
            platform_fee
        )

        metrics = calculate_financial_metrics(purchase_price, monthly_revenue)

        # Display results
        st.header('Investment Analysis')

        col1, col2 = st.columns(2)

        with col1:
            st.metric('Purchase Price', f"${purchase_price:,.2f}")
            st.metric('Monthly Revenue', f"${metrics['Monthly Revenue']:,.2f}")

        with col2:
            st.metric('Annual Revenue', f"${metrics['Annual Revenue']:,.2f}")
            st.metric('ROI (%)', f"{metrics['ROI (%)']:.1f}%")

        st.metric('Payback Period',
                 f"{metrics['Payback Period (months)']:.1f} months" if metrics['Payback Period (months)'] != float('inf')
                 else "N/A")

        # Create cash flow projection
        months = range(0, 25)
        cumulative_cash_flow = [-purchase_price + monthly_revenue * month for month in months]

        # Create DataFrame for line chart
        cash_flow_df = pd.DataFrame({
            'Month': months,
            'Cumulative Cash Flow': cumulative_cash_flow,
            'Break Even': [0] * len(months)  # Reference line for break-even point
        })

        st.subheader('Projected Cumulative Cash Flow')
        st.line_chart(
            cash_flow_df,
            x='Month',
            y=['Cumulative Cash Flow', 'Break Even'],
            height=400
        )

        # Revenue Breakdown
        st.header('Revenue Breakdown')

        # Calculate monthly revenues for spot and on-demand
        hours_per_month = 24 * 30  # 720 hours per month for more accurate calculation
        if cloud_type == 'Community':
            spot_revenue = (gpu_data['Community Spot'] * hours_per_month * (1 - ondemand_ratio)) * (1 - platform_fee/100) * (owner_split/100)
            ondemand_revenue = (gpu_data['Community OnDemand'] * hours_per_month * ondemand_ratio) * (1 - platform_fee/100) * (owner_split/100)
        else:
            spot_revenue = (gpu_data['Secure Spot'] * hours_per_month * (1 - ondemand_ratio)) * (1 - platform_fee/100) * (owner_split/100)
            ondemand_revenue = (gpu_data['Secure OnDemand'] * hours_per_month * ondemand_ratio) * (1 - platform_fee/100) * (owner_split/100)

        revenue_df = pd.DataFrame({
            'Type': ['Spot', 'On-Demand'],
            'Monthly Revenue': [spot_revenue, ondemand_revenue]
        })

        col1, col2 = st.columns(2)
        with col1:
            st.metric('Spot Revenue', f"${spot_revenue:,.2f}")
            st.metric('On-Demand Revenue', f"${ondemand_revenue:,.2f}")

        with col2:
            # Create a bar chart for revenue breakdown
            st.bar_chart(revenue_df.set_index('Type'))

        # Risk Analysis
        st.header('Risk Analysis')

        # Calculate metrics for different utilization scenarios
        utilization_scenarios = {
            'Pessimistic (60%)': 0.6,
            'Expected (80%)': 0.8,
            'Optimistic (95%)': 0.95
        }

        scenario_data = []
        for scenario, utilization in utilization_scenarios.items():
            # Calculate spot and ondemand revenues separately with utilization factor
            if cloud_type == 'Community':
                adjusted_spot_revenue = gpu_data['Community Spot'] * hours_per_month * (1 - ondemand_ratio) * utilization * (1 - platform_fee/100) * (owner_split/100)
                adjusted_ondemand_revenue = gpu_data['Community OnDemand'] * hours_per_month * ondemand_ratio * utilization * (1 - platform_fee/100) * (owner_split/100)
            else:
                adjusted_spot_revenue = gpu_data['Secure Spot'] * hours_per_month * (1 - ondemand_ratio) * utilization * (1 - platform_fee/100) * (owner_split/100)
                adjusted_ondemand_revenue = gpu_data['Secure OnDemand'] * hours_per_month * ondemand_ratio * utilization * (1 - platform_fee/100) * (owner_split/100)

            adjusted_monthly_revenue = adjusted_spot_revenue + adjusted_ondemand_revenue

            scenario_data.append({
                'Scenario': scenario,
                'Monthly Revenue': adjusted_monthly_revenue,
                'Annual Revenue': adjusted_monthly_revenue * 12,
                'ROI (%)': (adjusted_monthly_revenue * 12 / purchase_price) * 100,
                'Payback (months)': purchase_price / adjusted_monthly_revenue
            })

        scenario_df = pd.DataFrame(scenario_data)

        # Display scenario metrics
        for scenario in scenario_data:
            st.metric(
                scenario['Scenario'],
                f"${scenario['Monthly Revenue']:,.2f}/month",
                f"ROI: {scenario['ROI (%)']:.1f}%"
            )

        # Create comparison chart for scenarios
        st.subheader('Scenario Comparison')
        comparison_df = pd.DataFrame({
            'Month': range(24),
            'Pessimistic': [-purchase_price + scenario_data[0]['Monthly Revenue'] * m for m in range(24)],
            'Expected': [-purchase_price + scenario_data[1]['Monthly Revenue'] * m for m in range(24)],
            'Optimistic': [-purchase_price + scenario_data[2]['Monthly Revenue'] * m for m in range(24)]
        })

        st.line_chart(
            comparison_df,
            x='Month',
            y=['Pessimistic', 'Expected', 'Optimistic'],
            height=400
        )

    except FileNotFoundError:
        st.error("Error: Could not find 'price.csv' in the current directory. Please ensure the file exists and is named correctly.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()