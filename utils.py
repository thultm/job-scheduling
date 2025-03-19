import pandas as pd
import streamlit as st
import numpy as np
from visualization import visualize_schedule

def process_jobs(df):
    """
    Process jobs in the order provided in the dataframe and calculate metrics
    
    Args:
        df: Dataframe with jobs (already sorted according to algorithm rule)
        
    Returns:
        Processed dataframe with additional metrics columns
    """
    # Initialize variables
    current_time = 0
    start_times = []
    completion_times = []
    waiting_times = []
    flow_times = []
    lateness_values = []
    tardiness_values = []

    # Process each job in the order they appear
    for idx, job in df.iterrows():
        # Update current time (job can't start before its release date)
        current_time = max(current_time, job['Release_Date'])

        # Calculate start time for this job
        start_time = current_time
        start_times.append(start_time)

        # Process the job
        current_time += job['Processing_Time']
        completion_time = current_time
        completion_times.append(completion_time)

        # Calculate metrics
        waiting_time = start_time - job['Release_Date']
        flow_time = completion_time - job['Release_Date']
        lateness = completion_time - job['Due_Date']
        tardiness = max(0, completion_time - job['Due_Date'])

        # Save results
        waiting_times.append(waiting_time)
        flow_times.append(flow_time)
        lateness_values.append(lateness)
        tardiness_values.append(tardiness)

    # Add results to the dataframe
    df['Start_Time'] = start_times
    df['Completion_Time'] = completion_times
    df['Wait_Time'] = waiting_times
    df['Flow_Time'] = flow_times
    df['Lateness'] = lateness_values
    df['Tardiness'] = tardiness_values
    
    return df

def calculate_metrics(df):
    """
    Calculate summary metrics from processed jobs
    
    Args:
        df: Processed dataframe with job metrics
        
    Returns:
        Tuple of (makespan, results_table, summary_table, additional_table)
    """
    # Calculate total and average metrics
    completion_times = df['Completion_Time'].tolist()
    waiting_times = df['Wait_Time'].tolist()
    flow_times = df['Flow_Time'].tolist()
    lateness_values = df['Lateness'].tolist()
    tardiness_values = df['Tardiness'].tolist()
    
    total_completion_time = sum(completion_times)
    avg_completion_time = round(total_completion_time / len(df), 2)
    total_waiting_time = sum(waiting_times)
    avg_waiting_time = round(total_waiting_time / len(df), 2)
    total_flow_time = sum(flow_times)
    avg_flow_time = round(total_flow_time / len(df), 2)
    total_lateness = sum(lateness_values)
    avg_lateness = round(total_lateness / len(df), 2)
    total_tardiness = sum(tardiness_values)
    avg_tardiness = round(total_tardiness / len(df), 2)
    max_tardiness = max(tardiness_values)
    makespan = max(completion_times)

    # Calculate other metrics
    total_processing_time = sum(df['Processing_Time'])
    utilization = round((total_processing_time / total_flow_time) * 100, 2) if total_flow_time > 0 else 0
    avg_jobs_in_system = round(total_flow_time / total_processing_time, 2) if total_processing_time > 0 else 0
    num_tardy_jobs = sum(1 for t in tardiness_values if t > 0)

    # Weighted metrics
    weighted_completion_times = [c * w for c, w in zip(completion_times, df['Weight'])]
    weighted_flow_times = [f * w for f, w in zip(flow_times, df['Weight'])]
    weighted_wait_times = [wt * w for wt, w in zip(waiting_times, df['Weight'])]
    weighted_tardiness = [t * w for t, w in zip(tardiness_values, df['Weight'])]

    sum_weighted_completion = sum(weighted_completion_times)
    sum_weighted_flow = sum(weighted_flow_times)
    sum_weighted_wait = sum(weighted_wait_times)
    sum_weighted_tardiness = sum(weighted_tardiness)
    
    # Create results table
    results_table = df[['Job_ID', 'Start_Time', 'Completion_Time', 'Wait_Time',
                      'Flow_Time', 'Lateness', 'Tardiness']]
    
    # Create summary table
    summary_data = {
        'Metric': ['Completion Time', 'Wait Time', 'Flow Time', 'Lateness', 'Tardiness'],
        'Total': [total_completion_time, total_waiting_time, total_flow_time, total_lateness, total_tardiness],
        'Average': [avg_completion_time, avg_waiting_time, avg_flow_time, avg_lateness, avg_tardiness],
        'Maximum': [max(completion_times), max(waiting_times), max(flow_times), max(lateness_values), max_tardiness]
    }
    summary_table = pd.DataFrame(summary_data)
    
    # Create additional metrics table
    additional_metrics = {
        'Metric': ['Utilization (%)', 'Avg Jobs in System', 'Number of Tardy Jobs', 
                  'Weighted Completion Time', 'Weighted Flow Time', 'Weighted Wait Time', 'Weighted Tardiness'],
        'Value': [utilization, avg_jobs_in_system, num_tardy_jobs,
                sum_weighted_completion, sum_weighted_flow, sum_weighted_wait, sum_weighted_tardiness]
    }
    additional_table = pd.DataFrame(additional_metrics)
    
    return makespan, results_table, summary_table, additional_table

def display_results(algorithm_name, df, makespan, results_table, summary_table, additional_table, 
                   result_container, figure_container):
    """
    Display results in Streamlit containers
    
    Args:
        algorithm_name: Name of the scheduling algorithm
        df: Processed dataframe
        makespan: Maximum completion time
        results_table: Table with job metrics
        summary_table: Table with summary statistics
        additional_table: Table with additional metrics
        result_container: Streamlit container for results
        figure_container: Streamlit container for visualization
    """
    with result_container:
        st.subheader(f"Results: {algorithm_name}")
        st.write(f"Job sequence: {df['Job_ID'].tolist()}")
        st.write(f"Makespan (total completion time): {makespan}")

        st.subheader("Detailed Job Metrics:")
        st.dataframe(results_table, hide_index=True)

        st.subheader("Summary Statistics:")
        st.dataframe(summary_table, hide_index=True)
        
        st.subheader("Additional Metrics:")
        st.dataframe(additional_table, hide_index=True)
    
    with figure_container:
        fig = visualize_schedule(df, title=f"{algorithm_name} Schedule")
        st.plotly_chart(fig, use_container_width=True)