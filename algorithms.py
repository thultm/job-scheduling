import pandas as pd
import streamlit as st
import heapq
import networkx as nx

from visualization import *
from old_visualization import *
from utils import *

# Define FCFS function
def FCFS(jobs_df):
    # Create Streamlit containers
    result_container = st.container()
    figure_container = st.container()

    # Make a copy (no sorting - take jobs in the order they appear)
    df = jobs_df.copy()
    
    # Process jobs
    df = process_jobs(df)
    
    # Calculate metrics
    makespan, results_table, summary_table, additional_table = calculate_metrics(df)
    
    # Display results
    display_results("FCFS", df, makespan, results_table, summary_table, additional_table,
                   result_container, figure_container)

# Define SPT function
def SPT(jobs_df):
    # Create Streamlit containers
    result_container = st.container()
    figure_container = st.container()

    # Core algorithm: Sort by processing time
    df = jobs_df.copy().sort_values('Processing_Time')
    
    # Process jobs
    df = process_jobs(df)
    
    # Calculate metrics
    makespan, results_table, summary_table, additional_table = calculate_metrics(df)
    
    # Display results
    display_results("SPT", df, makespan, results_table, summary_table, additional_table,
                   result_container, figure_container)

def LPT(jobs_df):
    # Create Streamlit containers
    result_container = st.container()
    figure_container = st.container()
    # Make a copy (sorting by the decreasing processing time)
    df = jobs_df.copy().sort_values('Processing_Time', ascending=False)

    # Process jobs
    df = process_jobs(df)
    
    # Calculate metrics
    makespan, results_table, summary_table, additional_table = calculate_metrics(df)
    
    # Display results
    display_results("SPT", df, makespan, results_table, summary_table, additional_table,
                   result_container, figure_container)

def EDD(jobs_df):
    # Create Streamlit containers
    result_container = st.container()
    figure_container = st.container()
    # Make a copy (sorting by the increasing due_date)
    df = jobs_df.copy().sort_values('Due_Date')

    # Process jobs
    df = process_jobs(df)
    
    # Calculate metrics
    makespan, results_table, summary_table, additional_table = calculate_metrics(df)
    
    # Display results
    display_results("SPT", df, makespan, results_table, summary_table, additional_table,
                   result_container, figure_container)

def ERD(jobs_df):
    # Create Streamlit containers
    result_container = st.container()
    figure_container = st.container()
    # Make a copy (sorting by the increasing release_date)
    df = jobs_df.copy().sort_values('Release_Date')

    # Process jobs
    df = process_jobs(df)
    
    # Calculate metrics
    makespan, results_table, summary_table, additional_table = calculate_metrics(df)
    
    # Display results
    display_results("SPT", df, makespan, results_table, summary_table, additional_table,
                   result_container, figure_container)

def WSPT(jobs_df):
    # Create Streamlit containers
    result_container = st.container()
    figure_container = st.container()
    # Make a copy and calculate pj/wj ratio
    df = jobs_df.copy()
    df['pj/wj'] = round(df['Processing_Time']/df['Weight'], 2)
    
    # Sort by the increasing pj/wj ratio
    df = df.sort_values('pj/wj')

    # Process jobs
    df = process_jobs(df)
    
    # Calculate metrics
    makespan, results_table, summary_table, additional_table = calculate_metrics(df)
    
    # Display results
    display_results("SPT", df, makespan, results_table, summary_table, additional_table,
                   result_container, figure_container)

def RAND(jobs_df):
    # Create Streamlit containers
    result_container = st.container()
    figure_container = st.container()
    # Make a copy and randomly shuffle the jobs
    df = jobs_df.sample(frac=1).reset_index(drop=True)

    # Process jobs
    df = process_jobs(df)
    
    # Calculate metrics
    makespan, results_table, summary_table, additional_table = calculate_metrics(df)
    
    # Display results
    display_results("SPT", df, makespan, results_table, summary_table, additional_table,
                   result_container, figure_container)

def moore_rule(jobs_df):
    # Create Streamlit containers
    result_container = st.container()
    figure_container = st.container()
    # Make a copy and sort by due date (ascending)
    df = jobs_df.copy().sort_values('Due_Date')

    # Initialize variables
    current_time = 0
    scheduled_indices = []

    # Process each task in order of due date
    for idx, task in df.iterrows():
        # Add this task to our schedule
        scheduled_indices.append(idx)
        current_time += task['Processing_Time']

        # If we've missed the due date
        if current_time > task['Due_Date']:
            # Find and remove the task with the longest processing time
            longest_idx = max(
                [(i, df.loc[i, 'Processing_Time']) for i in scheduled_indices],
                key=lambda x: x[1]
            )[0]

            scheduled_indices.remove(longest_idx)
            current_time -= df.loc[longest_idx, 'Processing_Time']

    # Calculate late jobs
    all_indices = df.index.tolist()
    late_indices = [idx for idx in all_indices if idx not in scheduled_indices]

    # Display results in Streamlit
    with result_container:
        st.subheader("Results: Moore's Rule")
        st.write(f"Number of jobs completed on time: {len(jobs_df.loc[scheduled_indices])} {jobs_df.loc[scheduled_indices]['Job_ID'].tolist()}")
        st.write(f"Number of late jobs (Objective value): {len(jobs_df.loc[late_indices])} {jobs_df.loc[late_indices]['Job_ID'].tolist()}")

    # Create visualization
    with figure_container:
        fig = visualize_moore(jobs_df, scheduled_indices, late_indices)
        st.plotly_chart(fig, use_container_width=True)

def SRPT(jobs_df):
    # Create Streamlit containers
    result_container = st.container()
    figure_container_1 = st.container()
    figure_container_2 = st.container()
    # Make a copy of the dataframe
    df = jobs_df.copy()
    
    # Create a column to track remaining processing time
    df['Remaining_Time'] = df['Processing_Time'].copy()
    
    # Sort by release date
    sorted_jobs = df.sort_values('Release_Date').reset_index(drop=True)
    
    # List to store execution timeline
    timeline = []
    
    # Keep track of current time and events
    current_time = sorted_jobs['Release_Date'].min()
    next_release_times = sorted_jobs['Release_Date'].tolist()
    next_release_times.sort()
    
    # Keep track of available and completed jobs
    available_jobs = []  # (job_id, remaining_time)
    completed_jobs = set()
    
    # Process until all jobs are completed
    while len(completed_jobs) < len(df):
        # Add any newly released jobs to available pool
        newly_released = sorted_jobs[(sorted_jobs['Release_Date'] <= current_time) & 
                                    (~sorted_jobs['Job_ID'].isin([j[0] for j in available_jobs])) &
                                    (~sorted_jobs['Job_ID'].isin(completed_jobs))]
        
        for _, job in newly_released.iterrows():
            available_jobs.append((job['Job_ID'], job['Remaining_Time']))
        
        # If no available jobs, jump to next release time
        if not available_jobs:
            future_releases = [t for t in next_release_times if t > current_time]
            if future_releases:
                current_time = future_releases[0]
                continue
            else:
                break  # No more jobs to process
        
        # Find job with shortest remaining time
        available_jobs.sort(key=lambda x: x[1])  # Sort by remaining time
        current_job_id, current_job_remaining = available_jobs[0]
        
        # Determine how long to run this job
        # Find the next event (job completion or new release)
        next_release = min([t for t in next_release_times if t > current_time], default=float('inf'))
        job_completion_time = current_time + current_job_remaining
        
        # The next event is either job completion or next release, whichever comes first
        next_event_time = min(job_completion_time, next_release)
        duration = next_event_time - current_time
        
        # Execute the job for this duration
        timeline.append({
            'job_id': current_job_id,
            'start': current_time,
            'end': next_event_time
        })
        
        # Update remaining time
        job_index = sorted_jobs[sorted_jobs['Job_ID'] == current_job_id].index[0]
        sorted_jobs.loc[job_index, 'Remaining_Time'] -= duration
        
        # Update available jobs list
        if next_event_time == job_completion_time:
            # Job completed
            completed_jobs.add(current_job_id)
            available_jobs.pop(0)  # Remove this job
        else:
            # Job preempted, update its remaining time
            available_jobs[0] = (current_job_id, current_job_remaining - duration)
        
        # Advance time
        current_time = next_event_time
    
    # Calculate job metrics for display
    job_metrics = {}
    for segment in timeline:
        job_id = segment['job_id']
        if job_id not in job_metrics:
            job_metrics[job_id] = {
                'start_time': float('inf'),
                'completion_time': 0,
                'total_processing_time': sorted_jobs[sorted_jobs['Job_ID'] == job_id]['Processing_Time'].values[0]
            }
        
        # Update job metrics
        job_metrics[job_id]['start_time'] = min(job_metrics[job_id]['start_time'], segment['start'])
        job_metrics[job_id]['completion_time'] = max(job_metrics[job_id]['completion_time'], segment['end'])
    
    # Calculate additional metrics
    for job_id, metrics in job_metrics.items():
        job_row = sorted_jobs[sorted_jobs['Job_ID'] == job_id].iloc[0]
        metrics['release_time'] = job_row['Release_Date']
        metrics['due_date'] = job_row['Due_Date']
        metrics['flow_time'] = metrics['completion_time'] - metrics['release_time']
        metrics['lateness'] = metrics['completion_time'] - metrics['due_date']
        metrics['tardiness'] = max(0, metrics['lateness'])
        metrics['waiting_time'] = metrics['flow_time'] - metrics['total_processing_time']

    # Display results in Streamlit
    with result_container:
        st.subheader("Results: SRPT")
        
        # Show timeline info
        makespan = max(segment['end'] for segment in timeline)
        st.write(f"Makespan (total completion time): {makespan}")
        
        # Show preemption info
        preemptions = {}
        for job_id in job_metrics:
            segments = [seg for seg in timeline if seg['job_id'] == job_id]
            preemption_count = len(segments) - 1
            if preemption_count > 0:
                preemptions[job_id] = preemption_count
        
        if preemptions:
            st.subheader("Preemptions:")
            for job_id, count in preemptions.items():
                st.write(f"Job {job_id} was preempted {count} times")
        else:
            st.write("No jobs were preempted")
    
    # Visualize the timeline
    with figure_container_1:
        fig = visualize_srpt_plotly(timeline, df)
        st.plotly_chart(fig, use_container_width=True)

    with figure_container_2:
        fig = visualize_srpt_detail_plotly(timeline, df)
        st.plotly_chart(fig, use_container_width=True)
    
    # return timeline, job_metrics

def branch_and_bound_lmax(jobs_df):
    """
    Branch and Bound algorithm for 1|rj|Lmax using preemptive EDD lower bound
    with tracking of explored nodes
    """
    # Create a copy of the dataframe
    df = jobs_df.copy().reset_index(drop=True)
    
    # Create a node class for the branch and bound tree
    class Node:
        def __init__(self, level, sequence, completion_time, lmax, parent=None):
            self.id = None  # Will be assigned during tree building
            self.level = level
            self.sequence = sequence.copy()
            self.completion_time = completion_time
            self.lmax = lmax  # Max lateness of the scheduled jobs
            self.parent = parent
            self.children = []
            self.pruned = False
            self.is_best = False
            self.bound = float('inf')  # Store the bound separately
            self.explored = False  # Track if the node was explored during search
            
        def __lt__(self, other):
            return self.bound < other.bound
    
    # Check dominance rule: rj < min_{l∈J} (max(t,rl) + pl)
    def is_dominated(job_id, unscheduled, current_time):
        job = df[df['Job_ID'] == job_id].iloc[0]
        r_j = job['Release_Date']
        
        for other_id in unscheduled:
            if other_id == job_id:
                continue
                
            other_job = df[df['Job_ID'] == other_id].iloc[0]
            r_l = other_job['Release_Date']
            p_l = other_job['Processing_Time']
            
            # Check dominance condition
            if r_j >= max(current_time, r_l) + p_l:
                return True
                
        return False
    
    # Calculate preemptive EDD lower bound
    def preemptive_edd_bound(scheduled, current_time):
        # Calculate max lateness of scheduled jobs
        scheduled_lateness = 0
        if scheduled:
            t = 0
            for job_id in scheduled:
                job = df[df['Job_ID'] == job_id].iloc[0]
                t = max(t, job['Release_Date']) + job['Processing_Time']
                lateness = t - job['Due_Date']
                scheduled_lateness = max(scheduled_lateness, lateness)
        
        # Get unscheduled jobs
        unscheduled = [j for j in df['Job_ID'] if j not in scheduled]
        if not unscheduled:
            return scheduled_lateness
            
        # Create job data for unscheduled jobs
        jobs = []
        for job_id in unscheduled:
            job = df[df['Job_ID'] == job_id].iloc[0]
            jobs.append({
                'id': job_id,
                'release': job['Release_Date'],
                'processing': job['Processing_Time'],
                'due': job['Due_Date'],
                'remaining': job['Processing_Time']
            })
        
        # Run preemptive EDD
        t = current_time
        lateness = scheduled_lateness
        
        while any(job['remaining'] > 0 for job in jobs):
            # Get available jobs
            available = [j for j in jobs if j['release'] <= t and j['remaining'] > 0]
            
            if not available:
                # Jump to next release time
                next_time = min(j['release'] for j in jobs if j['remaining'] > 0)
                t = next_time
                continue
                
            # Sort by EDD
            available.sort(key=lambda x: x['due'])
            
            # Process the job with earliest due date
            current_job = available[0]
            
            # Find the next event (release or completion)
            next_releases = [j['release'] for j in jobs if j['release'] > t and j['remaining'] > 0]
            
            if next_releases:
                next_event = min(next_releases)
                process_time = min(current_job['remaining'], next_event - t)
                current_job['remaining'] -= process_time
                t += process_time
            else:
                # No more releases, process to completion
                t += current_job['remaining']
                current_job['remaining'] = 0
                
            # Update lateness when a job completes
            if current_job['remaining'] == 0:
                job_lateness = t - current_job['due']
                lateness = max(lateness, job_lateness)
        
        return lateness
    
    # Create root node
    root = Node(level=0, sequence=[], completion_time=0, lmax=0)
    root.id = 0
    root.bound = preemptive_edd_bound([], 0)
    root.explored = True  # Root is always explored
    
    # Best solution found so far
    best_solution = None
    best_lmax = float('inf')
    
    # Use priority queue for best-first search
    queue = [root]
    heapq.heapify(queue)
    
    # Keep track of all nodes for visualization
    all_nodes = {0: root}
    next_id = 1
    
    # Start branch and bound
    with st.spinner("Running Branch and Bound algorithm..."):
        while queue:
            current = heapq.heappop(queue)
            
            # Mark this node as explored (popped from queue)
            current.explored = True
            
            # Skip if already pruned
            if current.pruned:
                continue
                
            # If we've found a complete solution
            if current.level == len(df):
                if current.lmax < best_lmax:
                    if best_solution:
                        best_solution.is_best = False
                    best_lmax = current.lmax
                    best_solution = current
                    current.is_best = True
                    
                    # Important: Retrospectively prune nodes with bounds >= best_lmax
                    for _, node in all_nodes.items():
                        if not node.is_best and node.bound >= best_lmax:
                            node.pruned = True
                continue
                
            # Get unscheduled jobs
            unscheduled = [j for j in df['Job_ID'] if j not in current.sequence]
            
            # Generate children according to dominance rule
            for job_id in unscheduled:
                if is_dominated(job_id, unscheduled, current.completion_time):
                    continue
                    
                # Get job info
                job = df[df['Job_ID'] == job_id].iloc[0]
                
                # Calculate completion time
                start_time = max(current.completion_time, job['Release_Date'])
                completion_time = start_time + job['Processing_Time']
                
                # Calculate maximum lateness of scheduled jobs
                lateness = completion_time - job['Due_Date']
                new_lmax = max(current.lmax, lateness)
                
                # Create new sequence
                new_sequence = current.sequence + [job_id]
                
                # Create child node
                child = Node(
                    level=current.level + 1,
                    sequence=new_sequence,
                    completion_time=completion_time,
                    lmax=new_lmax,
                    parent=current
                )
                child.id = next_id
                next_id += 1
                
                # Add child to parent
                current.children.append(child)
                
                # Store for visualization
                all_nodes[child.id] = child
                
                # Calculate lower bound using preemptive EDD
                bound = preemptive_edd_bound(new_sequence, completion_time)
                child.bound = max(new_lmax, bound)  # Store the bound
                
                # Check if we can prune immediately
                if child.bound >= best_lmax:
                    child.pruned = True
                    continue
                    
                # Add to queue
                heapq.heappush(queue, child)
    
    # After finding the best solution, mark all nodes in the optimal path
    if best_solution:
        # Trace back from the optimal solution to the root
        current = best_solution
        while current:
            current.is_best = True
            current = current.parent
    
    # Create a tree for visualization
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node in all_nodes.items():
        # Generate node label
        if node.level == 0:
            label = "∅"
        else:
            label = str(node.sequence[-1])
            
        # Add node with attributes
        G.add_node(
            node_id, 
            level=node.level,
            sequence=node.sequence,
            lmax=node.bound,  # Use bound as the displayed value
            pruned=node.pruned,
            is_best=node.is_best,
            explored=node.explored,  # Include the explored flag
            label=label
        )
    
    # Add edges
    for node_id, node in all_nodes.items():
        for child in node.children:
            G.add_edge(node_id, child.id)
    
    # Return results
    if best_solution:
        return best_solution.sequence, best_lmax, G, all_nodes
    else:
        return None, float('inf'), G, all_nodes