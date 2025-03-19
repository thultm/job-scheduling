import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import numpy as np


def visualize_fcfs_schedule(df, fig, ax):
    """Visualize the FCFS schedule with a Gantt chart"""
    # Define colors for jobs
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    y_position = 0
    y_ticks = []
    y_labels = []

    # Draw jobs in FCFS order
    for idx, job in df.iterrows():
        job_id = int(job['Job_ID'])
        release_time = job['Release_Date']
        start_time = job['Start_Time']
        duration = job['Processing_Time']
        completion_time = job['Completion_Time']
        due_date = job['Due_Date']

        # Choose color based on job index
        color = colors[job_id % len(colors)]

        # Draw waiting time (lighter shade)
        if start_time > release_time:
            waiting_rect = patches.Rectangle(
                (release_time, y_position),
                start_time - release_time,
                0.6,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.3
            )
            ax.add_patch(waiting_rect)
            ax.text(
                release_time + (start_time - release_time)/2,
                y_position + 0.3,
                'Wait',
                ha='center',
                va='center',
                fontsize=8
            )

        # Draw processing time (full color)
        process_rect = patches.Rectangle(
            (start_time, y_position),
            duration,
            0.6,
            linewidth=1,
            edgecolor='black',
            facecolor=color
        )
        ax.add_patch(process_rect)

        # Add job ID text in the center of the rectangle
        ax.text(
            start_time + duration/2,
            y_position + 0.3,
            f"Job {job_id}",
            color='white',
            fontweight='bold',
            ha='center',
            va='center'
        )

        # Add time markers
        ax.annotate(f"R:{int(release_time)}", (release_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"D:{int(due_date)}", (completion_time + 0.5, y_position + 0.2), ha='center', fontsize=8, color='green')
        ax.annotate(f"S:{int(start_time)}", (start_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"C:{int(completion_time)}", (completion_time, y_position-0.3), ha='center', fontsize=8)

        # Next job position
        y_position += 1
        y_ticks.append(y_position - 0.7)
        y_labels.append(f"Job {job_id}")

    # Set up the axis
    max_time = max(df['Completion_Time'].max(), df['Due_Date'].max()) + 2
    ax.set_xlim(-1, max_time)
    ax.set_ylim(-0.5, y_position)

    # Add time markers
    time_ticks = np.arange(0, max_time + 5, 5)
    ax.set_xticks(time_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add grid
    ax.grid(True, axis='x', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_title('FCFS (First Come First Serve) Schedule')

    plt.tight_layout()

def visualize_spt_schedule(df, fig, ax):
    """Visualize the SPT schedule with a Gantt chart"""
    # Define colors for jobs
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    y_position = 0
    y_ticks = []
    y_labels = []

    # Draw jobs in SPT order
    for idx, job in df.iterrows():
        job_id = int(job['Job_ID'])
        release_time = job['Release_Date']
        start_time = job['Start_Time']
        duration = job['Processing_Time']
        completion_time = job['Completion_Time']
        due_date = job['Due_Date']

        # Choose color based on job index
        color = colors[job_id % len(colors)]

        # Draw waiting time (lighter shade)
        if start_time > release_time:
            waiting_rect = patches.Rectangle(
                (release_time, y_position),
                start_time - release_time,
                0.6,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.3
            )
            ax.add_patch(waiting_rect)
            ax.text(
                release_time + (start_time - release_time)/2,
                y_position + 0.3,
                'Wait',
                ha='center',
                va='center',
                fontsize=8
            )

        # Draw processing time (full color)
        process_rect = patches.Rectangle(
            (start_time, y_position),
            duration,
            0.6,
            linewidth=1,
            edgecolor='black',
            facecolor=color
        )
        ax.add_patch(process_rect)

        # Add job ID text in the center of the rectangle
        ax.text(
            start_time + duration/2,
            y_position + 0.3,
            f"Job {job_id}",
            color='white',
            fontweight='bold',
            ha='center',
            va='center'
        )

        # Add time markers
        ax.annotate(f"R:{int(release_time)}", (release_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"D:{int(due_date)}", (completion_time + 0.5, y_position + 0.2), ha='center', fontsize=8, color='green')
        ax.annotate(f"S:{int(start_time)}", (start_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"C:{int(completion_time)}", (completion_time, y_position-0.3), ha='center', fontsize=8)

        # Next job position
        y_position += 1
        y_ticks.append(y_position - 0.7)
        y_labels.append(f"Job {job_id}")

    # Set up the axis
    max_time = max(df['Completion_Time'].max(), df['Due_Date'].max()) + 2
    ax.set_xlim(-1, max_time)
    ax.set_ylim(-0.5, y_position)

    # Add time markers
    time_ticks = np.arange(0, max_time + 5, 5)
    ax.set_xticks(time_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add grid
    ax.grid(True, axis='x', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_title('SPT (Shortest Processing Time) Schedule')

    plt.tight_layout()

def visualize_lpt_schedule(df, fig, ax):
    """Visualize the LPT schedule with a Gantt chart"""
    # Define colors for jobs
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    y_position = 0
    y_ticks = []
    y_labels = []

    # Draw jobs in LPT order
    for idx, job in df.iterrows():
        job_id = int(job['Job_ID'])
        release_time = job['Release_Date']
        start_time = job['Start_Time']
        duration = job['Processing_Time']
        completion_time = job['Completion_Time']
        due_date = job['Due_Date']

        # Choose color based on job index
        color = colors[job_id % len(colors)]

        # Draw waiting time (lighter shade)
        if start_time > release_time:
            waiting_rect = patches.Rectangle(
                (release_time, y_position),
                start_time - release_time,
                0.6,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.3
            )
            ax.add_patch(waiting_rect)
            ax.text(
                release_time + (start_time - release_time)/2,
                y_position + 0.3,
                'Wait',
                ha='center',
                va='center',
                fontsize=8
            )

        # Draw processing time (full color)
        process_rect = patches.Rectangle(
            (start_time, y_position),
            duration,
            0.6,
            linewidth=1,
            edgecolor='black',
            facecolor=color
        )
        ax.add_patch(process_rect)

        # Add job ID text in the center of the rectangle
        ax.text(
            start_time + duration/2,
            y_position + 0.3,
            f"Job {job_id}",
            color='white',
            fontweight='bold',
            ha='center',
            va='center'
        )

        # Add time markers
        ax.annotate(f"R:{int(release_time)}", (release_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"D:{int(due_date)}", (completion_time + 0.5, y_position + 0.2), ha='center', fontsize=8, color='green')
        ax.annotate(f"S:{int(start_time)}", (start_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"C:{int(completion_time)}", (completion_time, y_position-0.3), ha='center', fontsize=8)

        # Next job position
        y_position += 1
        y_ticks.append(y_position - 0.7)
        y_labels.append(f"Job {job_id}")

    # Set up the axis
    max_time = max(df['Completion_Time'].max(), df['Due_Date'].max()) + 2
    ax.set_xlim(-1, max_time)
    ax.set_ylim(-0.5, y_position)

    # Add time markers
    time_ticks = np.arange(0, max_time + 5, 5)
    ax.set_xticks(time_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add grid
    ax.grid(True, axis='x', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_title('LPT (Longest Processing Time) Schedule')

    plt.tight_layout()

def visualize_edd_schedule(df, fig, ax):
    """Visualize the EDD schedule with a Gantt chart"""
    # Define colors for jobs
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    y_position = 0
    y_ticks = []
    y_labels = []

    # Draw jobs in EDD order
    for idx, job in df.iterrows():
        job_id = int(job['Job_ID'])
        release_time = job['Release_Date']
        start_time = job['Start_Time']
        duration = job['Processing_Time']
        completion_time = job['Completion_Time']
        due_date = job['Due_Date']

        # Choose color based on job index
        color = colors[job_id % len(colors)]

        # Draw waiting time (lighter shade)
        if start_time > release_time:
            waiting_rect = patches.Rectangle(
                (release_time, y_position),
                start_time - release_time,
                0.6,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.3
            )
            ax.add_patch(waiting_rect)
            ax.text(
                release_time + (start_time - release_time)/2,
                y_position + 0.3,
                'Wait',
                ha='center',
                va='center',
                fontsize=8
            )

        # Draw processing time (full color)
        process_rect = patches.Rectangle(
            (start_time, y_position),
            duration,
            0.6,
            linewidth=1,
            edgecolor='black',
            facecolor=color
        )
        ax.add_patch(process_rect)

        # Add job ID text in the center of the rectangle
        ax.text(
            start_time + duration/2,
            y_position + 0.3,
            f"Job {job_id}",
            color='white',
            fontweight='bold',
            ha='center',
            va='center'
        )

        # Add time markers
        ax.annotate(f"R:{int(release_time)}", (release_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"D:{int(due_date)}", (completion_time + 0.5, y_position + 0.2), ha='center', fontsize=8, color='green')
        ax.annotate(f"S:{int(start_time)}", (start_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"C:{int(completion_time)}", (completion_time, y_position-0.3), ha='center', fontsize=8)

        # Next job position
        y_position += 1
        y_ticks.append(y_position - 0.7)
        y_labels.append(f"Job {job_id}")

    # Set up the axis
    max_time = max(df['Completion_Time'].max(), df['Due_Date'].max()) + 2
    ax.set_xlim(-1, max_time)
    ax.set_ylim(-0.5, y_position)

    # Add time markers
    time_ticks = np.arange(0, max_time + 5, 5)
    ax.set_xticks(time_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add grid
    ax.grid(True, axis='x', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_title('EDD (Earliest Due Date) Schedule')

    plt.tight_layout()

def visualize_erd_schedule(df, fig, ax):
    """Visualize the ERD schedule with a Gantt chart"""
    # Define colors for jobs
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    y_position = 0
    y_ticks = []
    y_labels = []

    # Draw jobs in ERD order
    for idx, job in df.iterrows():
        job_id = int(job['Job_ID'])
        release_time = job['Release_Date']
        start_time = job['Start_Time']
        duration = job['Processing_Time']
        completion_time = job['Completion_Time']
        due_date = job['Due_Date']

        # Choose color based on job index
        color = colors[job_id % len(colors)]

        # Draw waiting time (lighter shade)
        if start_time > release_time:
            waiting_rect = patches.Rectangle(
                (release_time, y_position),
                start_time - release_time,
                0.6,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.3
            )
            ax.add_patch(waiting_rect)
            ax.text(
                release_time + (start_time - release_time)/2,
                y_position + 0.3,
                'Wait',
                ha='center',
                va='center',
                fontsize=8
            )

        # Draw processing time (full color)
        process_rect = patches.Rectangle(
            (start_time, y_position),
            duration,
            0.6,
            linewidth=1,
            edgecolor='black',
            facecolor=color
        )
        ax.add_patch(process_rect)

        # Add job ID text in the center of the rectangle
        ax.text(
            start_time + duration/2,
            y_position + 0.3,
            f"Job {job_id}",
            color='white',
            fontweight='bold',
            ha='center',
            va='center'
        )

        # Add time markers
        ax.annotate(f"R:{int(release_time)}", (release_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"D:{int(due_date)}", (completion_time + 0.5, y_position + 0.2), ha='center', fontsize=8, color='green')
        ax.annotate(f"S:{int(start_time)}", (start_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"C:{int(completion_time)}", (completion_time, y_position-0.3), ha='center', fontsize=8)

        # Next job position
        y_position += 1
        y_ticks.append(y_position - 0.7)
        y_labels.append(f"Job {job_id}")

    # Set up the axis
    max_time = max(df['Completion_Time'].max(), df['Due_Date'].max()) + 2
    ax.set_xlim(-1, max_time)
    ax.set_ylim(-0.5, y_position)

    # Add time markers
    time_ticks = np.arange(0, max_time + 5, 5)
    ax.set_xticks(time_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add grid
    ax.grid(True, axis='x', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_title('ERD (Earliest Release Date) Schedule')

    plt.tight_layout()

def visualize_wspt_schedule(df, fig, ax):
    """Visualize the WSPT schedule with a Gantt chart"""
    # Define colors for jobs
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    y_position = 0
    y_ticks = []
    y_labels = []

    # Draw jobs in WSPT order
    for idx, job in df.iterrows():
        job_id = int(job['Job_ID'])
        release_time = job['Release_Date']
        start_time = job['Start_Time']
        duration = job['Processing_Time']
        completion_time = job['Completion_Time']
        due_date = job['Due_Date']

        # Choose color based on job index
        color = colors[job_id % len(colors)]

        # Draw waiting time (lighter shade)
        if start_time > release_time:
            waiting_rect = patches.Rectangle(
                (release_time, y_position),
                start_time - release_time,
                0.6,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.3
            )
            ax.add_patch(waiting_rect)
            ax.text(
                release_time + (start_time - release_time)/2,
                y_position + 0.3,
                'Wait',
                ha='center',
                va='center',
                fontsize=8
            )

        # Draw processing time (full color)
        process_rect = patches.Rectangle(
            (start_time, y_position),
            duration,
            0.6,
            linewidth=1,
            edgecolor='black',
            facecolor=color
        )
        ax.add_patch(process_rect)

        # Add job ID text in the center of the rectangle
        ax.text(
            start_time + duration/2,
            y_position + 0.3,
            f"Job {job_id}",
            color='white',
            fontweight='bold',
            ha='center',
            va='center'
        )

        # Add time markers
        ax.annotate(f"R:{int(release_time)}", (release_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"D:{int(due_date)}", (completion_time + 0.5, y_position + 0.2), ha='center', fontsize=8, color='green')
        ax.annotate(f"S:{int(start_time)}", (start_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"C:{int(completion_time)}", (completion_time, y_position-0.3), ha='center', fontsize=8)

        # Next job position
        y_position += 1
        y_ticks.append(y_position - 0.7)
        y_labels.append(f"Job {job_id}")

    # Set up the axis
    max_time = max(df['Completion_Time'].max(), df['Due_Date'].max()) + 2
    ax.set_xlim(-1, max_time)
    ax.set_ylim(-0.5, y_position)

    # Add time markers
    time_ticks = np.arange(0, max_time + 5, 5)
    ax.set_xticks(time_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add grid
    ax.grid(True, axis='x', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_title('WSPT (Weighted Shortest Processing Time) Schedule')

    plt.tight_layout()

def visualize_rand_schedule(df, fig, ax):
    """Visualize the RAND schedule with a Gantt chart"""
    # Define colors for jobs
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    y_position = 0
    y_ticks = []
    y_labels = []

    # Draw jobs in RAND order
    for idx, job in df.iterrows():
        job_id = int(job['Job_ID'])
        release_time = job['Release_Date']
        start_time = job['Start_Time']
        duration = job['Processing_Time']
        completion_time = job['Completion_Time']
        due_date = job['Due_Date']

        # Choose color based on job index
        color = colors[job_id % len(colors)]

        # Draw waiting time (lighter shade)
        if start_time > release_time:
            waiting_rect = patches.Rectangle(
                (release_time, y_position),
                start_time - release_time,
                0.6,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.3
            )
            ax.add_patch(waiting_rect)
            ax.text(
                release_time + (start_time - release_time)/2,
                y_position + 0.3,
                'Wait',
                ha='center',
                va='center',
                fontsize=8
            )

        # Draw processing time (full color)
        process_rect = patches.Rectangle(
            (start_time, y_position),
            duration,
            0.6,
            linewidth=1,
            edgecolor='black',
            facecolor=color
        )
        ax.add_patch(process_rect)

        # Add job ID text in the center of the rectangle
        ax.text(
            start_time + duration/2,
            y_position + 0.3,
            f"Job {job_id}",
            color='white',
            fontweight='bold',
            ha='center',
            va='center'
        )

        # Add time markers
        ax.annotate(f"R:{int(release_time)}", (release_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"D:{int(due_date)}", (completion_time + 0.5, y_position + 0.2), ha='center', fontsize=8, color='green')
        ax.annotate(f"S:{int(start_time)}", (start_time, y_position-0.3), ha='center', fontsize=8)
        ax.annotate(f"C:{int(completion_time)}", (completion_time, y_position-0.3), ha='center', fontsize=8)

        # Next job position
        y_position += 1
        y_ticks.append(y_position - 0.7)
        y_labels.append(f"Job {job_id}")

    # Set up the axis
    max_time = max(df['Completion_Time'].max(), df['Due_Date'].max()) + 2
    ax.set_xlim(-1, max_time)
    ax.set_ylim(-0.5, y_position)

    # Add time markers
    time_ticks = np.arange(0, max_time + 5, 5)
    ax.set_xticks(time_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add grid
    ax.grid(True, axis='x', linestyle='-', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_title('RAND (Random Sequence) Schedule')

    plt.tight_layout()


def visualize_moore_schedule(jobs_df, completed_indices, late_indices, fig, ax):
    """Visualize the Moore schedule with completed and late jobs"""
    # Define colors for jobs
    colors = ['red', 'orange', 'blue', 'green', 'gray', 'purple', 'brown', 'pink', 'cyan', 'magenta']

    # Start with completed jobs
    completed_jobs = jobs_df.loc[completed_indices].sort_values('Due_Date')

    current_time = 0
    y_position = 0

    # Draw completed jobs first (in sequence)
    for idx, job in completed_jobs.iterrows():
        job_id = int(job['Job_ID'])
        duration = job['Processing_Time']

        # Choose color based on job index
        color = colors[job_id % len(colors)]

        # Create rectangle for the job
        rect = patches.Rectangle((current_time, y_position), duration, 1,
                                linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)

        # Add job ID text in the center of the rectangle
        ax.text(current_time + duration/2, y_position + 0.5, f"Job {job_id}",
                color='white', fontweight='bold', ha='center', va='center')

        # Move time forward
        current_time += duration

    # Draw a vertical line at the end of completed jobs
    ax.axvline(x=current_time, color='navy', linestyle='-', linewidth=2)

    # Now draw late jobs arbitrarily after completed jobs
    late_jobs = jobs_df.loc[late_indices]

    for idx, job in late_jobs.iterrows():
        job_id = int(job['Job_ID'])
        duration = job['Processing_Time']

        # Choose color based on job index
        color = colors[job_id % len(colors)]

        # Create rectangle for the job
        rect = patches.Rectangle((current_time, y_position), duration, 1,
                                linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)

        # Add job ID text in the center of the rectangle
        ax.text(current_time + duration/2, y_position + 0.5, f"Job {job_id}",
                color='white', fontweight='bold', ha='center', va='center')

        # Move time forward
        current_time += duration

    # Set up the axis
    ax.set_xlim(0, current_time + 1)
    ax.set_ylim(0, 1.5)

    # Add time markers
    time_ticks = np.arange(0, current_time + 4, 2)
    ax.set_xticks(time_ticks)
    ax.set_yticks([])

    # Add grid
    ax.grid(True, axis='x', linestyle='-', alpha=0.3)

    plt.title('Moore Rule Schedule: Completed Jobs | Late Jobs')
    plt.tight_layout()

def visualize_srpt_timeline(timeline, release_dates, fig, ax):
    """Create a linear visualization of the SRPT schedule"""
    # Define colors for jobs - more distinct colors
    colors = ['red', 'orange', 'blue', 'green', 'gray', 'purple', 'brown', 'pink', 'cyan', 'magenta']
    
    # Get max time for setting axis limits
    max_time = max(segment['end'] for segment in timeline)
    
    # Plot timeline segments
    for segment in timeline:
        start = segment['start']
        end = segment['end']
        job_id = segment['job_id']
        
        # Color by job ID (indexed from 1)
        color_idx = (job_id - 1) % len(colors)
        color = colors[color_idx]
        
        # Create rectangle for this job segment
        rect = patches.Rectangle(
            (start, 0), 
            end - start, 
            1,
            linewidth=1,
            edgecolor='black',
            facecolor=color
        )
        ax.add_patch(rect)
        
        # Add job ID text inside the rectangle
        ax.text(
            start + (end - start)/2,
            0.5,
            f"J{job_id}",
            color='white',
            fontweight='bold',
            ha='center',
            va='center',
            fontsize=10
        )
    
    # Identify preemption points and create flags
    job_segments = {}
    for segment in timeline:
        job_id = segment['job_id']
        if job_id not in job_segments:
            job_segments[job_id] = []
        job_segments[job_id].append((segment['start'], segment['end']))
    
    # Look for non-contiguous segments for each job
    for job_id, segments in job_segments.items():
        # Sort segments by start time
        segments.sort()
        
        # Check for preemptions (non-contiguous segments)
        for i in range(len(segments) - 1):
            if segments[i][1] != segments[i+1][0]:  # If end of segment != start of next segment
                preemption_time = segments[i][1]
                
                # Draw flag pole
                ax.plot([preemption_time, preemption_time], [1, 1.5], 'k-', linewidth=1.5)
                
                # Draw flag
                flag = patches.Rectangle(
                    (preemption_time, 1.5),
                    1.5,
                    0.5,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='darkgreen'
                )
                ax.add_patch(flag)
                
                # Add job ID to flag
                ax.text(
                    preemption_time + 0.75,
                    1.75,
                    f"J{job_id}",
                    color='white',
                    fontweight='bold',
                    ha='center',
                    va='center',
                    fontsize=9
                )
    
    # Add release time markers (red dashed lines)
    for t in sorted(set(release_dates)):
        ax.axvline(x=t, color='red', linestyle='--', linewidth=1)
    
    # Set up the axis
    ax.set_xlim(-1, max_time + 3)
    ax.set_ylim(-0.2, 2.5)
    
    # Add time markers on x-axis
    time_range = np.arange(0, max_time + 4, 2)
    ax.set_xticks(time_range)
    ax.set_yticks([])
    
    # Add grid
    ax.grid(True, axis='x', linestyle='-', alpha=0.3)
    
    # Remove y-axis
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add title
    ax.set_title('SRPT (Shortest Remaining Processing Time) Schedule', fontsize=14)
    
    plt.tight_layout()
    
    return fig, ax

def visualize_branch_bound_tree(G, fig_size=(15, 10)):
    """
    Improved Branch and Bound tree visualization showing the optimal path
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Get node levels
    levels = nx.get_node_attributes(G, 'level')
    max_level = max(levels.values())
    
    # Group nodes by level
    nodes_by_level = {}
    for node, level in levels.items():
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)
    
    # Calculate positions with improved spacing
    pos = {}
    h_space = 3.0  # Horizontal spacing factor
    v_space = 2.0  # Vertical spacing factor
    
    for level in range(max_level + 1):
        if level in nodes_by_level:
            nodes = sorted(nodes_by_level[level])
            width = len(nodes)
            
            # Apply spacing
            for i, node in enumerate(nodes):
                x = (i - width / 2 + 0.5) * h_space
                y = -level * v_space
                pos[node] = (x, y)
    
    # Identify different types of nodes - PATH-BASED CATEGORIZATION
    best_nodes = []           # Part of the optimal solution path
    path_nodes = []           # Nodes that led to further exploration but aren't optimal
    terminal_nodes = []       # Nodes that didn't lead anywhere (leaf nodes or pruned)
    
    for node in G.nodes():
        is_best = nx.get_node_attributes(G, 'is_best').get(node, False)
        
        if is_best:
            best_nodes.append(node)
        else:
            # Check if this node has children (successors)
            successors = list(G.successors(node))
            
            if successors:  # If node has children, it's part of the exploration path
                path_nodes.append(node)
            else:  # If no children, it's a terminal node
                terminal_nodes.append(node)
    
    # Draw the graph in order
    
    # 1. Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # 2. Draw terminal nodes 
    if terminal_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes, 
                                node_color='lightgray', 
                                node_size=900, 
                                alpha=0.9)
    
    # 3. Draw path nodes
    if path_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, 
                                node_color='skyblue',  
                                node_size=1000, 
                                alpha=1.0)
    
    # 4. Draw best solution nodes (ENTIRE optimal path)
    if best_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=best_nodes, 
                                node_color='lightgreen', 
                                node_size=1000, 
                                alpha=1.0)
    
    # Get attributes for labels
    labels = nx.get_node_attributes(G, 'label')
    sequences = nx.get_node_attributes(G, 'sequence')
    lmaxes = nx.get_node_attributes(G, 'lmax')
    
    # Create custom labels with sequence and bound
    node_labels = {}
    for node in G.nodes():
        sequence = sequences[node]
        bound = lmaxes[node]
        job_label = labels[node]
        
        if len(sequence) == 0:
            node_labels[node] = f"∅\nBound={bound}"
        else:
            node_labels[node] = f"Job {job_label}\nBound={bound}"
    
    # 5. Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    
    # Add a legend with matching colors
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, 
                label='Active Path Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=15, 
                label='Terminal/End Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, 
                label='Optimal Solution Path')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add a title
    ax.set_title("Branch and Bound Tree for 1|rj|Lmax", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    return fig, ax