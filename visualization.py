import plotly.graph_objects as go


def visualize_schedule(df, title="Job Schedule", timeline_type="simple", color_scheme="default"):
    """
    Create an interactive Gantt chart using Plotly
    
    Parameters:
    df: Dataframe with job data including Job_ID, Release_Date, Start_Time, Processing_Time, 
        Completion_Time, Due_Date
    title: Chart title
    timeline_type: 'simple' for normal job schedule, 'preemptive' for SRPT
    color_scheme: 'default', 'pastel', 'vivid', or 'categorical' for different color palettes
    
    Returns:
    fig: Plotly figure object
    """
    # Create figure with black background
    fig = go.Figure()
    
    # Enhanced color schemes with better contrast
    color_palettes = {
        "default": [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ],
        "pastel": [
            '#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9',
            '#fff2ae', '#f1e2cc', '#cccccc', '#d9d9d9', '#fddaec',
            '#dbc9eb', '#ffd8b1', '#e2f4c7', '#ffffcc', '#f2f2f2'
        ],
        "vivid": [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
            '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'
        ],
        "categorical": [
            '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
            '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
            '#ccebc5', '#ffed6f', '#a6cee3', '#1f78b4', '#b2df8a'
        ],
        # New high-contrast scheme
        "high_contrast": [
            '#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE',
            '#EE3377', '#BBBBBB', '#FFFFFF', '#FFDD00', '#99CC00'
        ]
    }
    
    # Select color palette (default to high_contrast if not specified)
    colors = color_palettes.get(color_scheme, color_palettes["high_contrast"])
    
    # Sort jobs for better visualization - by Job_ID for consistency
    # df_sorted = df.sort_values(['Start_Time', 'Job_ID'])
    df_sorted = df.sort_values('Completion_Time', ascending=True)
    
    # Calculate max completion and due time for x-axis range
    max_completion = df['Completion_Time'].max()
    max_due_date = df['Due_Date'].max()
    max_x = max(max_completion, max_due_date) * 1.1
    
    # Track lateness statistics for annotations
    late_jobs = []
    waiting_times = []
    total_processing_time = 0
    
    # Get unique jobs for better y-axis ordering
    # job_ids = sorted(df_sorted['Job_ID'].unique())
    job_ids = df_sorted['Job_ID'].unique()
    y_labels = [f"Job {job_id}" for job_id in job_ids]
    
    # For each job, add bars for waiting and processing
    for i, job in df_sorted.iterrows():
        job_id = int(job['Job_ID'])
        release_time = job['Release_Date']
        start_time = job['Start_Time']
        processing_time = job['Processing_Time']
        completion_time = job['Completion_Time']
        due_date = job['Due_Date']
        lateness = completion_time - due_date
        
        # Track statistics
        total_processing_time += processing_time
        
        # Color based on job ID with more saturation
        color = colors[(job_id - 1) % len(colors)]
        
        # Track late jobs for annotations
        if lateness > 0:
            late_jobs.append((job_id, lateness))
        
        # Add waiting time bar (if waiting occurred)
        if start_time > release_time:
            waiting_time = start_time - release_time
            waiting_times.append(waiting_time)
            
            # Use diagonal lines pattern for waiting time
            fig.add_trace(go.Bar(
                x=[waiting_time],
                y=[f"Job {job_id}"],
                orientation='h',
                base=[release_time],
                marker=dict(
                    color=color, 
                    opacity=0.3,  # More transparent
                    line=dict(color='rgba(255,255,255,0.5)', width=1),
                    pattern=dict(shape="/", solidity=0.4)  # Diagonal lines pattern
                ),
                name=f"Job {job_id} Waiting",
                text="Wait" if waiting_time > max_x/25 else "",  # Dynamic text threshold
                textposition="inside",
                textfont=dict(color='rgba(255,255,255,0.8)', size=10),
                showlegend=False,
                hoverinfo='text',
                hovertext=f"<b>Job {job_id} - Waiting</b><br>Duration: {waiting_time}<br>Release: {release_time}<br>Start: {start_time}"
            ))
        
        # Add processing time bar with conditional formatting for lateness
        is_late = lateness > 0
        border_color = '#FF5555' if is_late else 'rgba(255,255,255,0.7)'  # Brighter red for late jobs
        border_width = 2 if is_late else 1
        
        # Dynamic text size based on processing time
        text_size = min(14, max(9, int(10 + processing_time / max_x * 20)))
        
        fig.add_trace(go.Bar(
            x=[processing_time],
            y=[f"Job {job_id}"],
            orientation='h',
            base=[start_time],
            marker=dict(
                color=color, 
                opacity=0.9, 
                line=dict(color=border_color, width=border_width)
            ),
            name=f"Job {job_id}",
            text=f"J{job_id}" if processing_time > max_x/30 else "",  # Dynamic text threshold
            textposition="inside",
            textfont=dict(color='rgba(0,0,0,0.9)', size=text_size),
            showlegend=True,
            hoverinfo='text',
            hovertext=(f"<b>Job {job_id}</b><br>"
                      f"Release: {release_time}<br>"
                      f"Start: {start_time}<br>"
                      f"Processing: {processing_time}<br>"
                      f"Completion: {completion_time}<br>"
                      f"Due date: {due_date}<br>"
                      f"Lateness: <b>{lateness}</b>" +
                      (f"<br><span style='color:red;font-weight:bold'>LATE</span>" if is_late else "<br><span style='color:green;font-weight:bold'>On time</span>"))
        ))
        
        # Add due date marker with improved hover info
        fig.add_trace(go.Scatter(
            x=[due_date],
            y=[f"Job {job_id}"],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='#00BB00'),
            name='Due Date',
            showlegend=(i == 0),  # Only show in legend once
            hoverinfo='text',
            hovertext=f"<b>Job {job_id} - Due Date</b><br>Time: {due_date}"
        ))
        
        # Add release time marker with improved hover info
        fig.add_trace(go.Scatter(
            x=[release_time],
            y=[f"Job {job_id}"],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='#FF5555'),
            name='Release Time',
            showlegend=(i == 0),
            hoverinfo='text',
            hovertext=f"<b>Job {job_id} - Release Time</b><br>Time: {release_time}"
        ))
        
        # Add lateness indicator for late jobs - more visible
        if lateness > 0:
            fig.add_trace(go.Scatter(
                x=[completion_time, due_date],
                y=[f"Job {job_id}", f"Job {job_id}"],
                mode='lines',
                line=dict(color='#FF5555', width=2.5, dash='dot'),
                name='Lateness',
                showlegend=(i == 0),
                hoverinfo='text',
                hovertext=f"<b>Job {job_id} - Lateness</b><br>Amount: {lateness}"
            ))
    
    # Calculate additional statistics
    avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
    utilization = total_processing_time / max_completion if max_completion > 0 else 0
    
    # Update layout for Gantt chart with better spacing
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family="Arial, sans-serif", color="#FFFFFF")
        },
        xaxis=dict(
            title=dict(text='Time', font=dict(size=14, color="#FFFFFF")),
            showgrid=False,  # We'll add custom grid lines
            zeroline=True,
            zerolinecolor='rgba(255, 255, 255, 0.3)',
            zerolinewidth=1,
            showline=True,
            linecolor='rgba(255, 255, 255, 0.5)',
            linewidth=1,
            showticklabels=True,
            tickfont=dict(color="#FFFFFF"),
            range=[0, max_x],
            dtick=max(1, int(max_x/20))  # Dynamic tick spacing
        ),
        yaxis=dict(
            title=dict(text='Jobs', font=dict(size=14, color="#FFFFFF")),
            showgrid=True,
            gridcolor='rgba(100, 100, 100, 0.4)',
            zeroline=False,
            showline=True,
            linecolor='rgba(255, 255, 255, 0.5)',
            automargin=True,
            tickfont=dict(color="#FFFFFF"),
            categoryorder='array',
            categoryarray=y_labels  # Set fixed order for y-axis
        ),
        barmode='stack',
        height=max(350, min(800, 70 * len(df) + 120)),  # Dynamic height with margins for stats
        margin=dict(l=20, r=20, t=120, b=30),  # Increased top margin for stats
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(30, 30, 30, 0.9)',
            bordercolor='rgba(255, 255, 255, 0.3)',
            borderwidth=1,
            font=dict(color="#FFFFFF")
        ),
        plot_bgcolor='rgba(0, 0, 0, 1)',  # Black background
        paper_bgcolor='rgba(10, 10, 10, 1)',  # Slightly off-black for contrast
        font=dict(family="Arial, sans-serif", color="#FFFFFF")
    )
    
    # Add enhanced statistics annotation with better styling
    stats_text = []
    
    # Add lateness stats with better formatting
    if late_jobs:
        late_count = len(late_jobs)
        total_lateness = sum(l for _, l in late_jobs)
        avg_lateness = total_lateness / late_count
        max_lateness = max(l for _, l in late_jobs)
        stats_text.append(f"<b style='color:#FF5555'>Lateness:</b> {late_count}/{len(df)} jobs | Avg: {avg_lateness:.2f} | Max: {max_lateness}")
    else:
        stats_text.append("<b style='color:#00BB00'>Lateness:</b> No late jobs")
    
    # Add waiting and utilization stats
    stats_text.append(f"<b>Waiting:</b> Avg: {avg_waiting_time:.2f} | <b>Resource Utilization:</b> {utilization:.2%}")
    
    # Add completion time
    stats_text.append(f"<b>Makespan:</b> {max_completion}")
    
    # Calculate the length of stats content to adjust positioning - use more compact sizing
    stats_text_length = max([len(text) for text in stats_text])
    
    # Update layout with smaller bottom margin for compact stats panel
    fig.update_layout(
        # Reduced bottom margin for more compact stats panel
        margin=dict(l=20, r=20, t=120, b=60)
    )

    # Add stats panel at the bottom of the chart - more compact design
    fig.add_annotation(
        xref="paper", yref="paper",
        # Position at the bottom right corner
        x=0.99, y=0.02,
        xanchor="right", yanchor="bottom",
        text="<br>".join(stats_text),
        showarrow=False,
        font=dict(
            size=10,  # Smaller font size for compact display
            color="#FFFFFF"
        ),
        align="center",
        bgcolor="rgba(30, 30, 30, 0.95)",
        bordercolor="rgba(255, 255, 255, 0.3)",
        borderwidth=1,
        borderpad=5,  # Reduced padding
        width=min(600, max(300, stats_text_length * 6)),  # More compact width calculation
        captureevents=True,  # Makes the annotation interactive
        visible=False,  # Initially hidden
        name="Stats Panel"
    )
    
    # Add vertical grid lines at major time points with better styling
    for tick in range(0, int(max_x) + 1, max(1, int(max_x/10))):
        fig.add_shape(
            type="line",
            x0=tick, y0=0,
            x1=tick, y1=len(df),
            line=dict(color="rgba(100, 100, 100, 0.4)", width=1, dash="dot"),
            layer="below"
        )
    
    # Add today line if applicable - more visible
    if timeline_type == "simple":
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=0, y1=len(df),
            line=dict(color="rgba(255,255,255,0.7)", width=1.5, dash="dash"),
            name="Current time"
        )
    
    # Make the visualization interactive with enhanced buttons - better styling and positioned on the right
    fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="down",
            buttons=[
                dict(
                    label="<b>All Jobs</b>",
                    method="update",
                    args=[{"visible": [True] * len(fig.data)}]
                ),
                dict(
                    label="<b>Processing Only</b>",
                    method="update",
                    args=[{"visible": [
                        (trace.name is not None and 'Job ' in str(trace.name) and 'Waiting' not in str(trace.name))
                        or (trace.name in ['Release Time', 'Due Date', 'Lateness']) 
                        for trace in fig.data]}]
                ),
                dict(
                    label="<b>Late Jobs Only</b>",
                    method="update",
                    args=[{"visible": [
                        (trace.name is not None and 
                        (any(f"Job {job_id}" in str(trace.name) for job_id, _ in late_jobs) 
                        or trace.name in ['Release Time', 'Due Date', 'Lateness']))
                        for trace in fig.data]}]
                ),
                dict(
                    label="<b>Show Stats</b>",
                    method="relayout",
                    args=["annotations[0].visible", True]
                ),
                dict(
                    label="<b>Hide Stats</b>",
                    method="relayout",
                    args=["annotations[0].visible", False]
                )
            ],
            pad={"r": 15, "t": 10, "b": 10, "l": 15},
            showactive=True,
            x=1.15,  # Move to the right side, outside the main plot
            xanchor="left",
            y=0.95,  # Adjust for better spacing
            yanchor="top",
            bgcolor="rgba(50, 50, 50, 0.9)",
            bordercolor="rgba(255, 255, 255, 0.3)",
            borderwidth=1,
            font=dict(size=12, color="#FF0000")
            ),
        ]
    )

    
    return fig


def visualize_srpt_detail_plotly(timeline, jobs_df):
    """
    Create an interactive Gantt chart for SRPT using Plotly
    
    Parameters:
    timeline: List of dictionaries with job segments {job_id, start, end}
    jobs_df: Dataframe with job data
    
    Returns:
    fig: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Define colors
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Get job IDs in the timeline
    job_ids = sorted(list(set([segment['job_id'] for segment in timeline])))
    
    # Dictionary to track preemptions
    preemptions = {}
    for job_id in job_ids:
        segments = [seg for seg in timeline if seg['job_id'] == job_id]
        preemption_count = len(segments) - 1
        if preemption_count > 0:
            preemptions[job_id] = preemption_count
    
    # Get maximum time for axis range
    max_time = max([segment['end'] for segment in timeline])
    
    # For each job, create a row
    for job_id in job_ids:
        # Get job segments
        job_segments = [seg for seg in timeline if seg['job_id'] == job_id]
        
        # Get job details
        job_row = jobs_df[jobs_df['Job_ID'] == job_id].iloc[0]
        release_time = job_row['Release_Date']
        due_date = job_row['Due_Date']
        color = colors[(job_id-1) % len(colors)]
        
        # Add segments for this job
        for i, segment in enumerate(job_segments):
            start = segment['start']
            end = segment['end']
            duration = end - start
            is_preempted = i < len(job_segments) - 1
            
            # Add processing segment
            fig.add_trace(go.Bar(
                x=[duration],
                y=[f"Job {job_id}"],
                orientation='h',
                base=[start],
                marker=dict(
                        color=color,  
                        opacity=0.9,
                        pattern_shape="/" if is_preempted else "",  # Add stripes for preempted jobs
                        line=dict(
                            color='black',  
                            width=1
                        )
                    )
                    ,
                name=f"Job {job_id}",
                text=f"J{job_id}",
                textposition="inside",
                showlegend=False,
                hoverinfo='text',
                hovertext=(f"Job {job_id}<br>"
                          f"Start: {start}<br>"
                          f"End: {end}<br>"
                          f"Duration: {duration}<br>"
                          f"{'Preempted' if is_preempted else 'Completed'}")
            ))
        
        # Add release time marker
        fig.add_trace(go.Scatter(
            x=[release_time],
            y=[f"Job {job_id}"],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='red'),
            name='Release Time',
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Release time: {release_time}"
        ))
        
        # Add due date marker
        fig.add_trace(go.Scatter(
            x=[due_date],
            y=[f"Job {job_id}"],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='green'),
            name='Due Date',
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Due date: {due_date}"
        ))

    # Update layout for Gantt chart
    fig.update_layout(
        title="SRPT (Shortest Remaining Processing Time) Schedule",
        xaxis=dict(
            title='Time',
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            range=[0, max_time * 1.1]
        ),
        yaxis=dict(
            title='Jobs',
            showgrid=True,
            zeroline=True,
            showline=True,
        ),
        barmode='stack',
        height=50 * len(job_ids) + 100,  # Dynamic height based on number of jobs
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    # Add legend for markers
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='triangle-up', size=10, color='red'),
        name='Release Time', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='green'),
        name='Due Date', showlegend=True
    ))
    
    # Add annotations for preemptions
    if preemptions:
        preemption_text = "Preemptions: " + ", ".join([f"Job {j}: {c}" for j, c in preemptions.items()])
        fig.add_annotation(
            x=0, y=1.06, xref="paper", yref="paper",
            text=preemption_text, showarrow=False,
            font=dict(size=12)
        )
    
    return fig

def visualize_srpt_plotly(timeline, df):
    """Create an interactive Plotly visualization of the SRPT schedule"""
    # Define colors for jobs - more distinct colors
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'gray']
    
    fig = go.Figure()
    
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
        
        # Add rectangle for this job segment
        fig.add_shape(
            type="rect",
            x0=start,
            y0=0,
            x1=end,
            y1=1,
            line=dict(color="black", width=1),
            fillcolor=color,
        )
        
        # Add job ID text inside the rectangle
        fig.add_annotation(
            x=(start + end)/2,
            y=0.5,
            text=f"J{job_id}",
            showarrow=False,
            font=dict(color="white", size=12, family="Arial, bold"),
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
                fig.add_shape(
                    type="line",
                    x0=preemption_time,
                    y0=1,
                    x1=preemption_time,
                    y1=1.5,
                    line=dict(color="black", width=2),
                )
                
                # Draw flag
                fig.add_shape(
                    type="rect",
                    x0=preemption_time,
                    y0=1.5,
                    x1=preemption_time + 1.5,
                    y1=2,
                    line=dict(color="black", width=1),
                    fillcolor="darkgreen",
                )
                
                # Add job ID to flag
                fig.add_annotation(
                    x=preemption_time + 0.75,
                    y=1.75,
                    text=f"J{job_id}",
                    showarrow=False,
                    font=dict(color="white", size=10, family="Arial, bold"),
                )
    
    # Add release time markers (red dashed lines)
    for t in sorted(set(df['Release_Date'])):
        fig.add_shape(
            type="line",
            x0=t,
            y0=-0.2,
            x1=t,
            y1=1,
            line=dict(color="red", width=1, dash="dash"),
        )
    
    # Configure layout
    fig.update_layout(
        title="SRPT (Shortest Remaining Processing Time) Schedule",
        xaxis=dict(
            title="Time",
            range=[-1, max_time + 3],
            tickmode="array",
            tickvals=list(range(0, int(max_time + 4), 2)),
            gridcolor="lightgrey",
            showgrid=True,
        ),
        yaxis=dict(
            range=[-0.2, 2.5],
            showticklabels=False,
            showgrid=False,
        ),
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    return fig

def visualize_branch_bound_plotly(G):
    """
    Create an interactive visualization of Branch and Bound tree using Plotly
    
    Parameters:
    G: NetworkX graph with node attributes
    
    Returns:
    fig: Plotly figure object
    """
    import networkx as nx
    
    # Get node levels
    levels = nx.get_node_attributes(G, 'level')
    max_level = max(levels.values())
    
    # Group nodes by level
    nodes_by_level = {}
    for node, level in levels.items():
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)
    
    # Calculate positions
    pos = {}
    h_space = 1.0  # Horizontal spacing factor
    v_space = 1.0  # Vertical spacing factor
    
    for level in range(max_level + 1):
        if level in nodes_by_level:
            nodes = sorted(nodes_by_level[level])
            width = len(nodes)
            
            for i, node in enumerate(nodes):
                x = (i - width / 2 + 0.5) * h_space
                y = -level * v_space
                pos[node] = (x, y)
    
    # Get node attributes
    labels = nx.get_node_attributes(G, 'label')
    is_best = nx.get_node_attributes(G, 'is_best')
    bound = nx.get_node_attributes(G, 'lmax')
    pruned = nx.get_node_attributes(G, 'pruned')
    
    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
        )
    
    # Create node traces (separate by type)
    node_best = go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        marker=dict(
            color='lightgreen',
            size=30,
            line=dict(width=1, color='black')
        ),
        text=[],
        textposition="middle center",
        textfont=dict(color='darkgreen', size=12),
        hoverinfo="text",
        hovertext=[],
        name="Optimal Path"
    )
    
    node_active = go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        marker=dict(
            color='skyblue',
            size=30,
            line=dict(width=1, color='black')
        ),
        text=[],
        textposition="middle center",
        textfont=dict(color='navy', size=12),
        hoverinfo="text",
        hovertext=[],
        name="Active Path",
    )
    
    node_terminal = go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        marker=dict(
            color='lightgray',
            size=30,
            line=dict(width=1, color='black')
        ),
        text=[],
        textposition="middle center",
        textfont=dict(color='black', size=12),
        hoverinfo="text",
        hovertext=[],
        name="Terminal/Pruned"
    )
    
    # Fill node traces
    for node in G.nodes():
        x, y = pos[node]
        
        # Custom label with job ID and bound
        label = labels.get(node, "")
        node_bound = bound.get(node, 0)
        display_text = f"{label}\n{node_bound}"
        
        # Hover text with more details
        hover_text = f"Node {node}<br>Job: {label}<br>Bound: {node_bound}"
        
        # Optimal path
        if is_best.get(node, False):
            node_best.x = node_best.x + (x,)
            node_best.y = node_best.y + (y,)
            node_best.text = node_best.text + (display_text,)
            node_best.hovertext = node_best.hovertext + (hover_text,)
        # Active path (has children and not pruned)
        else:
            successors = list(G.successors(node))
            if successors:
                node_active.x = node_active.x + (x,)
                node_active.y = node_active.y + (y,)
                node_active.text = node_active.text + (display_text,)
                node_active.hovertext = node_active.hovertext + (hover_text,)
            else:
                node_terminal.x = node_terminal.x + (x,)
                node_terminal.y = node_terminal.y + (y,)
                node_terminal.text = node_terminal.text + (display_text,)
                node_terminal.hovertext = node_terminal.hovertext + (hover_text,)
    
    # Create figure and add traces
    fig = go.Figure(data=edge_trace + [node_terminal, node_active, node_best])
    
    # Update layout
    fig.update_layout(
        title="Branch and Bound Tree Visualization",
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=800,
        width=800
    )
    
    return fig

def visualize_moore(jobs_df, completed_indices, late_indices):
    """
    Create a Plotly visualization for Moore's rule schedule
    
    Args:
        jobs_df: Original jobs dataframe
        completed_indices: Indices of jobs completed on time
        late_indices: Indices of jobs that are late
        
    Returns:
        A plotly figure object
    """
    # Define a colormap
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'grey', 'cyan', 'magenta']
    
    # Create figure
    fig = go.Figure()
    
    # Process completed jobs
    completed_jobs = jobs_df.loc[completed_indices].sort_values('Due_Date')
    current_time = 0
    
    for idx, job in completed_jobs.iterrows():
        job_id = int(job['Job_ID'])
        duration = job['Processing_Time']
        
        # Choose color based on job index
        color = colors[job_id % len(colors)]
        
        # Add job bar
        fig.add_trace(go.Bar(
            y=["Completed Jobs"],
            x=[duration],
            base=current_time,
            marker_color=color,
            orientation='h',
            name=f"Job {job_id}",
            text=f"Job {job_id}",
            textposition="inside",
            hoverinfo="text",
            hovertext=f"Job {job_id} from {current_time} to {current_time + duration}",
            showlegend=False
        ))
        
        # Move time forward
        current_time += duration
    
    # Add separator
    if late_indices:
        fig.add_shape(
            type="line",
            x0=current_time,
            y0=0,
            x1=current_time,
            y1=1,
            line=dict(color="navy", width=2),
            layer="below"
        )
    
    # Process late jobs
    late_jobs = jobs_df.loc[late_indices]
    
    for idx, job in late_jobs.iterrows():
        job_id = int(job['Job_ID'])
        duration = job['Processing_Time']
        
        # Choose color based on job index
        color = colors[job_id % len(colors)]
        
        # Add job bar
        fig.add_trace(go.Bar(
            y=["Late Jobs"],
            x=[duration],
            base=current_time,
            marker_color=color,
            orientation='h',
            name=f"Job {job_id}",
            text=f"Job {job_id}",
            textposition="inside",
            hoverinfo="text",
            hovertext=f"Job {job_id} (late) from {current_time} to {current_time + duration}",
            showlegend=False
        ))
        
        # Move time forward
        current_time += duration
    
    # Configure layout
    fig.update_layout(
        title="Moore's Rule Schedule: Completed Jobs | Late Jobs",
        xaxis_title="Time",
        barmode='overlay',
        xaxis=dict(
            range=[0, current_time * 1.1],
            tickmode='linear',
            tick0=0,
            dtick=2
        ),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    return fig