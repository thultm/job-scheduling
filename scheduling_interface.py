import streamlit as st
import pandas as pd
import time
from algorithms import *
from visualization import *
from old_visualization import *

# Set page title
st.title("Job Scheduling Algorithms")

# Sidebar for algorithm selection
st.sidebar.header("Select Algorithm")
algorithm = st.sidebar.selectbox(
    "Choose a scheduling algorithm",
    [
        "First Come First Serve (FCFS)", 
        "Shortest Processing Time (SPT)", 
        "Longest Processing Time (LPT)", 
        "Earliest Due Date (EDD)", 
        "Earliest Release Date (ERD)",
        "Weighted Shortest Processing Time (WSPT)",
        "Random Sequencing (RAND)",
        "Moore's Rule (Minimize Late Jobs)", 
        "Shortest Remaining Processing Time (SRPT)",
        "Branch and Bound (Minimize Maximum Lateness)"
    ]
)

# Data input section
st.header("Input Job Data")
upload_option = st.radio("Choose input method:", ["Upload CSV/Excel", "Manual Input"])

df = None

if upload_option == "Upload CSV/Excel":
    uploaded_file = st.file_uploader("Upload job data file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Without 
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.write("Uploaded data:")
        st.dataframe(df, hide_index=True)
else:
    # Manual job input
    st.subheader("Enter job details")
    num_jobs = st.number_input("Number of jobs", min_value=1, max_value=20, value=3)

    job_data = {
        "Job_ID": [],
        "Processing_Time": [],
        "Due_Date": [],
        "Weight": [],
        "Release_Date": []
    }

    for i in range(num_jobs):
        st.markdown(f"**Job {i+1}**")
        cols = st.columns(5)

        job_data["Job_ID"].append(cols[0].number_input(f"Job ID {i+1}", value=i+1, key=f"id_{i}"))
        job_data["Processing_Time"].append(cols[1].number_input(f"Processing Time {i+1}", value=5, min_value=1, key=f"pt_{i}"))
        job_data["Due_Date"].append(cols[2].number_input(f"Due Date {i+1}", value=10, min_value=1, key=f"dd_{i}"))
        job_data["Weight"].append(cols[3].number_input(f"Weight {i+1}", value=1, min_value=1, key=f"w_{i}"))
        job_data["Release_Date"].append(cols[4].number_input(f"Release Date {i+1}", value=0, min_value=0, key=f"rd_{i}"))

    df = pd.DataFrame(job_data)
    st.write("Entered data:")
    st.dataframe(df, hide_index=True)

# Run the algorithm
if df is not None and st.button("Run Algorithm"):
    # Check if Release_Date column exists, add if not
    if 'Release_Date' not in df.columns:
        df['Release_Date'] = 0  # Default to zero if not provided
    
    # Check if Weight column exists, add if not
    if 'Weight' not in df.columns:
        df['Weight'] = 1  # Default to one if not provided

    if algorithm == "First Come First Serve (FCFS)":
        FCFS(df)

    # 2. SPT Algorithm
    elif algorithm == "Shortest Processing Time (SPT)":
        SPT(df)

    # 3. LPT Algorithm
    elif algorithm == "Longest Processing Time (LPT)":
        LPT(df)

    # 4. EDD Algorithm
    elif algorithm == "Earliest Due Date (EDD)":
        EDD(df)

    # 5. ERD Algorithm
    elif algorithm == "Earliest Release Date (ERD)":
        ERD(df)

    # 6. WSPT Algorithm
    elif algorithm == "Weighted Shortest Processing Time (WSPT)":
        WSPT(df)

    # 7. Random Sequencing Algorithm
    elif algorithm == "Random Sequencing (RAND)":
        RAND(df) 

    # 8. Moore's Rule Algorithm
    elif algorithm == "Moore's Rule (Minimize Late Jobs)":
        moore_rule(df)

    # 9. SRPT Algorithm
    elif algorithm == "Shortest Remaining Processing Time (SRPT)":
        SRPT(df)

    # 10. Branch and Bound Algorithm
    elif algorithm == "Branch and Bound (Minimize Maximum Lateness)":
        # Create containers for results and visualization
        result_container = st.container()
        figure_container = st.container()
        # Run the Branch and Bound algorithm
        start_time = time.time()
        optimal_sequence, optimal_lmax, tree, all_nodes = branch_and_bound_lmax(df)
        end_time = time.time()

        # Display results
        with result_container:
            st.subheader("Branch and Bound Results")
            st.write(f"Optimal sequence: {optimal_sequence}")
            st.write(f"Minimum maximum lateness: {optimal_lmax}")
            st.write(f"Computation time: {end_time - start_time:.3f} seconds")
            
            # Count node types for statistics
            path_nodes = 0
            terminal_nodes = 0
            best_nodes = 0
            
            for node_id, node in all_nodes.items():
                if node.is_best:
                    best_nodes += 1
                elif node.children:
                    path_nodes += 1
                else:
                    terminal_nodes += 1
            
            st.write(f"Active path nodes: {path_nodes}")
            st.write(f"Terminal nodes: {terminal_nodes}")
            st.write(f"Optimal path nodes: {best_nodes}")

        with figure_container:
            # Visualize the branch and bound tree
            st.subheader("Branch and Bound Tree")
            fig = visualize_branch_bound_plotly(tree)
            st.plotly_chart(fig, use_container_width=True)