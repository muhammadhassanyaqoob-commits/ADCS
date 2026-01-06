import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="CubeSat ADCS Optimization Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        border-bottom: 2px solid #60A5FA;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(45deg, #3B82F6, #8B5CF6);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚀 CubeSat ADCS Optimization Methodology Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/287/287226.png", width=100)
    st.title("Configuration")
    
    st.markdown("### Optimization Parameters")
    
    # Problem setup
    n_orbits = st.slider("Number of Orbits", 1, 50, 10)
    events_per_orbit = st.slider("Max Events per Orbit", 1, 20, 5)
    
    st.markdown("### Objectives Weighting")
    energy_weight = st.slider("Energy Consumption Weight (w₁)", 0.0, 2.0, 1.0, 0.1)
    imaging_weight = st.slider("Imaging Events Weight (w₂)", 0.0, 2.0, 1.0, 0.1)
    
    st.markdown("### Safety Margins")
    soc_min = st.slider("Minimum SOC (%)", 50, 90, 76)
    slew_rate_max = st.slider("Max Slew Rate (deg/s)", 0.1, 5.0, 2.0, 0.1)
    
    st.markdown("### Configuration Selection")
    payload_direction = st.selectbox("Payload Facing", ["-Z", "-X"])
    panel_attachment = st.selectbox("Panel Attachment", ["Y-face", "Z-face"])
    panel_angle = st.slider("Panel Cant Angle (deg)", 0, 45, 15)
    
    if st.button("🚀 Run Optimization", use_container_width=True):
        st.session_state.optimization_run = True

# Main content area
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "🔧 Methodology", 
    "📈 Pareto Analysis", 
    "⚙️ Constraints",
    "📅 Schedule",
    "📊 Results"
])

with tab1:
    st.markdown('<h2 class="section-header">System Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Problem Type</h3>
            <p style="font-size: 1.5rem;">Multi-Objective MINLP</p>
            <p>Mixed-Integer Nonlinear Programming</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Decision Variables</h3>
            <p style="font-size: 1.5rem;">4 Binary + 2 Continuous</p>
            <p>Configuration & Schedule</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Constraints</h3>
            <p style="font-size: 1.5rem;">9 Categories</p>
            <p>Safety, Power, Dynamics</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Methodology Flow")
    
    # Create flowchart
    fig = go.Figure()
    
    # Node positions
    node_positions = {
        'Start': (0, 5),
        'Init': (2, 5),
        'Config': (4, 5),
        'Config1': (6, 6),
        'Config2': (6, 5),
        'Config3': (6, 4),
        'Config4': (6, 3),
        'MINLP': (8, 5),
        'Constraints': (10, 5),
        'Feasible': (12, 6),
        'Infeasible': (12, 4),
        'Pareto': (14, 5),
        'Robustness': (16, 5),
        'Solution': (18, 5)
    }
    
    # Add edges
    edges = [
        ('Start', 'Init'), ('Init', 'Config'),
        ('Config', 'Config1'), ('Config', 'Config2'), ('Config', 'Config3'), ('Config', 'Config4'),
        ('Config1', 'MINLP'), ('Config2', 'MINLP'), ('Config3', 'MINLP'), ('Config4', 'MINLP'),
        ('MINLP', 'Constraints'), ('Constraints', 'Feasible'), ('Constraints', 'Infeasible'),
        ('Feasible', 'Pareto'), ('Pareto', 'Robustness'), ('Robustness', 'Solution')
    ]
    
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = node_positions[edge[0]]
        x1, y1 = node_positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    for node, pos in node_positions.items():
        node_x.append(pos[0])
        node_y.append(pos[1])
        node_text.append(node)
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=40,
            color=['#10B981'] + ['#3B82F6'] * 3 + ['#8B5CF6'] * 10 + ['#EF4444'] * 2,
            line=dict(color='white', width=2)
        ),
        textfont=dict(size=10, color='white')
    ))
    
    fig.update_layout(
        title="Optimization Flowchart",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # System components
    st.markdown("### System Components")
    
    components = pd.DataFrame({
        'Component': ['Payload', 'Solar Array', 'Battery', 'Magnetorquer', 'Camera', 'Comm System'],
        'Status': ['Active', 'Active', 'Charging', 'Idle', 'Standby', 'Idle'],
        'Power (W)': [15.0, 12.5, 0.0, 0.0, 5.0, 8.0],
        'Temperature (°C)': [22.5, 25.0, 18.0, 20.0, 21.0, 23.0]
    })
    
    st.dataframe(components, use_container_width=True)

with tab2:
    st.markdown('<h2 class="section-header">Methodology Details</h2>', unsafe_allow_html=True)
    
    # Methodology steps
    steps = [
        {
            "step": 1,
            "title": "Problem Formulation",
            "description": "Define MINLP with geometric configuration and operational scheduling variables",
            "details": "Mixed-integer variables for configuration selection, continuous variables for angles, binary variables for activity scheduling"
        },
        {
            "step": 2,
            "title": "Configuration Enumeration",
            "description": "Systematically explore all geometric configurations (4 combinations)",
            "details": "Payload facing: [-Z, -X] × Panel attachment: [Y-face, Z-face]"
        },
        {
            "step": 3,
            "title": "Reduced MINLP Solution",
            "description": "Solve branch-and-bound with convex relaxations for each configuration",
            "details": "Fix configuration variables, optimize panel angle and schedule"
        },
        {
            "step": 4,
            "title": "Constraint Verification",
            "description": "Check hard constraints (SOC ≥ 76%, activity exclusivity, dynamics)",
            "details": "Immediate pruning of infeasible solutions"
        },
        {
            "step": 5,
            "title": "Pareto Front Generation",
            "description": "ε-constraint method for multi-objective optimization",
            "details": "Trade-off between energy consumption and imaging events"
        },
        {
            "step": 6,
            "title": "Robustness Analysis",
            "description": "Multi-scenario evaluation under uncertainty",
            "details": "Magnetic field, solar degradation, efficiency variations"
        },
        {
            "step": 7,
            "title": "Solution Selection",
            "description": "Weight calibration from Pareto front slopes",
            "details": "Select optimal configuration and schedule"
        }
    ]
    
    for step in steps:
        with st.expander(f"Step {step['step']}: {step['title']}"):
            st.write(f"**{step['description']}**")
            st.write(step['details'])
            
            # Add visualization for each step
            if step['step'] == 1:
                fig = go.Figure(data=[
                    go.Bar(name='Continuous', x=['Variables'], y=[2], marker_color='#3B82F6'),
                    go.Bar(name='Binary', x=['Variables'], y=[4], marker_color='#8B5CF6'),
                    go.Bar(name='Constraints', x=['Constraints'], y=[9], marker_color='#EF4444')
                ])
                fig.update_layout(barmode='stack', title="Problem Structure", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            elif step['step'] == 2:
                fig = go.Figure(data=[
                    go.Table(
                        header=dict(values=['Payload Facing', 'Panel Attachment', 'Combination ID']),
                        cells=dict(values=[
                            ['-Z', '-Z', '-X', '-X'],
                            ['Y-face', 'Z-face', 'Y-face', 'Z-face'],
                            ['C1', 'C2', 'C3', 'C4']
                        ])
                    )
                ])
                fig.update_layout(title="Configuration Space", height=200)
                st.plotly_chart(fig, use_container_width=True)
            
            elif step['step'] == 3:
                # Create a simple optimization convergence plot
                x = np.linspace(0, 100, 50)
                y = 500 * np.exp(-x/30) + 50 * np.sin(x/5) + 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Objective Value',
                                        line=dict(color='#3B82F6', width=2)))
                fig.update_layout(title="Branch-and-Bound Convergence", 
                                xaxis_title="Iterations", yaxis_title="Energy Consumption (W-hr)",
                                height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            elif step['step'] == 4:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SOC Safety Margin", "24%", "+4%")
                with col2:
                    st.metric("Constraint Violations", "0", "-")

with tab3:
    st.markdown('<h2 class="section-header">Pareto Analysis</h2>', unsafe_allow_html=True)
    
    # Generate synthetic Pareto data
    np.random.seed(42)
    n_points = 50
    
    # Energy vs Imaging events trade-off
    energy_base = np.linspace(200, 800, n_points)
    imaging_events = 30 - 0.02 * energy_base + np.random.normal(0, 2, n_points)
    imaging_events = np.clip(imaging_events, 5, 25)
    
    # Different configurations
    configs = ['C1 (-Z, Y-face)', 'C2 (-Z, Z-face)', 'C3 (-X, Y-face)', 'C4 (-X, Z-face)']
    config_colors = ['#3B82F6', '#10B981', '#8B5CF6', '#EF4444']
    
    fig = go.Figure()
    
    for i, config in enumerate(configs):
        offset = i * 0.5
        config_energy = energy_base + np.random.normal(0, 20, n_points)
        config_events = imaging_events + np.random.normal(offset, 1, n_points)
        
        fig.add_trace(go.Scatter(
            x=config_energy, y=config_events,
            mode='markers',
            name=config,
            marker=dict(size=10, color=config_colors[i], opacity=0.7),
            hovertemplate=f'<b>{config}</b><br>Energy: %{{x:.1f}} W-hr<br>Events: %{{y:.1f}}<extra></extra>'
        ))
    
    # Highlight Pareto front
    pareto_energy = np.array([250, 350, 450, 550, 650, 750])
    pareto_events = np.array([24, 22, 19, 16, 13, 10])
    
    fig.add_trace(go.Scatter(
        x=pareto_energy, y=pareto_events,
        mode='lines',
        name='Pareto Front',
        line=dict(color='#FF6B6B', width=3, dash='dash'),
        hovertemplate='<b>Pareto Optimal</b><br>Energy: %{x:.1f} W-hr<br>Events: %{y:.1f}<extra></extra>'
    ))
    
    # Interactive point
    if 'selected_point' not in st.session_state:
        st.session_state.selected_point = 3
    
    selected_idx = st.slider("Select Pareto Point", 0, len(pareto_energy)-1, 
                            st.session_state.selected_point, key="pareto_slider")
    
    fig.add_trace(go.Scatter(
        x=[pareto_energy[selected_idx]],
        y=[pareto_events[selected_idx]],
        mode='markers',
        name='Selected Solution',
        marker=dict(size=20, color='#FFD93D', symbol='star'),
        hovertemplate=f'<b>Selected Solution</b><br>Energy: {pareto_energy[selected_idx]:.1f} W-hr<br>Events: {pareto_events[selected_idx]:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Pareto Front: Energy vs Imaging Events",
        xaxis_title="Total Energy Consumption (W-hr)",
        yaxis_title="Number of Imaging Events",
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Weight calibration
    st.markdown("### Weight Calibration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        slopes = -np.diff(pareto_events) / np.diff(pareto_energy)
        weight_ratios = -1 / slopes
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pareto_energy[:-1], y=weight_ratios,
            mode='lines+markers',
            line=dict(color='#8B5CF6', width=2),
            name='w₁/w₂ Ratio'
        ))
        
        fig2.update_layout(
            title="Weight Ratio Along Pareto Front",
            xaxis_title="Energy (W-hr)",
            yaxis_title="Weight Ratio w₁/w₂",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Current weights
        current_ratio = energy_weight / imaging_weight if imaging_weight > 0 else float('inf')
        
        fig3 = go.Figure()
        fig3.add_trace(go.Indicator(
            mode="gauge+number",
            value=current_ratio,
            title={'text': "Current Weight Ratio"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 5]},
                'bar': {'color': "#3B82F6"},
                'steps': [
                    {'range': [0, 1], 'color': "#10B981"},
                    {'range': [1, 3], 'color': "#FBBF24"},
                    {'range': [3, 5], 'color': "#EF4444"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2.5
                }
            }
        ))
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.markdown('<h2 class="section-header">Constraint Analysis</h2>', unsafe_allow_html=True)
    
    # Constraint categories
    constraint_categories = {
        "Safety Constraints": {
            "SOC ≥ 76%": {"status": "Satisfied", "margin": "24%", "criticality": "High"},
            "Activity Exclusivity": {"status": "Satisfied", "margin": "100%", "criticality": "Medium"},
            "Slew Rate Limits": {"status": "Satisfied", "margin": "40%", "criticality": "High"}
        },
        "Physical Constraints": {
            "Power Limits": {"status": "Satisfied", "margin": "35%", "criticality": "High"},
            "Thermal Limits": {"status": "Satisfied", "margin": "15°C", "criticality": "Medium"},
            "Torque Limits": {"status": "Satisfied", "margin": "50%", "criticality": "High"}
        },
        "Temporal Constraints": {
            "Eclipse Exclusion": {"status": "Satisfied", "margin": "100%", "criticality": "Medium"},
            "Recovery Time": {"status": "Satisfied", "margin": "30min", "criticality": "Medium"},
            "Time Budget": {"status": "Satisfied", "margin": "45min", "criticality": "Low"}
        }
    }
    
    # Display constraints in expandable sections
    for category, constraints in constraint_categories.items():
        with st.expander(f"{category} ({len(constraints)} constraints)"):
            for constraint, details in constraints.items():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                with col1:
                    st.write(f"**{constraint}**")
                with col2:
                    status_color = "🟢" if details["status"] == "Satisfied" else "🔴"
                    st.write(f"{status_color} {details['status']}")
                with col3:
                    st.write(f"Margin: {details['margin']}")
                with col4:
                    crit_color = {
                        "High": "🔴",
                        "Medium": "🟡", 
                        "Low": "🟢"
                    }
                    st.write(f"{crit_color[details['criticality']]} {details['criticality']}")
    
    # Constraint visualization
    st.markdown("### Constraint Margins Visualization")
    
    # Create radar chart for constraint margins
    categories = list(constraint_categories.keys())
    sub_constraints = []
    margins = []
    
    for category in categories:
        for constraint, details in constraint_categories[category].items():
            sub_constraints.append(f"{constraint[:15]}...")
            # Convert margin to numeric value for visualization
            margin_str = details['margin']
            if '%' in margin_str:
                margins.append(float(margin_str.replace('%', '')))
            elif 'min' in margin_str:
                margins.append(float(margin_str.replace('min', '')) * 10)
            elif '°C' in margin_str:
                margins.append(float(margin_str.replace('°C', '')) * 10)
            else:
                margins.append(50)  # Default
    
    # Create polar plot
    fig = go.Figure(data=go.Scatterpolar(
        r=margins,
        theta=sub_constraints,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='rgb(59, 130, 246)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            ),
        ),
        showlegend=False,
        title="Constraint Margins (Higher is Better)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Constraint satisfaction over time
    st.markdown("### Constraint Satisfaction Over Optimization")
    
    # Generate timeline data
    iterations = np.arange(100)
    soc_margin = 20 + 10 * np.sin(iterations/10) + np.random.normal(0, 2, 100)
    soc_margin = np.maximum(soc_margin, 0)
    
    power_margin = 30 + 15 * np.sin(iterations/15 + 2) + np.random.normal(0, 3, 100)
    power_margin = np.maximum(power_margin, 0)
    
    time_margin = 40 + 20 * np.sin(iterations/20 + 4) + np.random.normal(0, 4, 100)
    time_margin = np.maximum(time_margin, 0)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=iterations, y=soc_margin, mode='lines', 
                            name='SOC Margin', line=dict(color='#10B981')))
    fig2.add_trace(go.Scatter(x=iterations, y=power_margin, mode='lines', 
                            name='Power Margin', line=dict(color='#3B82F6')))
    fig2.add_trace(go.Scatter(x=iterations, y=time_margin, mode='lines', 
                            name='Time Margin', line=dict(color='#8B5CF6')))
    
    fig2.update_layout(
        title="Constraint Margins Evolution During Optimization",
        xaxis_title="Iteration",
        yaxis_title="Margin (%)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab5:
    st.markdown('<h2 class="section-header">Operational Schedule</h2>', unsafe_allow_html=True)
    
    # Generate schedule data
    np.random.seed(42)
    n_orbits_schedule = min(n_orbits, 10)
    
    schedule_data = []
    activity_colors = {
        'imaging': '#3B82F6',
        'slew': '#EF4444',
        'recovery': '#10B981',
        'transmission': '#8B5CF6',
        'reception': '#F59E0B',
        'idle': '#6B7280'
    }
    
    for orbit in range(n_orbits_schedule):
        time_offset = orbit * 90  # 90 minutes per orbit
        
        # Add eclipse period
        schedule_data.append({
            'Activity': 'eclipse',
            'Start': time_offset + 60,
            'End': time_offset + 75,
            'Orbit': orbit + 1,
            'Color': '#000000'
        })
        
        # Add random activities
        activities = ['imaging', 'slew', 'recovery', 'transmission', 'idle']
        for i in range(events_per_orbit):
            activity = np.random.choice(activities, p=[0.3, 0.2, 0.2, 0.2, 0.1])
            start = time_offset + np.random.uniform(0, 60)
            duration = np.random.uniform(1, 15)
            
            schedule_data.append({
                'Activity': activity,
                'Start': start,
                'End': start + duration,
                'Orbit': orbit + 1,
                'Color': activity_colors[activity]
            })
    
    # Create Gantt chart
    df_schedule = pd.DataFrame(schedule_data)
    
    fig = go.Figure()
    
    for activity in df_schedule['Activity'].unique():
        df_activity = df_schedule[df_schedule['Activity'] == activity]
        fig.add_trace(go.Bar(
            x=[f"Orbit {int(row['Orbit'])}" for _, row in df_activity.iterrows()],
            y=df_activity['End'] - df_activity['Start'],
            base=df_activity['Start'],
            name=activity.capitalize(),
            marker_color=df_activity['Color'].iloc[0],
            text=[f"{activity}<br>{row['Start']:.1f}-{row['End']:.1f} min" 
                  for _, row in df_activity.iterrows()],
            textposition='none',
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=f"Operational Schedule ({n_orbits_schedule} Orbits)",
        barmode='stack',
        xaxis_title="Orbit",
        yaxis_title="Time (minutes from mission start)",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # SOC Profile
    st.markdown("### Battery SOC Profile")
    
    # Generate SOC data
    mission_time = np.linspace(0, n_orbits_schedule * 90, 500)
    soc = 100 - 10 * np.sin(mission_time/20) - 5 * np.sin(mission_time/5) + np.random.normal(0, 1, 500)
    soc = np.clip(soc, soc_min, 100)
    
    # Add drops during activities
    for _, row in df_schedule.iterrows():
        mask = (mission_time >= row['Start']) & (mission_time <= row['End'])
        if row['Activity'] == 'imaging':
            soc[mask] -= 3
        elif row['Activity'] in ['slew', 'transmission']:
            soc[mask] -= 1.5
    
    soc = np.maximum(soc, soc_min)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=mission_time, y=soc,
        mode='lines',
        name='SOC',
        line=dict(color='#10B981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    # Add minimum SOC line
    fig2.add_hline(y=soc_min, line_dash="dash", line_color="red", 
                   annotation_text=f"Minimum SOC ({soc_min}%)")
    
    # Add activity markers
    for _, row in df_schedule.iterrows():
        if row['Activity'] != 'eclipse':
            fig2.add_vrect(
                x0=row['Start'], x1=row['End'],
                fillcolor=activity_colors[row['Activity']],
                opacity=0.1,
                layer="below",
                line_width=0,
            )
    
    fig2.update_layout(
        title="State of Charge Profile",
        xaxis_title="Mission Time (minutes)",
        yaxis_title="SOC (%)",
        yaxis_range=[soc_min - 5, 105],
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Schedule metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_imaging = len(df_schedule[df_schedule['Activity'] == 'imaging'])
    total_transmission = len(df_schedule[df_schedule['Activity'] == 'transmission'])
    avg_soc = np.mean(soc)
    min_soc = np.min(soc)
    
    with col1:
        st.metric("Total Imaging Events", total_imaging)
    with col2:
        st.metric("Total Transmission Events", total_transmission)
    with col3:
        st.metric("Average SOC", f"{avg_soc:.1f}%")
    with col4:
        delta_min_soc = min_soc - soc_min
        st.metric("Minimum SOC", f"{min_soc:.1f}%", f"{delta_min_soc:+.1f}%")

with tab6:
    st.markdown('<h2 class="section-header">Optimization Results</h2>', unsafe_allow_html=True)
    
    if 'optimization_run' not in st.session_state:
        st.warning("Please run the optimization from the sidebar to see results.")
    else:
        # Generate results
        np.random.seed(42)
        
        # Configuration performance comparison
        configs = ['C1 (-Z, Y-face)', 'C2 (-Z, Z-face)', 'C3 (-X, Y-face)', 'C4 (-X, Z-face)']
        
        results_data = {
            'Configuration': configs,
            'Energy (W-hr)': np.random.uniform(300, 700, 4),
            'Imaging Events': np.random.randint(15, 25, 4),
            'SOC Margin (%)': np.random.uniform(20, 40, 4),
            'Power Margin (%)': np.random.uniform(25, 45, 4),
            'Feasibility': ['Feasible', 'Feasible', 'Infeasible', 'Feasible']
        }
        
        df_results = pd.DataFrame(results_data)
        
        # Highlight best configurations
        def highlight_row(row):
            if row['Feasibility'] == 'Infeasible':
                return ['background-color: #FEE2E2'] * len(row)
            elif row['Energy (W-hr)'] == df_results['Energy (W-hr)'].min():
                return ['background-color: #D1FAE5'] * len(row)
            elif row['Imaging Events'] == df_results['Imaging Events'].max():
                return ['background-color: #DBEAFE'] * len(row)
            return [''] * len(row)
        
        st.markdown("### Configuration Performance Comparison")
        st.dataframe(df_results.style.apply(highlight_row, axis=1), use_container_width=True)
        
        # 3D visualization of solution space
        st.markdown("### Solution Space Exploration")
        
        # Generate 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=df_results['Energy (W-hr)'],
            y=df_results['Imaging Events'],
            z=df_results['SOC Margin (%)'],
            mode='markers+text',
            marker=dict(
                size=12,
                color=df_results['Power Margin (%)'],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Power Margin (%)")
            ),
            text=df_results['Configuration'],
            textposition="top center"
        )])
        
        fig.update_layout(
            title="Solution Space (3D)",
            scene=dict(
                xaxis_title='Energy (W-hr)',
                yaxis_title='Imaging Events',
                zaxis_title='SOC Margin (%)'
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.markdown("### Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Energy efficiency
            fig_energy = go.Figure(go.Indicator(
                mode="gauge+number",
                value=df_results['Energy (W-hr)'].min(),
                title={'text': "Best Energy (W-hr)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [300, 700]},
                    'bar': {'color': "#10B981"},
                    'steps': [
                        {'range': [300, 450], 'color': "#D1FAE5"},
                        {'range': [450, 600], 'color': "#FEF3C7"},
                        {'range': [600, 700], 'color': "#FEE2E2"}
                    ]
                }
            ))
            fig_energy.update_layout(height=250)
            st.plotly_chart(fig_energy, use_container_width=True)
        
        with col2:
            # Imaging performance
            fig_imaging = go.Figure(go.Indicator(
                mode="gauge+number",
                value=df_results['Imaging Events'].max(),
                title={'text': "Max Imaging Events"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [15, 25]},
                    'bar': {'color': "#3B82F6"},
                    'steps': [
                        {'range': [15, 18], 'color': "#FEE2E2"},
                        {'range': [18, 22], 'color': "#FEF3C7"},
                        {'range': [22, 25], 'color': "#D1FAE5"}
                    ]
                }
            ))
            fig_imaging.update_layout(height=250)
            st.plotly_chart(fig_imaging, use_container_width=True)
        
        with col3:
            # Safety margin
            fig_safety = go.Figure(go.Indicator(
                mode="gauge+number",
                value=df_results['SOC Margin (%)'].max(),
                title={'text': "Best SOC Margin (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [20, 40]},
                    'bar': {'color': "#8B5CF6"},
                    'steps': [
                        {'range': [20, 25], 'color': "#FEE2E2"},
                        {'range': [25, 35], 'color': "#FEF3C7"},
                        {'range': [35, 40], 'color': "#D1FAE5"}
                    ]
                }
            ))
            fig_safety.update_layout(height=250)
            st.plotly_chart(fig_safety, use_container_width=True)
        
        # Download results
        st.markdown("### Export Results")
        
        # Convert results to JSON
        results_json = df_results.to_json(orient='records')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📥 Download Results (JSON)",
                data=results_json,
                file_name=f"adcs_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"adcs_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Generate and download report
            report = f"""
            ADCS Optimization Results Report
            ================================
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Selected Configuration:
            - Payload Facing: {payload_direction}
            - Panel Attachment: {panel_attachment}
            - Panel Angle: {panel_angle}°
            
            Optimization Parameters:
            - Orbits: {n_orbits}
            - Events per Orbit: {events_per_orbit}
            - SOC Minimum: {soc_min}%
            - Weight Ratio (w₁/w₂): {energy_weight/imaging_weight:.2f}
            
            Best Configuration Performance:
            - Configuration: {df_results.loc[df_results['Energy (W-hr)'].idxmin(), 'Configuration']}
            - Energy Consumption: {df_results['Energy (W-hr)'].min():.1f} W-hr
            - Imaging Events: {df_results['Imaging Events'].max()}
            - SOC Margin: {df_results['SOC Margin (%)'].max():.1f}%
            
            Feasibility Summary:
            - Total Configurations: {len(df_results)}
            - Feasible Configurations: {len(df_results[df_results['Feasibility'] == 'Feasible'])}
            - Infeasible Configurations: {len(df_results[df_results['Feasibility'] == 'Infeasible'])}
            """
            
            st.download_button(
                label="📥 Download Report (TXT)",
                data=report,
                file_name=f"adcs_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280;">
    <p>CubeSat ADCS Optimization Methodology Dashboard • Powered by Streamlit</p>
    <p>Mixed-Integer Nonlinear Programming for Geometric Configuration and Operational Scheduling</p>
</div>
""", unsafe_allow_html=True)

Add a commit message at the bottom
