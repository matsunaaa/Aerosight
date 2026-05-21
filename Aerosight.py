import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# SIGNAL PROCESSING FUNCTIONS
# =============================================================================

def apply_butterworth_filter(data, sample_rate, cutoff_freq, order=2):
    """
    Apply a Butterworth low-pass filter with zero phase delay.
    """
    nyquist = sample_rate / 2
    
    if cutoff_freq >= nyquist:
        cutoff_freq = nyquist * 0.9
    
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    
    pad_len = min(len(data) // 4, 50)
    if pad_len < 10 or len(data) < 20:
        return data
    
    padded = np.pad(data, pad_len, mode='edge')
    filtered = signal.filtfilt(b, a, padded)
    
    return filtered[pad_len:-pad_len]


def calculate_sample_rate(time_array):
    """Calculate sample rate from time array using median for robustness."""
    dt_array = np.diff(time_array)
    median_dt = np.median(dt_array)
    return 1.0 / median_dt if median_dt > 0 else 20.0


def derive_velocity(altitude, time, sample_rate, filter_cutoff):
    """
    Derive velocity from altitude using filtered differentiation.
    """
    filtered_alt = apply_butterworth_filter(altitude, sample_rate, filter_cutoff)
    velocity = np.gradient(filtered_alt, time)
    return velocity, filtered_alt


# =============================================================================
# FLIGHT EVENT DETECTION
# =============================================================================

def detect_flight_events(time, altitude, velocity):
    """
    Detect key flight events from telemetry data.
    """
    events = {}
    
    # Apogee: maximum altitude
    apogee_idx = np.argmax(altitude)
    events['apogee'] = {
        'index': apogee_idx,
        'time': time[apogee_idx],
        'altitude': altitude[apogee_idx]
    }
    
    # Launch: first point where altitude exceeds threshold with positive velocity
    launch_alt_threshold = 10.0
    launch_candidates = np.where(
        (altitude > launch_alt_threshold) & 
        (velocity > 5.0)
    )[0]
    
    if len(launch_candidates) > 0:
        first_high = launch_candidates[0]
        launch_idx = first_high
        for i in range(first_high, 0, -1):
            if velocity[i] < 2.0:
                launch_idx = i + 1
                break
        launch_idx = max(0, min(launch_idx, len(time) - 1))
        events['launch'] = {
            'index': launch_idx,
            'time': time[launch_idx],
            'altitude': altitude[launch_idx]
        }
    else:
        events['launch'] = {
            'index': 0,
            'time': time[0],
            'altitude': altitude[0]
        }
    
    # Burnout: maximum velocity during ascent
    ascent_velocity = velocity[:apogee_idx]
    if len(ascent_velocity) > 0:
        burnout_idx = np.argmax(ascent_velocity)
        events['burnout'] = {
            'index': burnout_idx,
            'time': time[burnout_idx],
            'altitude': altitude[burnout_idx],
            'velocity': velocity[burnout_idx]
        }
    
    # Deployment: velocity stabilization after apogee
    if apogee_idx < len(time) - 10:
        descent_velocity = velocity[apogee_idx:]
        
        window_size = min(20, len(descent_velocity) // 4)
        if window_size > 5:
            velocity_variance = []
            for i in range(len(descent_velocity) - window_size):
                window = descent_velocity[i:i + window_size]
                velocity_variance.append(np.var(window))
            
            if len(velocity_variance) > 0:
                variance_threshold = np.median(velocity_variance) * 0.5
                stable_points = np.where(np.array(velocity_variance) < variance_threshold)[0]
                
                if len(stable_points) > 0:
                    deploy_local_idx = stable_points[0]
                    deploy_idx = apogee_idx + deploy_local_idx
                    
                    if deploy_idx > apogee_idx:
                        events['deployment'] = {
                            'index': deploy_idx,
                            'time': time[deploy_idx],
                            'altitude': altitude[deploy_idx]
                        }
    
    if 'deployment' not in events:
        events['deployment'] = {
            'index': apogee_idx,
            'time': time[apogee_idx],
            'altitude': altitude[apogee_idx]
        }
    
    # Landing
    landing_idx = len(time) - 1
    for i in range(len(altitude) - 1, max(apogee_idx, 0), -1):
        if altitude[i] > 5.0:
            landing_idx = min(i + 1, len(time) - 1)
            break
    
    events['landing'] = {
        'index': landing_idx,
        'time': time[landing_idx],
        'altitude': altitude[landing_idx]
    }
    
    return events


def calculate_descent_rate(altitude, velocity, events):
    """
    Calculate average descent rate under parachute.
    """
    if 'deployment' not in events or 'landing' not in events:
        return None
    
    deploy_idx = events['deployment']['index']
    land_idx = events['landing']['index']
    
    transient_skip = int((land_idx - deploy_idx) * 0.2)
    start_idx = deploy_idx + transient_skip
    
    if start_idx >= land_idx - 5:
        return None
    
    descent_velocities = velocity[start_idx:land_idx]
    
    if len(descent_velocities) > 5:
        descent_rate = -np.median(descent_velocities)
        return descent_rate if descent_rate > 0 else None
    
    return None


def assess_velocity_quality(velocity, sample_rate):
    """
    Assess the quality/noise level of derived velocity data.
    """
    nyquist = sample_rate / 2
    high_cutoff = min(0.4 * nyquist, 5.0) / nyquist
    
    if high_cutoff > 0 and high_cutoff < 1:
        b, a = signal.butter(2, high_cutoff, btype='high')
        try:
            noise_component = signal.filtfilt(b, a, velocity)
            noise_power = np.std(noise_component)
            signal_power = np.std(velocity)
            snr = signal_power / (noise_power + 0.001)
        except Exception:
            snr = 10
    else:
        snr = 10
    
    if snr > 20:
        quality = 'good'
    elif snr > 5:
        quality = 'moderate'
    else:
        quality = 'poor'
    
    return {
        'quality': quality,
        'snr': snr
    }


# =============================================================================
# LANDING ZONE ESTIMATION
# =============================================================================

def estimate_landing_zone(
    apogee_altitude_m,
    descent_rate_m_s,
    wind_speed_m_s,
    wind_direction_deg,
    drift_factor=0.6
):
    """
    Estimate landing zone based on wind drift during descent.
    """
    if descent_rate_m_s <= 0:
        return None
    
    descent_time_s = apogee_altitude_m / descent_rate_m_s
    
    effective_wind = wind_speed_m_s * drift_factor
    drift_distance_m = effective_wind * descent_time_s
    
    drift_direction_deg = (wind_direction_deg + 180) % 360
    drift_direction_rad = np.radians(drift_direction_deg)
    
    drift_x_m = drift_distance_m * np.sin(drift_direction_rad)
    drift_y_m = drift_distance_m * np.cos(drift_direction_rad)
    
    return {
        'descent_time_s': descent_time_s,
        'drift_distance_m': drift_distance_m,
        'drift_distance_ft': drift_distance_m * 3.28084,
        'drift_direction_deg': drift_direction_deg,
        'drift_x_m': drift_x_m,
        'drift_y_m': drift_y_m
    }


def create_landing_zone_plot(drift_data, wind_direction_deg):
    """Create a 2D overhead view of estimated landing zone."""
    
    fig = go.Figure()
    
    # Launch point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(size=15, color='green', symbol='triangle-up'),
        name='Launch Pad'
    ))
    
    # Estimated landing point
    fig.add_trace(go.Scatter(
        x=[drift_data['drift_x_m']],
        y=[drift_data['drift_y_m']],
        mode='markers',
        marker=dict(size=15, color='red', symbol='x'),
        name='Estimated Landing'
    ))
    
    # Drift line
    fig.add_trace(go.Scatter(
        x=[0, drift_data['drift_x_m']],
        y=[0, drift_data['drift_y_m']],
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
        name='Drift Path'
    ))
    
    # Uncertainty ellipse
    uncertainty_radius = drift_data['drift_distance_m'] * 0.2
    theta = np.linspace(0, 2 * np.pi, 50)
    ellipse_x = drift_data['drift_x_m'] + uncertainty_radius * np.cos(theta)
    ellipse_y = drift_data['drift_y_m'] + uncertainty_radius * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=ellipse_x,
        y=ellipse_y,
        mode='lines',
        line=dict(color='rgba(255, 100, 100, 0.5)', width=2),
        fill='toself',
        fillcolor='rgba(255, 100, 100, 0.1)',
        name='Uncertainty Zone'
    ))
    
    # Wind arrow
    arrow_length = drift_data['drift_distance_m'] * 0.3
    wind_rad = np.radians(wind_direction_deg)
    arrow_end_x = -arrow_length * np.sin(wind_rad)
    arrow_end_y = -arrow_length * np.cos(wind_rad)
    
    fig.add_annotation(
        x=arrow_end_x,
        y=arrow_end_y,
        ax=0,
        ay=0,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='blue'
    )
    
    fig.add_trace(go.Scatter(
        x=[arrow_end_x],
        y=[arrow_end_y],
        mode='text',
        text=['Wind'],
        textposition='top center',
        textfont=dict(color='blue'),
        showlegend=False
    ))
    
    max_range = max(abs(drift_data['drift_x_m']), abs(drift_data['drift_y_m']), 50) * 1.5
    
    fig.update_layout(
        title='Landing Zone Estimate (Overhead View)',
        xaxis_title='East-West (m)',
        yaxis_title='North-South (m)',
        xaxis=dict(
            range=[-max_range, max_range],
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(range=[-max_range, max_range]),
        height=500,
        showlegend=True,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
        )
    )
    
    return fig


# =============================================================================
# FLIGHT ANALYSIS PLOTS
# =============================================================================

def create_flight_plots(df, events, velocity_quality):
    """Generate flight analysis plots."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Altitude vs Time',
            'Vertical Velocity vs Time',
            'Flight Profile Overlay',
            'Descent Analysis'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    time = df['time_s'].values
    
    # Plot 1: Altitude vs Time
    fig.add_trace(
        go.Scatter(
            x=time,
            y=df['altitude_raw_m'],
            mode='lines',
            name='Raw Altitude',
            line=dict(color='lightgray', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=df['altitude_filtered_m'],
            mode='lines',
            name='Filtered Altitude',
            line=dict(color='#2E86AB', width=2)
        ),
        row=1, col=1
    )
    
    # Event markers
    event_colors = {
        'launch': 'green',
        'burnout': 'orange',
        'apogee': 'red',
        'deployment': 'purple',
        'landing': 'brown'
    }
    
    for event_name, event_data in events.items():
        idx = event_data['index']
        fig.add_trace(
            go.Scatter(
                x=[time[idx]],
                y=[df.loc[idx, 'altitude_filtered_m']],
                mode='markers',
                name=event_name.capitalize(),
                marker=dict(size=10, color=event_colors.get(event_name, 'gray')),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot 2: Velocity vs Time
    velocity_color = '#E94F37' if velocity_quality['quality'] != 'poor' else '#999999'
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=df['velocity_m_s'],
            mode='lines',
            name='Velocity',
            line=dict(color=velocity_color, width=1.5),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=2)
    
    # Plot 3: Flight Profile Overlay
    alt_normalized = df['altitude_filtered_m'] / df['altitude_filtered_m'].max()
    vel_max = max(abs(df['velocity_m_s'].max()), abs(df['velocity_m_s'].min()))
    vel_normalized = df['velocity_m_s'] / vel_max if vel_max > 0 else df['velocity_m_s']
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=alt_normalized,
            mode='lines',
            name='Altitude (norm)',
            line=dict(color='#2E86AB', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time,
            y=vel_normalized,
            mode='lines',
            name='Velocity (norm)',
            line=dict(color='#E94F37', width=2, dash='dot'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Plot 4: Descent Analysis
    apogee_idx = events['apogee']['index']
    landing_idx = events['landing']['index']
    
    descent_time = time[apogee_idx:landing_idx + 1]
    descent_vel = df['velocity_m_s'].values[apogee_idx:landing_idx + 1]
    
    if len(descent_time) > 0:
        fig.add_trace(
            go.Scatter(
                x=descent_time,
                y=-np.array(descent_vel),
                mode='lines',
                name='Descent Rate',
                line=dict(color='#7B2D8E', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Axis labels
    fig.update_xaxes(title_text='Time (s)', row=1, col=1)
    fig.update_yaxes(title_text='Altitude (m)', row=1, col=1)
    
    fig.update_xaxes(title_text='Time (s)', row=1, col=2)
    fig.update_yaxes(title_text='Velocity (m/s)', row=1, col=2)
    
    fig.update_xaxes(title_text='Time (s)', row=2, col=1)
    fig.update_yaxes(title_text='Normalized Value', row=2, col=1)
    
    fig.update_xaxes(title_text='Time (s)', row=2, col=2)
    fig.update_yaxes(title_text='Descent Rate (m/s)', row=2, col=2)
    
    fig.update_layout(
        height=650,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=40, t=80, b=40)
    )
    
    return fig


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_synthetic_flight(
    avg_thrust_n=50,
    burn_time_s=1.8,
    dry_mass_kg=0.45,
    propellant_mass_kg=0.06,
    diameter_m=0.041,
    cd_rocket=0.5,
    chute_diameter_m=0.6,
    cd_chute=1.5,
    sample_rate_hz=20,
    altitude_noise_m=1.0
):
    """
    Generate realistic TARC flight data with proper physics.
    Only outputs Time and Altitude (what a real altimeter provides).
    """
    
    dt = 1.0 / sample_rate_hz
    g = 9.80665
    rho = 1.225
    
    A_rocket = np.pi * (diameter_m / 2) ** 2
    A_chute = np.pi * (chute_diameter_m / 2) ** 2
    
    altitude = 0.0
    velocity = 0.0
    time = 0.0
    
    times = []
    altitudes = []
    
    propellant_remaining = propellant_mass_kg
    mass_flow_rate = propellant_mass_kg / burn_time_s
    motor_burning = True
    apogee_reached = False
    
    max_time = 120.0
    
    while time < max_time:
        current_mass = dry_mass_kg + propellant_remaining
        
        if not apogee_reached:
            drag_area = cd_rocket * A_rocket
        else:
            drag_area = cd_chute * A_chute
        
        if motor_burning and propellant_remaining > 0:
            thrust = avg_thrust_n
            propellant_remaining -= mass_flow_rate * dt
            if propellant_remaining <= 0:
                propellant_remaining = 0
                motor_burning = False
        else:
            thrust = 0
        
        drag = 0.5 * rho * velocity ** 2 * drag_area
        if velocity > 0:
            drag = -drag
        else:
            drag = abs(drag)
        
        weight = -current_mass * g
        
        net_force = thrust + drag + weight
        acceleration = net_force / current_mass
        
        velocity += acceleration * dt
        altitude += velocity * dt
        
        if not apogee_reached and velocity <= 0 and altitude > 10:
            apogee_reached = True
        
        times.append(time)
        altitudes.append(max(0, altitude))
        
        time += dt
        
        if apogee_reached and altitude <= 0:
            break
    
    times = np.array(times)
    altitudes = np.array(altitudes)
    
    noise = np.random.normal(0, altitude_noise_m, len(altitudes))
    noisy_altitudes = np.maximum(altitudes + noise, 0)
    
    altitudes_ft = noisy_altitudes / 0.3048
    
    return pd.DataFrame({
        'Time(s)': np.round(times, 3),
        'Altitude(ft)': np.round(altitudes_ft, 1)
    })


# =============================================================================
# DATA PROCESSING FUNCTION
# =============================================================================

def process_flight_data(df_input, filter_cutoff):
    """
    Process raw flight data and return processed dataframe, events, metrics.
    """
    df = pd.DataFrame()
    df['time_s'] = df_input['Time(s)'].values.astype(float)
    df['altitude_ft'] = df_input['Altitude(ft)'].values.astype(float)
    df['altitude_raw_m'] = df['altitude_ft'] * 0.3048
    
    # Remove pre-launch pad sitting
    launch_threshold = 3.0
    motion_idx = df[df['altitude_raw_m'] > launch_threshold].index.min()
    
    if pd.isna(motion_idx):
        return None, None, None, None, "No altitude data above launch threshold."
    
    start_idx = max(0, motion_idx - 5)
    df = df.loc[start_idx:].reset_index(drop=True)
    
    # Calculate sample rate
    sample_rate = calculate_sample_rate(df['time_s'].values)
    
    # Process data
    velocity, filtered_altitude = derive_velocity(
        df['altitude_raw_m'].values,
        df['time_s'].values,
        sample_rate,
        filter_cutoff
    )
    
    df['altitude_filtered_m'] = filtered_altitude
    df['velocity_m_s'] = velocity
    
    # Assess velocity quality
    velocity_quality = assess_velocity_quality(velocity, sample_rate)
    
    # Detect events
    events = detect_flight_events(
        df['time_s'].values,
        df['altitude_filtered_m'].values,
        df['velocity_m_s'].values
    )
    
    # Calculate descent rate
    descent_rate = calculate_descent_rate(
        df['altitude_filtered_m'].values,
        df['velocity_m_s'].values,
        events
    )
    
    return df, events, velocity_quality, descent_rate, None


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

st.set_page_config(
    page_title="AeroSight Telemetry",
    layout="wide",
    page_icon="🚀"
)

st.title("🚀 AeroSight: TARC Flight Telemetry Analyzer")
st.markdown(
    "Analyze altimeter data with signal processing. "
    "Detects flight events, calculates metrics, and estimates landing zones."
)

# Initialize session state for demo data
if 'demo_data' not in st.session_state:
    st.session_state.demo_data = None

# Sidebar
with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader(
        "Upload Altimeter CSV",
        type=['csv'],
        help="CSV with 'Time(s)' and 'Altitude(ft)' columns"
    )
    
    st.header("Signal Processing")
    filter_cutoff = st.slider(
        "Low-Pass Filter Cutoff (Hz)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Lower values = smoother data, but may lose sharp events"
    )
    
    st.header("Wind Conditions")
    wind_speed_mph = st.slider(
        "Wind Speed (mph)",
        min_value=0.0,
        max_value=25.0,
        value=8.0,
        step=1.0
    )
    wind_direction = st.slider(
        "Wind FROM Direction (°)",
        min_value=0,
        max_value=359,
        value=270,
        step=5,
        help="Meteorological convention: 270° = wind from West"
    )
    
    wind_speed_mps = wind_speed_mph * 0.44704

# Determine data source
raw_df = None

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        raw_df.columns = raw_df.columns.str.strip()
        st.session_state.demo_data = None  # Clear demo data when file uploaded
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

elif st.session_state.demo_data is not None:
    raw_df = st.session_state.demo_data

# Main content
if raw_df is not None:
    # Validate columns
    required_cols = ['Time(s)', 'Altitude(ft)']
    missing = [c for c in required_cols if c not in raw_df.columns]
    
    if missing:
        st.error(f"Missing columns: {missing}")
        st.info(f"Found: {list(raw_df.columns)}")
    else:
        # Process data
        df, events, velocity_quality, descent_rate, error = process_flight_data(
            raw_df, filter_cutoff
        )
        
        if error:
            st.error(error)
        elif df is not None:
            # Display metrics
            st.subheader("Flight Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                apogee_ft = events['apogee']['altitude'] / 0.3048
                st.metric("Apogee", f"{apogee_ft:.0f} ft")
            
            with col2:
                if 'burnout' in events and 'velocity' in events['burnout']:
                    max_vel = events['burnout']['velocity']
                    st.metric("Max Velocity", f"{max_vel:.1f} m/s")
                else:
                    st.metric("Max Velocity", "N/A")
            
            with col3:
                if descent_rate:
                    st.metric("Descent Rate", f"{descent_rate:.1f} m/s")
                else:
                    st.metric("Descent Rate", "N/A")
            
            with col4:
                flight_time = events['landing']['time'] - events['launch']['time']
                st.metric("Flight Time", f"{flight_time:.1f} s")
            
            # Velocity quality indicator
            quality_colors = {'good': '🟢', 'moderate': '🟡', 'poor': '🔴'}
            st.caption(
                f"Velocity data quality: {quality_colors.get(velocity_quality['quality'], '⚪')} "
                f"{velocity_quality['quality'].capitalize()} (SNR: {velocity_quality['snr']:.1f})"
            )
            
            # Flight events timeline
            st.subheader("Flight Events")
            
            event_order = ['launch', 'burnout', 'apogee', 'deployment', 'landing']
            available_events = [e for e in event_order if e in events]
            
            event_cols = st.columns(len(available_events))
            
            for i, event_name in enumerate(available_events):
                event_data = events[event_name]
                with event_cols[i]:
                    st.markdown(f"**{event_name.capitalize()}**")
                    st.markdown(f"T+ {event_data['time']:.2f} s")
                    st.markdown(f"{event_data['altitude']:.1f} m")
            
            st.markdown("---")
            
            # Tabs
            tab1, tab2, tab3 = st.tabs([
                "📈 Flight Analysis",
                "🎯 Landing Zone",
                "📊 Data Export"
            ])
            
            with tab1:
                fig = create_flight_plots(df, events, velocity_quality)
                st.plotly_chart(fig, use_container_width=True)
                
                if velocity_quality['quality'] == 'poor':
                    st.warning(
                        "⚠️ Velocity data has high noise levels. This is normal for "
                        "barometric altitude data. Consider using a lower filter cutoff."
                    )
            
            with tab2:
                if descent_rate and descent_rate > 0:
                    drift_data = estimate_landing_zone(
                        events['apogee']['altitude'],
                        descent_rate,
                        wind_speed_mps,
                        wind_direction
                    )
                    
                    if drift_data:
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("### Drift Estimates")
                            st.metric(
                                "Descent Time",
                                f"{drift_data['descent_time_s']:.1f} s"
                            )
                            st.metric(
                                "Drift Distance",
                                f"{drift_data['drift_distance_m']:.0f} m "
                                f"({drift_data['drift_distance_ft']:.0f} ft)"
                            )
                            st.metric(
                                "Drift Direction",
                                f"{drift_data['drift_direction_deg']:.0f}°"
                            )
                            
                            st.markdown("---")
                            st.markdown("### Assumptions")
                            st.caption(
                                "• Drift factor: 60% of wind speed\n"
                                "• Constant wind (no shear)\n"
                                "• Straight-line drift\n"
                                "• Uncertainty zone: ±20%"
                            )
                        
                        with col2:
                            landing_fig = create_landing_zone_plot(
                                drift_data,
                                wind_direction
                            )
                            st.plotly_chart(landing_fig, use_container_width=True)
                else:
                    st.warning("Cannot estimate landing zone: descent rate not available.")
            
            with tab3:
                st.markdown("### Processed Telemetry Data")
                
                export_df = df[[
                    'time_s',
                    'altitude_raw_m',
                    'altitude_filtered_m',
                    'velocity_m_s'
                ]].copy()
                
                export_df.columns = [
                    'Time (s)',
                    'Raw Altitude (m)',
                    'Filtered Altitude (m)',
                    'Velocity (m/s)'
                ]
                
                st.dataframe(
                    export_df.round(2),
                    use_container_width=True,
                    height=400
                )
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    "Download Processed Data (CSV)",
                    csv,
                    "processed_telemetry.csv",
                    "text/csv"
                )
                
                st.markdown("### Event Summary")
                event_summary = []
                for name, data in events.items():
                    event_summary.append({
                        'Event': name.capitalize(),
                        'Time (s)': data['time'],
                        'Altitude (m)': data['altitude'],
                        'Altitude (ft)': data['altitude'] / 0.3048
                    })
                
                event_df = pd.DataFrame(event_summary)
                st.dataframe(event_df.round(2), use_container_width=True)
            
            # Technical notes
            with st.expander("Technical Notes"):
                st.markdown(f"""
                **Signal Processing:**
                - Filter: 2nd-order Butterworth low-pass
                - Cutoff frequency: {filter_cutoff} Hz
                - Method: Zero-phase (filtfilt) to avoid time delay
                
                **Velocity Derivation:**
                - Altitude filtered first to reduce noise
                - Central difference method (numpy.gradient)
                - Quality assessed via signal-to-noise ratio
                
                **Event Detection:**
                - Apogee: Maximum filtered altitude
                - Burnout: Maximum velocity during ascent
                - Deployment: Velocity stabilization after apogee
                - Launch/Landing: Threshold-based detection
                
                **Landing Zone Estimation:**
                - Assumes 60% wind drift factor
                - No wind shear modeling
                - 20% uncertainty margin shown
                """)

else:
    # No data loaded - show welcome screen with demo option
    st.info("👈 Upload an altimeter CSV file to begin analysis, or try the demo below.")
    
    st.markdown("---")
    
    # Demo section
    st.subheader("🧪 Try Demo Flight")
    st.markdown(
        "No data? Generate a synthetic TARC flight to explore the tool's features."
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Flight Parameters**")
        demo_thrust = st.selectbox(
            "Motor Class",
            options=["E (30N)", "F (50N)", "G (80N)"],
            index=1
        )
        thrust_map = {"E (30N)": 30, "F (50N)": 50, "G (80N)": 80}
        thrust_n = thrust_map[demo_thrust]
        
        demo_mass = st.slider(
            "Rocket Mass (kg)",
            min_value=0.3,
            max_value=0.8,
            value=0.45,
            step=0.05
        )
        
        demo_chute = st.slider(
            "Chute Diameter (m)",
            min_value=0.3,
            max_value=1.0,
            value=0.6,
            step=0.1
        )
        
        if st.button("🚀 Generate Demo Flight", type="primary", use_container_width=True):
            with st.spinner("Simulating flight..."):
                st.session_state.demo_data = generate_synthetic_flight(
                    avg_thrust_n=thrust_n,
                    dry_mass_kg=demo_mass,
                    chute_diameter_m=demo_chute,
                    altitude_noise_m=0.8
                )
            st.rerun()
    
    with col2:
        st.markdown("**Expected CSV Format**")
        st.markdown(
            """
            Required columns:
            - `Time(s)` — Elapsed time from logger start
            - `Altitude(ft)` — Barometric altitude in feet AGL
            
            Example:
            ```
            Time(s),Altitude(ft)
            0.00,0
            0.05,5
            0.10,22
            0.15,48
            ...
            ```
            
            Compatible with most barometric altimeters:
            - Stratologger
            - Altus Metrum
            - Eggtimer
            - PerfectFlite
            """
        )
    
    # Clear demo data button (if demo is active)
    if st.session_state.demo_data is not None:
        if st.button("Clear Demo Data"):
            st.session_state.demo_data = None
            st.rerun()