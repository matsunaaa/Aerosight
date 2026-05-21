import pandas as pd
import numpy as np

def generate_tarc_flight(
    # Motor parameters
    avg_thrust_n=45,           # Average thrust (Newtons) - typical F motor
    burn_time_s=1.8,           # Motor burn time
    
    # Rocket parameters  
    dry_mass_kg=0.4,           # Rocket mass without motor
    propellant_mass_kg=0.06,   # Propellant mass
    diameter_m=0.041,          # Body tube diameter (41mm = common size)
    cd_rocket=0.5,             # Drag coefficient (no chute)
    
    # Recovery parameters
    drogue_diameter_m=0.3,     # Drogue chute diameter
    main_diameter_m=0.9,       # Main chute diameter
    main_deploy_alt_m=150,     # Main deployment altitude AGL
    cd_chute=1.5,              # Parachute drag coefficient
    
    # Simulation parameters
    sample_rate_hz=20,         # Altimeter sample rate
    altitude_noise_m=1.0,      # Barometric noise standard deviation
    
    # Environment
    rail_length_m=1.0          # Launch rail length
):
    """
    Generate realistic TARC flight data.
    
    Physics modeled:
    - Variable mass during burn (propellant consumption)
    - Velocity-dependent drag (proportional to v²)
    - Gravity
    - Parachute deployment dynamics
    
    Returns DataFrame with Time(s) and Altitude(ft) only,
    matching what a real barometric altimeter outputs.
    """
    
    dt = 1.0 / sample_rate_hz
    g = 9.80665  # m/s²
    rho = 1.225  # Air density kg/m³
    
    # Rocket cross-sectional area
    A_rocket = np.pi * (diameter_m / 2) ** 2
    
    # Chute areas
    A_drogue = np.pi * (drogue_diameter_m / 2) ** 2
    A_main = np.pi * (main_diameter_m / 2) ** 2
    
    # State variables
    altitude = 0.0
    velocity = 0.0
    time = 0.0
    
    # Data storage
    times = []
    altitudes = []
    
    # Flight phases
    on_rail = True
    motor_burning = True
    apogee_reached = False
    main_deployed = False
    
    # Track propellant mass
    propellant_remaining = propellant_mass_kg
    mass_flow_rate = propellant_mass_kg / burn_time_s
    
    max_iterations = int(120 / dt)  # 2 minute max
    
    for _ in range(max_iterations):
        # Current mass
        current_mass = dry_mass_kg + propellant_remaining
        
        # Determine drag area based on flight phase
        if not apogee_reached:
            drag_area = cd_rocket * A_rocket
        elif not main_deployed:
            drag_area = cd_chute * A_drogue
        else:
            drag_area = cd_chute * A_main
        
        # Thrust (if motor still burning)
        if motor_burning and propellant_remaining > 0:
            thrust = avg_thrust_n
            propellant_remaining -= mass_flow_rate * dt
            if propellant_remaining <= 0:
                propellant_remaining = 0
                motor_burning = False
        else:
            thrust = 0
        
        # Drag force (always opposes velocity)
        # F_drag = 0.5 * rho * v² * Cd * A
        drag = 0.5 * rho * velocity ** 2 * drag_area
        if velocity > 0:
            drag = -drag  # Opposes upward motion
        else:
            drag = abs(drag)  # Opposes downward motion (positive = upward force)
        
        # Weight
        weight = -current_mass * g
        
        # Net force and acceleration
        net_force = thrust + drag + weight
        acceleration = net_force / current_mass
        
        # Rail constraint (no negative velocity on rail)
        if on_rail:
            if velocity < 0:
                velocity = 0
                acceleration = max(0, acceleration)
            if altitude >= rail_length_m:
                on_rail = False
        
        # Integrate
        velocity += acceleration * dt
        altitude += velocity * dt
        
        # Check for apogee
        if not apogee_reached and velocity <= 0 and altitude > rail_length_m:
            apogee_reached = True
        
        # Check for main deployment
        if apogee_reached and not main_deployed and altitude <= main_deploy_alt_m:
            main_deployed = True
        
        # Store data
        times.append(time)
        altitudes.append(altitude)
        
        time += dt
        
        # Check for landing
        if apogee_reached and altitude <= 0:
            altitudes[-1] = 0  # Clamp to ground
            break
    
    # Convert to numpy arrays
    times = np.array(times)
    altitudes = np.array(altitudes)
    
    # Add realistic barometric noise
    noise = np.random.normal(0, altitude_noise_m, len(altitudes))
    noisy_altitudes = altitudes + noise
    noisy_altitudes = np.maximum(noisy_altitudes, 0)  # Can't read negative altitude
    
    # Convert to feet (TARC uses imperial)
    altitudes_ft = noisy_altitudes / 0.3048
    
    return pd.DataFrame({
        'Time(s)': np.round(times, 3),
        'Altitude(ft)': np.round(altitudes_ft, 1)
    })


# Generate sample flight
df = generate_tarc_flight(
    avg_thrust_n=50,          # ~F motor
    burn_time_s=1.5,
    dry_mass_kg=0.45,
    propellant_mass_kg=0.05,
    altitude_noise_m=0.8,     # Reasonable barometer noise
    sample_rate_hz=20
)

df.to_csv('synthetic_tarc_flight.csv', index=False)

# Print summary
print(f"Flight duration: {df['Time(s)'].max():.1f} s")
print(f"Apogee: {df['Altitude(ft)'].max():.0f} ft")
print(f"Data points: {len(df)}")