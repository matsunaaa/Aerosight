# AeroSight

Rocketry flight telemetry analyzer with signal processing, event detection, and landing zone estimation.

## Overview

AeroSight processes barometric altimeter data from high-power rocket flights. It applies digital signal processing to extract flight metrics, automatically detects flight events, and estimates landing zones based on wind conditions.

Built for TARC (The American Rocketry Challenge) post-flight analysis.

## Features

- **Signal Processing**: Butterworth low-pass filtering for noise reduction
- **Event Detection**: Automatic identification of launch, burnout, apogee, deployment, and landing
- **Velocity Derivation**: Calculates vertical velocity from altitude with quality assessment
- **Landing Zone Estimation**: Predicts drift distance and direction based on wind input
- **Data Export**: Download processed telemetry and event summaries as CSV

## Installation

```bash
git clone https://github.com/yourusername/aerosight.git
cd aerosight
pip install -r requirements.txt
streamlit run app.py
```

### Dependencies

```
streamlit
pandas
numpy
scipy
plotly
```

## Usage

### Input Format

CSV file with two required columns:

```csv
Time(s),Altitude(ft)
0.00,0
0.05,5
0.10,22
```

- `Time(s)`: Elapsed time from logger start
- `Altitude(ft)`: Barometric altitude in feet AGL

### Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Filter Cutoff | 1-10 Hz | Butterworth filter cutoff frequency. Lower = smoother, higher = more detail |
| Wind Speed | 0-25 mph | Surface wind speed for drift calculation |
| Wind Direction | 0-359° | Direction wind is coming FROM (meteorological convention) |

## Technical Details

### Signal Processing Pipeline

```
Raw Altitude → Butterworth LPF → Filtered Altitude → Differentiation → Velocity
```

**Filter**: 2nd-order Butterworth low-pass filter applied using `scipy.signal.filtfilt` for zero phase delay.

**Differentiation**: Central difference method via `numpy.gradient`. Filtering before differentiation reduces noise amplification.

**Why not derive acceleration?** Double differentiation of barometric data produces unreliable results. Acceleration measurement requires an accelerometer.

### Event Detection

| Event | Detection Method |
|-------|------------------|
| Launch | First altitude threshold crossing with positive velocity |
| Burnout | Maximum velocity during ascent phase |
| Apogee | Maximum filtered altitude |
| Deployment | Velocity stabilization after apogee (variance-based) |
| Landing | Altitude stabilization near ground level |

### Velocity Quality Assessment

Signal-to-noise ratio (SNR) calculated by comparing signal power to high-frequency noise content:

| SNR | Quality | Interpretation |
|-----|---------|----------------|
| >20 | Good | Reliable velocity data |
| 5-20 | Moderate | Usable with caution |
| <5 | Poor | Consider lower filter cutoff |

### Landing Zone Estimation

Drift distance calculated as:

```
drift = wind_speed × drift_factor × (apogee_altitude / descent_rate)
```

**Assumptions**:
- Drift factor: 0.6 (parachute does not drift at full wind speed)
- Constant wind (no altitude-dependent shear)
- Straight-line drift path
- Uncertainty margin: ±20%

## Output Metrics

| Metric | Unit | Source |
|--------|------|--------|
| Apogee | ft | Max filtered altitude |
| Max Velocity | m/s | Peak velocity at burnout |
| Descent Rate | m/s | Median velocity during chute descent |
| Flight Time | s | Launch to landing duration |
| Drift Distance | m | Calculated from wind and descent rate |

## Limitations

1. **No horizontal position data**: Barometric altimeters measure altitude only. True trajectory reconstruction requires IMU (accelerometer + gyroscope) data.

2. **Velocity is derived, not measured**: Differentiation amplifies sensor noise. Velocity values are approximate.

3. **Acceleration not provided**: Deriving acceleration from barometric altitude produces unreliable data.

4. **Landing zone is an estimate**: Based on assumed wind conditions and simplified drift model.

## Example Output

**Flight Metrics**
- Apogee: 2221 ft
- Max Velocity: 135.8 m/s
- Descent Rate: 5.0 m/s
- Flight Time: 119.5 s

**Flight Events**
| Event | Time | Altitude |
|-------|------|----------|
| Launch | T+0.40s | 10.1 m |
| Burnout | T+1.45s | 106.4 m |
| Apogee | T+11.25s | 676.9 m |
| Deployment | T+13.05s | 667.2 m |
| Landing | T+119.95s | 28.3 m |

## License

MIT

## Author

Emily Wu