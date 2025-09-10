# MCP Weather & AQI Health Assistant

A Model Context Protocol (MCP) based application that provides weather data, air quality index (AQI) reports, and AI-powered health recommendations using Intel® Core™ Ultra Processors.

## Quick Start

### Prerequisites
- Python 3.13 or higher
- `uv` package manager


### 1. Install uv (if not already installed)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Set Python Version
```powershell
uv python pin 3.13
```
To run OV model: 
& "$env:USERPROFILE\.local\bin\uv.exe" add optimum[openvino,intel]
& "$env:USERPROFILE\.local\bin\uv.exe" add openvino-genai huggingface-hub

### 3. Start the MCP Servers

Open **4 separate terminal windows** and run each command:

#### Terminal 1 - Weather Server (Port 8000)
```powershell
uv run 1_weather_server.py
```

#### Terminal 2 - AQI Server (Port 8001)
```powershell
uv run 2_Air_Quality_Index_server.py
```

#### Terminal 3 - LLM Server (Port 8002)

**Option A: Standard Qwen Model (6GB)**
```powershell
uv run 3_LLM_inference_server.py
```

**Option B: OpenVINO Optimized Model (1.5GB)**  
```powershell
uv run 3_LLM_inference_server_OV.py
```

*Note: First run will download the selected AI model. Option A downloads ~6GB Qwen 2.5-3B model, Option B downloads ~1.5GB optimized model.*

#### Terminal 4 - Run the Client
```powershell
uv run Weather_AQI_MCP_Assistant.py
```

##  Usage

1. The client will prompt you for a location
2. Enter any city name (e.g., "Seattle", "New York", "London")
3. Get comprehensive reports:
   - Current weather conditions
   - Air quality index and pollutant levels
   - AI-generated health and safety recommendations
4. Type `exit` to quit

## Alternative Setup (If uv is not in PATH)

If you encounter `uv : The term 'uv' is not recognized` errors, use the full path commands in separate terminals:

```powershell
# Terminal 1 - Weather Server
& "$env:USERPROFILE\.local\bin\uv.exe" run 1_weather_server.py

# Terminal 2 - AQI Server  
& "$env:USERPROFILE\.local\bin\uv.exe" run 2_Air_Quality_Index_server.py

# Terminal 3 - LLM Server (Option A: Standard)
& "$env:USERPROFILE\.local\bin\uv.exe" run 3_LLM_inference_server.py

# Terminal 3 - LLM Server (Option B: OpenVINO Optimized)
& "$env:USERPROFILE\.local\bin\uv.exe" run 3_LLM_inference_server_OV.py

# Terminal 4 - Client
& "$env:USERPROFILE\.local\bin\uv.exe" run Weather_AQI_MCP_Assistant.py
```

## Example Output

```
Enter location to check for Weather and AQI reports (or 'exit' to quit): Seattle

Fetching Weather & AQI data...

Weather Report:
- Location: Seattle, United States
- Coordinates: 47.60621, -122.33207
- Temperature: 13.9°C
- Wind Speed: 8.8 km/h

Air Quality Index Report for 'Seattle':
- Location: seattle, United States
- AQI Level: 1 (Good)
- Pollutants (μg/m3): CO: 88.89, NO2: 2.59, O3: 46.19...

Health & Safety Advice:
 Overall Outdoor Safety: EXCELLENT for outdoor activities
 Perfect weather for outdoor activities!
 AIR QUALITY: Excellent - safe for all outdoor activities
...
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Weather Server  │    │   AQI Server    │    │   LLM Server    │
│   Port 8000     │    │   Port 8001     │    │   Port 8002     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  MCP Client     │
                    │  Assistant      │
                    └─────────────────┘
```

##  Components

- **Weather Server** (`1_weather_server.py`): Fetches weather data using Open-Meteo API
- **AQI Server** (`2_Air_Quality_Index_server.py`): Retrieves air quality data using OpenWeatherMap API  
- **LLM Server** (`3_LLM_inference_server.py`): Provides AI health recommendations using Qwen 2.5-3B model
- **LLM Server (OpenVINO)** (`3_LLM_inference_server_OV_fixed.py`): Optimized AI recommendations using Intel OpenVINO (smaller, faster)
- **Client** (`Weather_AQI_MCP_Assistant.py`): Orchestrates all servers and provides user interface

## Troubleshooting

### Port Already in Use
If you get port errors, kill existing Python processes:
```powershell
taskkill /F /IM python.exe
```

### LLM Model Issues
- **Option A**: First run downloads ~6GB Qwen model - be patient
- **Option B**: First run downloads ~1.5GB OpenVINO model - faster and smaller
- If Qwen fails, both servers automatically fall back to DistilGPT-2
- Ensure stable internet connection for model download
- Choose Option B for faster inference and smaller memory footprint

### uv Not Found
Add to PATH or use full path:
```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run [script_name]
```

## APIs Used

- [Open-Meteo Weather API](https://open-meteo.com/) - Weather data
- [OpenWeatherMap Air Pollution API](https://openweathermap.org/api/air-pollution) - AQI data
- [Qwen 2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) - AI model for health recommendations

## Intel® Acceleration

This project leverages Intel® Core™ Ultra Processors with PyTorch XPU backend for accelerated AI inference, providing fast and efficient health recommendations.

## Technical Details

### MCP Framework
Built using FastMCP, a high-level Pythonic framework inspired by FastAPI that simplifies Model Context Protocol implementation.

### AI Models
- **Standard**: Qwen 2.5-3B-Instruct (6GB download)
- **OpenVINO Optimized**: Qwen 2.5-1.5B-Instruct-INT8-OV (1.5GB download, faster inference)
- **Fallback**: DistilGPT-2 for compatibility
- **Hardware**: Optimized for Intel® Core™ Ultra Processors with OpenVINO acceleration

### Data Sources
- Real-time weather data from Open-Meteo
- Air quality measurements from OpenWeatherMap
- Comprehensive pollutant analysis (CO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃)