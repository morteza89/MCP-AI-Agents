
from fastmcp import FastMCP
import httpx
import os
from dotenv import load_dotenv

# Initialize MCP Server
mcp = FastMCP("Air Quality Index", host="0.0.0.0", port=8001)

# Load API key for OpenWeatherMap AQI
load_dotenv()
OPENWEATHERMAP_API_KEY = os.getenv("AQI_API_KEY")


# Geocoding function to get latitude/longitude from location using Open-Meteo
async def get_coordinates(location: str):
    """
    Fetch the geographical coordinates (latitude and longitude)
    and country name for a given location using the Open-Meteo Geocoding API.

    Args:
        location (str): The name of the location to geocode (e.g., 'Delhi').

    Returns:
        tuple: A tuple containing:
            - latitude (float or None): The latitude of the location.
            - longitude (float or None): The longitude of the location.
            - country (str or None): The country name of the location.
        If the location cannot be found, returns (None, None, None).
    """
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
    try:
        async with httpx.AsyncClient() as client:
            geo_response = await client.get(geo_url)
            geo_response.raise_for_status()  # raises HTTPStatusError for bad responses
            geo_data = geo_response.json()

        if "results" not in geo_data or not geo_data["results"]:
            return None, None, None

        result = geo_data["results"][0]
        return result["latitude"], result["longitude"], result["country"]

    except httpx.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")

    return None, None, None


# AQI Tool


@mcp.tool()
async def get_aqi(location: str) -> str:
    """
    Retrieve the Air Quality Index (AQI) and pollutant concentrations
    for a given location using the OpenWeatherMap Air Pollution API.

    This tool first fetches the latitude and longitude for the location,
    then requests the AQI data from OpenWeatherMap.

    Args:
        location (str): The name of the location to get AQI data for.

    Returns:
        str: A formatted string containing:
            - Location and country
            - Coordinates (latitude, longitude)
            - AQI level and description
            - Concentrations of various pollutants (CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3)

        Returns an error message if:
            - The AQI API key is missing
            - Coordinates cannot be found for the location
            - Network request fails
            - Invalid response is received
    """
    if not OPENWEATHERMAP_API_KEY:
        return "AQI API key is missing. Set the 'AQI_API_KEY' environment variable."

    try:
        lat, lon, country = await get_coordinates(location)
    except Exception as e:
        return f"Failed to get coordinates for '{location}': {str(e)}"

    if lat is None or lon is None:
        return f"Unable to get coordinates for '{location}'."

    aqi_url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution?"
        f"lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(aqi_url)
            response.raise_for_status()  # Raises an HTTPStatusError for 4xx/5xx
            aqi_data = response.json()
    except httpx.RequestError as e:
        return f"Network error while fetching AQI: {str(e)}"
    except httpx.HTTPStatusError as e:
        return f"API returned an error: {e.response.status_code} {e.response.text}"
    except Exception as e:
        return f"Unexpected error while fetching AQI: {str(e)}"

    if "list" not in aqi_data or not aqi_data["list"]:
        return f"No AQI data found for '{location}'."

    try:
        aqi = aqi_data["list"][0]["main"]["aqi"]
        components = aqi_data["list"][0]["components"]
        # AQI level explanation based on OpenWeatherMap
        levels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}

        return f"""
    - Location: {location}, {country}
    - Coordinates: {lat}, {lon}
    - AQI Level: {aqi} ({levels.get(aqi, 'Unknown')})

     Pollutants (μg/m3):
     - CO: {components.get('co', 'N/A')}
     - NO: {components.get('no', 'N/A')}
     - NO2: {components.get('no2', 'N/A')}
     - O3: {components.get('o3', 'N/A')}
     - SO2: {components.get('so2', 'N/A')}
     - PM2.5: {components.get('pm2_5', 'N/A')}
     - PM10: {components.get('pm10', 'N/A')}
     - NH3: {components.get('nh3', 'N/A')}
     """
    except Exception as e:
        return f"Failed to parse AQI response: {str(e)}"


if __name__ == "__main__":
    mcp.run("sse")