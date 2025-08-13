# weather_server.py

import httpx
from mcp.server.fastmcp import FastMCP

# Initialize MCP Server
mcp = FastMCP("weather", host="0.0.0.0", port=8000)

GEOCODE_API = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

# Weather tool


@mcp.tool()
async def get_weather(location: str) -> str:
    """
    Retrieve the current weather information for a given city or country.

    This function performs two steps:
      1. Uses the Open-Meteo Geocoding API to find the latitude and longitude
         of the specified location.
      2. Uses the Open-Meteo Weather API to fetch the current weather forecast
         for those coordinates.

    The returned weather report includes:
      - Location name and country
      - Coordinates (latitude and longitude)
      - Current temperature (°C)
      - Wind speed (km/h)

    Args:
        location (str): The name of the city or country to get weather data for.

    Returns:
        str: A formatted string containing the location details and current weather.
             If the location is not found or weather data is unavailable,
             an appropriate error message is returned instead.
    """
    # Step 1: Geocoding to get latitude/longitude
    geo_url = f"{GEOCODE_API}?name={location}&count=1&language=en&format=json"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            geo_response = await client.get(geo_url)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
    except httpx.RequestError as e:
        return f"Network error while fetching coordinates: {str(e)}"
    except httpx.HTTPStatusError as e:
        return f"Geocoding API returned an error: {e.response.status_code} {e.response.text}"
    except Exception as e:
        return f"Unexpected error during geocoding: {str(e)}"

    if "results" not in geo_data or not geo_data["results"]:
        return f"Location '{location}' not found."

    try:
        result = geo_data["results"][0]
        lat = result["latitude"]
        lon = result["longitude"]
        name = result["name"]
        country = result.get("country", "Unknown")
    except Exception as e:
        return f"Failed to parse geocoding response: {str(e)}"

    # Step 2: Get weather forecast
    forecast_url = (
        f"{WEATHER_API}?latitude={lat}&longitude={lon}"
        "&current_weather=true&timezone=auto"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            weather_response = await client.get(forecast_url)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
    except httpx.RequestError as e:
        return f"Network error while fetching weather: {str(e)}"
    except httpx.HTTPStatusError as e:
        return (
            f"Weather API returned an error: {e.response.status_code} {e.response.text}"
        )
    except Exception as e:
        return f"Unexpected error during weather fetch: {str(e)}"

    if "current_weather" not in weather_data:
        return f"Weather data not available for '{name}, {country}'."

    try:
        current = weather_data["current_weather"]
        temp = current.get("temperature", "N/A")
        wind = current.get("windspeed", "N/A")
    except Exception as e:
        return f"Failed to parse weather data: {str(e)}"

    return f"""
        - Location: {name}, {country}
        - Coordinates: {lat}, {lon}
        - Temperature: {temp}°C
        - Wind Speed: {wind} km/h
        """


if __name__ == "__main__":
    mcp.run("sse")