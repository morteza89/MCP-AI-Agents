import nest_asyncio
nest_asyncio.apply()

import asyncio
import os
from fastmcp import Client

import asyncio
from fastmcp import Client

class AQI_Weather_Advisor:
    """
    AQI_Weather_Advisor is responsible for interacting with Weather, AQI, and LLM MCP servers.

    It provides methods to:
    - Fetch weather information for a given location.
    - Retrieve AQI (Air Quality Index) reports.
    - Generate health and safety recommendations based on weather and AQI data.
    """

    def __init__(self, weather_url: str, aqi_server_url: str, llm_server_url: str):
        """
        Initialize the AQI_Weather_Advisor with URLs for Weather, AQI, and LLM MCP servers.

        Args:
            weather_url (str): URL of the Weather MCP server.
            aqi_server_url (str): URL of the AQI MCP server.
            llm_server_url (str): URL of the LLM MCP server.
        """
        self.weather_url = weather_url
        self.aqi_server_url = aqi_server_url
        self.llm_server_url = llm_server_url

    async def get_weather(self, location: str) -> str:
        """
        Retrieve weather data for a given location from the Weather MCP server.

        Args:
            location (str): Name of the location to get weather information for.

        Returns:
            str: Raw weather data response or error message.
        """
        try:
            async with Client(f"{self.weather_url}/sse") as client:
                return await client.call_tool("get_weather", {"location": location})
        except Exception as e:
            return f" Failed to get weather data for '{location}': {str(e)}"

    async def get_aqi_report(self, location: str) -> str:
        """
        Retrieve AQI (Air Quality Index) report for a given location from the AQI MCP server.

        Args:
            location (str): Name of the location to get AQI report for.

        Returns:
            str: AQI report as plain text or error message.
        """
        try:
            async with Client(f"{self.aqi_server_url}/sse") as client:
                result = await client.call_tool("get_aqi", {"location": location})
                return self._extract_text(result)
        except Exception as e:
            return f" Failed to get AQI report for '{location}': {str(e)}"

    async def get_health_recommendations(self, weather_report: str, aqi_report: str) -> str:
        """
        Get health and safety recommendations by calling the LLM MCP server.

        Args:
            weather_report (str): Weather report text.
            aqi_report (str): AQI report text.

        Returns:
            str: Health and safety recommendations or error message.
        """
        try:
            async with Client(f"{self.llm_server_url}/sse") as client:
                result = await client.call_tool("safety_guidelines", {
                    "weather_report": weather_report,
                    "aqi_report": aqi_report
                })
                return self._extract_text(result)
        except Exception as e:
            return f" Failed to get safety recommendations: {str(e)}"

    def _extract_text(self, result) -> str:
        """
        Helper method to extract plain text from MCP results.

        Args:
            result (Any): The result returned from an MCP tool call.

        Returns:
            str: Extracted text or string representation.
        """
        try:
            if isinstance(result, list):
                return "\n".join(block.text for block in result if hasattr(block, "text"))
            elif hasattr(result, "text"):
                return result.text
            return str(result)
        except Exception as e:
            return f" Failed to parse result: {str(e)}"

async def main():
    """
    Main entry point of the MCP based AQI_Weather_Health Assistant.

    Continuously prompts the user for a location, fetches weather and AQI data,
    and provides health and safety recommendations until the user chooses to exit.
    """
    agent = AQI_Weather_Advisor("http://localhost:8000" , "http://localhost:8001", "http://localhost:8002")

    while True:
        location = input("\n Enter location to check for Weather and AQI reports (or 'exit' to quit): ").strip()
        if location.lower() == "exit":
            print(" Exiting Weather & AQI Assistant.")
            break

        print("\n Fetching Weather & AQI data...")
        try:
                   
            weather_raw = await agent.get_weather(location)
            weather_report = weather_raw[0].text if isinstance(weather_raw, list) else str(weather_raw)
            print(f"\n Weather Report:\n {weather_report}")
            
            aqi_report = await agent.get_aqi_report(location)
            print(f"\n Air Quality Index Report for '{location}':\n{aqi_report}")

            print("\n Getting health precautions...")
            recommendations = await agent.get_health_recommendations(weather_report, aqi_report)
            print("\n Health & Safety Advice:\n", recommendations)
        except Exception as e:
            print(" An error occurred:", str(e))

if __name__ == "__main__":
    asyncio.run(main())