import nest_asyncio
nest_asyncio.apply()

import asyncio
import os
from fastmcp import Client

import asyncio
from fastmcp import Client

class Stock_Market_Advisor:
    """
    Stock_Market_Advisor is responsible for interacting with Stock Price, Company Info, and LLM MCP servers.

    It provides methods to:
    - Fetch stock price information for a given company symbol.
    - Retrieve company information and financial metrics.
    - Generate investment analysis and recommendations based on stock and company data.
    """

    def __init__(self, stock_price_url: str, company_info_url: str, llm_server_url: str):
        """
        Initialize the Stock_Market_Advisor with URLs for Stock Price, Company Info, and LLM MCP servers.

        Args:
            stock_price_url (str): URL of the Stock Price MCP server.
            company_info_url (str): URL of the Company Info MCP server.
            llm_server_url (str): URL of the LLM MCP server.
        """
        self.stock_price_url = stock_price_url
        self.company_info_url = company_info_url
        self.llm_server_url = llm_server_url

    async def get_stock_price(self, symbol: str) -> str:
        """
        Retrieve stock price data for a given symbol from the Stock Price MCP server.

        Args:
            symbol (str): Stock symbol to get price information for.

        Returns:
            str: Raw stock price data response or error message.
        """
        try:
            async with Client(f"{self.stock_price_url}/sse") as client:
                return await client.call_tool("get_stock_price", {"symbol": symbol})
        except Exception as e:
            return f" Failed to get stock price data for '{symbol}': {str(e)}"

    async def get_company_info(self, symbol: str) -> str:
        """
        Retrieve company information for a given symbol from the Company Info MCP server.

        Args:
            symbol (str): Stock symbol to get company information for.

        Returns:
            str: Company information as plain text or error message.
        """
        try:
            async with Client(f"{self.company_info_url}/sse") as client:
                result = await client.call_tool("get_company_info", {"symbol": symbol})
                return self._extract_text(result)
        except Exception as e:
            return f" Failed to get company information for '{symbol}': {str(e)}"

    async def get_investment_analysis(self, stock_data: str, company_info: str) -> str:
        """
        Get investment analysis and recommendations by calling the LLM MCP server.

        Args:
            stock_data (str): Stock price and trading data.
            company_info (str): Company information and financial metrics.

        Returns:
            str: Investment analysis and recommendations or error message.
        """
        try:
            async with Client(f"{self.llm_server_url}/sse") as client:
                result = await client.call_tool("investment_analysis", {
                    "stock_data": stock_data,
                    "company_info": company_info
                })
                return self._extract_text(result)
        except Exception as e:
            return f" Failed to get investment analysis: {str(e)}"

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
    Main entry point of the MCP based Stock Market Investment Assistant.

    Continuously prompts the user for a stock symbol, fetches stock price and company data,
    and provides investment analysis and recommendations until the user chooses to exit.
    """
    agent = Stock_Market_Advisor("http://localhost:8000" , "http://localhost:8001", "http://localhost:8002")

    print(" Welcome to the Stock Market Investment Assistant!")
    print(" Get real-time stock data, company information, and AI-powered investment analysis")
    print(" Enter stock symbols like AAPL, TSLA, MSFT, GOOGL, AMZN, etc.")
    print(" You can also enter company names like 'Apple', 'Tesla', 'Microsoft'")
    print(" Popular symbols: AAPL (Apple), TSLA (Tesla), MSFT (Microsoft), GOOGL (Google), AMZN (Amazon)")
    
    while True:
        symbol = input("\n  Enter stock symbol or company name to analyze (or 'exit' to quit): ").strip()
        if symbol.lower() == "exit":
            print("  Exiting Stock Market Investment Assistant. Happy investing!")
            break

        if not symbol:
            print(" ❌ Please enter a valid stock symbol or company name.")
            continue

        print(f"\n  Analyzing {symbol.upper()}...")
        try:
            print("  Fetching stock price data...")           
            stock_data_raw = await agent.get_stock_price(symbol)
            stock_data = stock_data_raw[0].text if isinstance(stock_data_raw, list) else str(stock_data_raw)
            
            # Check if we got rate limited or other errors
            if "Rate limit exceeded" in stock_data or "Too Many Requests" in stock_data:
                print(f"\n  Rate limit encountered. Waiting 30 seconds before retrying...")
                print("    Tip: Try again in a minute, or use a different symbol")
                continue
            elif "not found" in stock_data.lower():
                print(f"\n ❌ Stock Price Data for {symbol.upper()}:\n{stock_data}")
                continue
            else:
                print(f"\n  Stock Price Data for {symbol.upper()}:\n{stock_data}")
            
            print("  Fetching company information...")
            company_info = await agent.get_company_info(symbol)
            
            if "Unable to find company information" in company_info:
                print(f"\n ❌ Company Information for {symbol.upper()}:\n{company_info}")
                continue
            else:
                print(f"\n  Company Information for {symbol.upper()}:\n{company_info}")

            print("  Generating AI investment analysis...")
            analysis = await agent.get_investment_analysis(stock_data, company_info)
            print(f"\n  Investment Analysis for {symbol.upper()}:\n{analysis}")
            
        except Exception as e:
            print(f" ❌ An error occurred: {str(e)}")
            print("     Try again with a different symbol or check your internet connection")

if __name__ == "__main__":
    asyncio.run(main())