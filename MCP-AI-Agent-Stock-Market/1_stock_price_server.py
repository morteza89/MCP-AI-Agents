# stock_price_server.py

import httpx
from mcp.server.fastmcp import FastMCP
import json
import asyncio
from datetime import datetime, timedelta

# Initialize MCP Server
mcp = FastMCP("stock-price", host="0.0.0.0", port=8000)

# Using Yahoo Finance Alternative API (free)
YAHOO_FINANCE_API = "https://query1.finance.yahoo.com/v8/finance/chart"

# Common company name to symbol mapping
COMPANY_SYMBOL_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "nvidia": "NVDA",
    "intel": "INTC",
    "amd": "AMD",
    "ibm": "IBM",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "paypal": "PYPL",
    "uber": "UBER",
    "airbnb": "ABNB",
    "spotify": "SPOT",
    "zoom": "ZM",
    "slack": "WORK",
    "twitter": "TWTR",
    "snap": "SNAP",
    "pinterest": "PINS",
    "square": "SQ",
    "coinbase": "COIN",
    "robinhood": "HOOD"
}

def suggest_symbol(input_text: str) -> str:
    """
    Convert company name to stock symbol if possible
    """
    input_lower = input_text.lower().strip()
    
    # Direct match
    if input_lower in COMPANY_SYMBOL_MAP:
        return COMPANY_SYMBOL_MAP[input_lower]
    
    # Partial match
    for company, symbol in COMPANY_SYMBOL_MAP.items():
        if input_lower in company or company in input_lower:
            return symbol
    
    # Return original if no match found
    return input_text.upper()

# Stock price tool


@mcp.tool()
async def get_stock_price(symbol: str) -> str:
    """
    Retrieve current stock price and basic financial information for a given stock symbol.

    This function uses the Yahoo Finance API to fetch real-time stock data including:
      - Current stock price
      - Daily change (absolute and percentage)
      - Trading volume
      - Market cap (if available)
      - 52-week high and low
      - Previous close price

    Args:
        symbol (str): The stock symbol or company name to get price data for (e.g., 'AAPL', 'Apple', 'TSLA', 'Tesla').

    Returns:
        str: A formatted string containing the stock information.
             If the symbol is not found or data is unavailable,
             an appropriate error message is returned instead.
    """
    # Clean up the input and try to convert company name to symbol
    original_input = symbol.strip()
    symbol = suggest_symbol(original_input)
    
    # If we converted the input, let the user know
    conversion_note = ""
    if symbol != original_input.upper():
        conversion_note = f"(Converted '{original_input}' to symbol '{symbol}')\n        "
    
    # Yahoo Finance API endpoint
    url = f"{YAHOO_FINANCE_API}/{symbol}"
    
    # Add retry logic for rate limiting
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Add headers to look more like a browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                break  # Success, exit retry loop
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limited
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return f"Rate limit exceeded for symbol '{symbol}'. Please try again in a few minutes."
            else:
                return f"Stock API returned an error: {e.response.status_code} {e.response.text}"
        except httpx.RequestError as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            else:
                return f"Network error while fetching stock data: {str(e)}"
        except Exception as e:
            return f"Unexpected error during stock data fetch: {str(e)}"

    try:
        if "chart" not in data or not data["chart"]["result"]:
            # Suggest similar symbols if available
            suggestions = []
            input_lower = original_input.lower()
            for company, sym in COMPANY_SYMBOL_MAP.items():
                if input_lower in company or company in input_lower:
                    suggestions.append(f"{company.title()} ({sym})")
            
            suggestion_text = ""
            if suggestions:
                suggestion_text = f"\n        💡 Did you mean: {', '.join(suggestions[:3])}?"
            
            return f"Stock symbol '{symbol}' not found or no data available.{suggestion_text}\n        📝 Try using the official stock symbol (e.g., AAPL for Apple, TSLA for Tesla)"
        
        result = data["chart"]["result"][0]
        meta = result["meta"]
        
        # Extract key information
        current_price = meta.get("regularMarketPrice", "N/A")
        previous_close = meta.get("previousClose", "N/A")
        currency = meta.get("currency", "USD")
        exchange = meta.get("exchangeName", "N/A")
        company_name = meta.get("longName", symbol)
        
        # Calculate change if we have both current and previous prices
        if current_price != "N/A" and previous_close != "N/A":
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
            change_indicator = "📈" if change >= 0 else "📉"
        else:
            change = "N/A"
            change_percent = "N/A"
            change_indicator = "📊"
        
        # Get additional data if available
        market_cap = meta.get("marketCap", "N/A")
        volume = meta.get("regularMarketVolume", "N/A")
        day_high = meta.get("regularMarketDayHigh", "N/A")
        day_low = meta.get("regularMarketDayLow", "N/A")
        fifty_two_week_high = meta.get("fiftyTwoWeekHigh", "N/A")
        fifty_two_week_low = meta.get("fiftyTwoWeekLow", "N/A")
        
        # Format market cap for readability
        if market_cap != "N/A" and isinstance(market_cap, (int, float)):
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"
        
        # Format volume for readability
        if volume != "N/A" and isinstance(volume, (int, float)):
            if volume >= 1e6:
                volume_str = f"{volume/1e6:.2f}M"
            elif volume >= 1e3:
                volume_str = f"{volume/1e3:.2f}K"
            else:
                volume_str = f"{volume:,.0f}"
        else:
            volume_str = "N/A"

        # Format the final output with proper conditional formatting
        change_str = f"{change:.2f}" if change != "N/A" else "N/A"
        change_percent_str = f"{change_percent:.2f}" if change_percent != "N/A" else "N/A"

        return f"""
        {conversion_note}- Company: {company_name} ({symbol})
        - Exchange: {exchange}
        - Current Price: ${current_price} {currency}
        - Previous Close: ${previous_close} {currency}
        - Daily Change: {change_indicator} {change_str} ({change_percent_str}%)
        - Day Range: ${day_low} - ${day_high} {currency}
        - 52-Week Range: ${fifty_two_week_low} - ${fifty_two_week_high} {currency}
        - Volume: {volume_str}
        - Market Cap: {market_cap_str}
        """

    except Exception as e:
        return f"Failed to parse stock data: {str(e)}"


if __name__ == "__main__":
    mcp.run("sse")