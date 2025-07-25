
from fastmcp import FastMCP
import httpx
import os
import asyncio
from dotenv import load_dotenv
import json

# Initialize MCP Server
mcp = FastMCP("Company Financial Info", host="0.0.0.0", port=8001)

# Load API key for financial data
load_dotenv()
FINANCIAL_MODELING_PREP_API_KEY = os.getenv("FMP_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Using free financial APIs
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Common company name to symbol mapping (same as stock server)
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


# Company search function to get company information
async def get_company_overview(symbol: str):
    """
    Fetch company overview and fundamental data using Alpha Vantage API.

    Args:
        symbol (str): The stock symbol to get company information for.

    Returns:
        dict: Company information or None if not found.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return None
        
    url = f"{ALPHA_VANTAGE_BASE_URL}?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            if "Symbol" in data:
                return data
            return None
    except Exception as e:
        print(f"Error fetching company overview: {e}")
        return None


# Alternative function using free data sources (no API key required)
async def get_basic_company_info(symbol: str):
    """
    Get basic company information using free Yahoo Finance API.
    
    Args:
        symbol (str): The stock symbol to get information for.
        
    Returns:
        dict: Basic company information or None if not found.
    """
    symbol = symbol.strip().upper()
    
    # Try multiple Yahoo Finance endpoints
    urls = [
        f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=summaryProfile,financialData,defaultKeyStatistics",
        f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=summaryProfile,financialData,defaultKeyStatistics",
        f"https://query1.finance.yahoo.com/v1/finance/quoteSummary/{symbol}?modules=summaryProfile,financialData,defaultKeyStatistics"
    ]
    
    max_retries = 3
    retry_delay = 2
    
    for url in urls:
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'application/json',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                    response = await client.get(url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "quoteSummary" in data and data["quoteSummary"]["result"]:
                            return data["quoteSummary"]["result"][0]
                    elif response.status_code == 429:  # Rate limited
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            print(f"Rate limit exceeded for company info: {symbol}")
                            break  # Try next URL
                    else:
                        print(f"HTTP {response.status_code} for {url}")
                        break  # Try next URL
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print(f"Error fetching basic company info from {url}: {e}")
                    break  # Try next URL
    
    return None


# Fallback function using the chart API (same as stock price server)
async def get_basic_info_from_chart(symbol: str):
    """
    Get basic company name and info from the chart API as fallback
    """
    symbol = symbol.strip().upper()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = await client.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if "chart" in data and data["chart"]["result"]:
                    meta = data["chart"]["result"][0]["meta"]
                    return {
                        "longName": meta.get("longName", symbol),
                        "currency": meta.get("currency", "USD"),
                        "exchangeName": meta.get("exchangeName", "N/A"),
                        "marketCap": meta.get("marketCap", "N/A"),
                        "regularMarketPrice": meta.get("regularMarketPrice", "N/A")
                    }
    except Exception as e:
        print(f"Error in chart fallback: {e}")
    
    return None


# Company financial information tool


@mcp.tool()
async def get_company_info(symbol: str) -> str:
    """
    Retrieve comprehensive company information and financial metrics
    for a given stock symbol.

    This tool fetches company data including:
    - Company name, description, and sector
    - Market capitalization and financial metrics
    - Key statistics (P/E ratio, dividend yield, etc.)
    - Business summary

    Args:
        symbol (str): The stock symbol or company name to get information for.

    Returns:
        str: A formatted string containing:
            - Company name and description
            - Sector and industry
            - Market cap and financial metrics
            - Key ratios and statistics
            - Business summary

        Returns an error message if:
            - Symbol is not found
            - Network request fails
            - Invalid response is received
    """
    # Convert company name to symbol if needed
    original_input = symbol.strip()
    symbol = suggest_symbol(original_input)
    
    # If we converted the input, let the user know
    conversion_note = ""
    if symbol != original_input.upper():
        conversion_note = f"(Converted '{original_input}' to symbol '{symbol}')\n    "
    
    # Try to get detailed company information
    company_data = None
    
    # First try Alpha Vantage if API key is available
    if ALPHA_VANTAGE_API_KEY:
        company_data = await get_company_overview(symbol)
    
    # Fallback to free Yahoo Finance data
    if not company_data:
        yahoo_data = await get_basic_company_info(symbol)
        if yahoo_data:
            company_data = yahoo_data
    
    # Last resort: try chart API for basic info
    if not company_data:
        chart_data = await get_basic_info_from_chart(symbol)
        if chart_data:
            # Format chart data to look like company data
            company_data = {
                "chart_fallback": True,
                "longName": chart_data.get("longName", symbol),
                "marketCap": chart_data.get("marketCap", "N/A"),
                "currency": chart_data.get("currency", "USD"),
                "exchangeName": chart_data.get("exchangeName", "N/A")
            }

    if not company_data:
        # Suggest similar symbols if available
        suggestions = []
        input_lower = original_input.lower()
        for company, sym in COMPANY_SYMBOL_MAP.items():
            if input_lower in company or company in input_lower:
                suggestions.append(f"{company.title()} ({sym})")
        
        suggestion_text = ""
        if suggestions:
            suggestion_text = f"\n    💡 Did you mean: {', '.join(suggestions[:3])}?"
        
        return f"Unable to find company information for symbol '{symbol}'. Please check if the symbol is correct.{suggestion_text}\n    📝 Try using the official stock symbol (e.g., AAPL for Apple, TSLA for Tesla)"

    try:
        # Extract information from different data sources
        if "chart_fallback" in company_data:  # Chart API fallback
            company_name = company_data.get("longName", symbol)
            description = "Limited company information available from basic data source."
            sector = "N/A"
            industry = "N/A"
            market_cap = company_data.get("marketCap", "N/A")
            pe_ratio = "N/A"
            dividend_yield = "N/A"
            profit_margin = "N/A"
            price_to_book = "N/A"
            revenue_ttm = "N/A"
            gross_profit_ttm = "N/A"
            
            # Format market cap if available
            if market_cap != "N/A" and isinstance(market_cap, (int, float)):
                if market_cap >= 1e12:
                    market_cap = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap = f"${market_cap/1e9:.2f}B"
                elif market_cap >= 1e6:
                    market_cap = f"${market_cap/1e6:.2f}M"
                else:
                    market_cap = f"${market_cap:,.0f}"
            
        elif "Symbol" in company_data:  # Alpha Vantage format
            company_name = company_data.get("Name", symbol)
            description = company_data.get("Description", "N/A")
            sector = company_data.get("Sector", "N/A")
            industry = company_data.get("Industry", "N/A")
            market_cap = company_data.get("MarketCapitalization", "N/A")
            pe_ratio = company_data.get("PERatio", "N/A")
            dividend_yield = company_data.get("DividendYield", "N/A")
            profit_margin = company_data.get("ProfitMargin", "N/A")
            price_to_book = company_data.get("PriceToBookRatio", "N/A")
            revenue_ttm = company_data.get("RevenueTTM", "N/A")
            gross_profit_ttm = company_data.get("GrossProfitTTM", "N/A")
            
            # Format market cap
            if market_cap != "N/A" and market_cap.isdigit():
                market_cap_num = int(market_cap)
                if market_cap_num >= 1e12:
                    market_cap = f"${market_cap_num/1e12:.2f}T"
                elif market_cap_num >= 1e9:
                    market_cap = f"${market_cap_num/1e9:.2f}B"
                elif market_cap_num >= 1e6:
                    market_cap = f"${market_cap_num/1e6:.2f}M"
                else:
                    market_cap = f"${market_cap_num:,}"
            
        else:  # Yahoo Finance format
            summary_profile = company_data.get("summaryProfile", {})
            financial_data = company_data.get("financialData", {})
            key_stats = company_data.get("defaultKeyStatistics", {})
            
            company_name = summary_profile.get("longName", symbol)
            description = summary_profile.get("longBusinessSummary", "N/A")
            sector = summary_profile.get("sector", "N/A")
            industry = summary_profile.get("industry", "N/A")
            
            # Financial metrics
            market_cap = key_stats.get("marketCap", {}).get("raw", "N/A")
            pe_ratio = key_stats.get("trailingPE", {}).get("raw", "N/A")
            dividend_yield = key_stats.get("dividendYield", {}).get("raw", "N/A")
            profit_margin = financial_data.get("profitMargins", {}).get("raw", "N/A")
            price_to_book = key_stats.get("priceToBook", {}).get("raw", "N/A")
            revenue_ttm = financial_data.get("totalRevenue", {}).get("raw", "N/A")
            gross_profit_ttm = financial_data.get("grossProfits", {}).get("raw", "N/A")
            
            # Format numbers
            if isinstance(market_cap, (int, float)) and market_cap != "N/A":
                if market_cap >= 1e12:
                    market_cap = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap = f"${market_cap/1e9:.2f}B"
                elif market_cap >= 1e6:
                    market_cap = f"${market_cap/1e6:.2f}M"
                else:
                    market_cap = f"${market_cap:,.0f}"
            
            # Format percentages
            if isinstance(dividend_yield, (int, float)) and dividend_yield != "N/A":
                dividend_yield = f"{dividend_yield*100:.2f}%"
            
            if isinstance(profit_margin, (int, float)) and profit_margin != "N/A":
                profit_margin = f"{profit_margin*100:.2f}%"
        
        # Truncate description if too long
        if description != "N/A" and len(description) > 500:
            description = description[:500] + "..."

        return f"""
    {conversion_note}- Company: {company_name} ({symbol})
    - Sector: {sector}
    - Industry: {industry}
    - Market Cap: {market_cap}
    
    Financial Metrics:
    - P/E Ratio: {pe_ratio}
    - Price-to-Book: {price_to_book}
    - Dividend Yield: {dividend_yield}
    - Profit Margin: {profit_margin}
    - Revenue (TTM): {revenue_ttm}
    - Gross Profit (TTM): {gross_profit_ttm}
    
    Business Description:
    {description}
    """

    except Exception as e:
        return f"Failed to parse company information: {str(e)}"


if __name__ == "__main__":
    mcp.run("sse")