# MCP Stock Market Investment Assistant

A Model Context Protocol (MCP) based application that provides real-time stock price data, company financial information, and AI-powered investment analysis using Intel® Core™ Ultra Processors.

## Features

🚀 **Real-time Stock Data**: Get current stock prices, trading volumes, market caps, and key metrics  
🏢 **Company Information**: Detailed company profiles, financial metrics, and business insights  
🤖 **AI Investment Analysis**: Intelligent investment recommendations based on technical and fundamental analysis  
💡 **Multi-server Architecture**: Scalable MCP server design for different data sources  
⚡ **Intel Optimized**: Leverages Intel® Core™ Ultra processors for fast AI inference  

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

### 3. Install Dependencies
```powershell
uv add yfinance requests python-dotenv
```

For OpenVINO optimized model: 
```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" add optimum[openvino,intel]
& "$env:USERPROFILE\.local\bin\uv.exe" add openvino-genai huggingface-hub
```

### 4. Set Up API Keys (Optional)

Create a `.env` file in the project directory:
```env
# Optional: For enhanced company data (get free API key from Alpha Vantage)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Optional: For additional financial data (get free API key from Financial Modeling Prep)
FMP_API_KEY=your_fmp_api_key_here
```

**Note**: The system works without API keys using free Yahoo Finance data, but API keys provide more detailed company information.

### 5. Start the System

#### Option A: Easy Start (Recommended)
```powershell
uv run run_all.py
```
This starts all servers automatically and then runs the client.

#### Option B: Manual Start (4 separate terminals)

Open **4 separate terminal windows** and run each command:

##### Terminal 1 - Stock Price Server (Port 8000)
```powershell
uv run 1_stock_price_server.py
```

##### Terminal 2 - Company Info Server (Port 8001)
```powershell
uv run 2_company_info_server.py
```

##### Terminal 3 - LLM Server (Port 8002)

**Option A: Standard Qwen Model (6GB)**
```powershell
uv run 3_LLM_inference_server.py
```

**Option B: OpenVINO Optimized Model (1.5GB)**  
```powershell
uv run 3_LLM_inference_server_OV.py
```

*Note: First run will download the selected AI model. Option A downloads ~6GB Qwen 2.5-3B model, Option B downloads ~1.5GB optimized model.*

##### Terminal 4 - Run the Stock Market Assistant
```powershell
uv run Stock_Market_MCP_Assistant.py
```

## 📊 Usage

1. The client will prompt you for a stock symbol or company name
2. Enter any valid input:
   - **Stock symbols**: AAPL, TSLA, MSFT, GOOGL, AMZN
   - **Company names**: Apple, Tesla, Microsoft, Google, Amazon
3. Get comprehensive investment analysis:
   - 💰 Real-time stock price and trading data
   - 🏢 Detailed company information and financials
   - 🤖 AI-generated investment analysis and recommendations
4. Type `exit` to quit

### Smart Symbol Recognition
The system automatically converts company names to stock symbols:
- "Apple" → AAPL
- "Tesla" → TSLA  
- "Microsoft" → MSFT
- "Google" or "Alphabet" → GOOGL
- And many more!

## 🛠️ Alternative Setup (If uv is not in PATH)

If you encounter `uv : The term 'uv' is not recognized` errors, use the full path commands:

```powershell
# Easy start
& "$env:USERPROFILE\.local\bin\uv.exe" run run_all.py

# Or manual start in separate terminals:
& "$env:USERPROFILE\.local\bin\uv.exe" run 1_stock_price_server.py
& "$env:USERPROFILE\.local\bin\uv.exe" run 2_company_info_server.py
& "$env:USERPROFILE\.local\bin\uv.exe" run 3_LLM_inference_server_OV.py
& "$env:USERPROFILE\.local\bin\uv.exe" run Stock_Market_MCP_Assistant.py
```

## Example Output

```
🚀 Welcome to the Stock Market Investment Assistant!
📈 Get real-time stock data, company information, and AI-powered investment analysis
💡 Enter stock symbols like AAPL, TSLA, MSFT, GOOGL, AMZN, etc.
🏢 You can also enter company names like 'Apple', 'Tesla', 'Microsoft'

📊 Enter stock symbol or company name to analyze (or 'exit' to quit): apple

🔍 Analyzing APPLE...
📈 Fetching stock price data...

💰 Stock Price Data for APPLE:
(Converted 'apple' to symbol 'AAPL')
- Company: Apple Inc. (AAPL)
- Exchange: NASDAQ
- Current Price: $150.25 USD
- Previous Close: $148.50 USD
- Daily Change: 📈 +1.75 (+1.18%)
- Day Range: $149.20 - $151.30 USD
- 52-Week Range: $124.17 - $199.62 USD
- Volume: 45.2M
- Market Cap: $2.35T

🏢 Fetching company information...

📋 Company Information for AAPL:
(Converted 'apple' to symbol 'AAPL')
- Company: Apple Inc. (AAPL)
- Sector: Technology
- Industry: Consumer Electronics
- Market Cap: $2.35T
...

🤖 Generating AI investment analysis...

💡 Investment Analysis for AAPL:
Overall Assessment: BULLISH
Strong fundamentals with consistent revenue growth...
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Stock Price      │    │Company Info     │    │   LLM Server    │
│Server Port 8000 │    │Server Port 8001 │    │   Port 8002     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Stock Market   │
                    │  MCP Client     │
                    └─────────────────┘
```

## 📦 Components

- **Stock Price Server** (`1_stock_price_server.py`): Fetches real-time stock prices using Yahoo Finance API
- **Company Info Server** (`2_company_info_server.py`): Retrieves company financials using multiple APIs  
- **LLM Server** (`3_LLM_inference_server.py`): Provides AI investment analysis using Qwen 2.5-3B model
- **LLM Server (OpenVINO)** (`3_LLM_inference_server_OV.py`): Optimized AI analysis using Intel OpenVINO
- **Client** (`Stock_Market_MCP_Assistant.py`): Orchestrates all servers and provides user interface
- **Easy Runner** (`run_all.py`): Starts all servers automatically for convenience

## 🔧 Troubleshooting

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

### API Rate Limits
- Yahoo Finance API is free but may have rate limits (429 errors)
- The system automatically retries with exponential backoff
- If you hit rate limits, wait 1-2 minutes and try again
- Consider getting free API keys for Alpha Vantage or Financial Modeling Prep for enhanced data
- System gracefully handles API failures and provides helpful suggestions

### Symbol Not Found
- Use official stock symbols (AAPL, not Apple Inc.)
- Try company names - the system will convert them automatically
- Check symbol suggestions when searches fail
- Popular symbols: AAPL, TSLA, MSFT, GOOGL, AMZN, META, NVDA

### String Formatting Errors
- These have been fixed in the latest version
- Make sure you're using the updated servers
- Restart servers if you encounter formatting issues

### uv Not Found
Add to PATH or use full path:
```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run [script_name]
```

Example:
```powershell
# Easy start
& "$env:USERPROFILE\.local\bin\uv.exe" run run_all.py

# Or manual start in separate terminals:
& "$env:USERPROFILE\.local\bin\uv.exe" run 1_stock_price_server.py
& "$env:USERPROFILE\.local\bin\uv.exe" run 2_company_info_server.py
& "$env:USERPROFILE\.local\bin\uv.exe" run 3_LLM_inference_server_OV.py
& "$env:USERPROFILE\.local\bin\uv.exe" run Stock_Market_MCP_Assistant.py
```

## 🚀 Quick Test

Test individual servers using the test client:
```powershell
uv run client.py
```

Test the improvements:
```powershell
uv run test_improvements.py
```

## 🔗 APIs Used

- [Yahoo Finance API](https://finance.yahoo.com/) - Real-time stock price data (free)
- [Alpha Vantage API](https://www.alphavantage.co/) - Company fundamentals (optional, free tier available)
- [Financial Modeling Prep](https://financialmodelingprep.com/) - Financial data (optional, free tier available)
- [Qwen 2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) - AI model for investment analysis

## 📜 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This software is for educational and informational purposes only. It does not constitute financial advice, investment recommendations, or investment research. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.

## ⚡ Intel® Acceleration

This project leverages Intel® Core™ Ultra Processors with PyTorch XPU backend for accelerated AI inference, providing fast and efficient investment analysis.

## Technical Details

### MCP Framework
Built using FastMCP, a high-level Pythonic framework inspired by FastAPI that simplifies Model Context Protocol implementation.

### AI Models
- **Standard**: Qwen 2.5-3B-Instruct (6GB download)
- **OpenVINO Optimized**: Qwen 2.5-1.5B-Instruct-INT8-OV (1.5GB download, faster inference)
- **Fallback**: DistilGPT-2 for compatibility
- **Hardware**: Optimized for Intel® Core™ Ultra Processors with OpenVINO acceleration

### Data Sources
- Real-time stock data from Yahoo Finance
- Company information from multiple financial APIs
- Comprehensive financial analysis (prices, volumes, market caps, ratios)
