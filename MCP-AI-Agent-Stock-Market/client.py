#!/usr/bin/env python3
"""
Simple test client for individual MCP servers.
Use this to test each server individually during development.
"""

import asyncio
from fastmcp import Client

async def test_stock_price_server():
    """Test the stock price server"""
    print("Testing Stock Price Server...")
    try:
        async with Client("http://localhost:8000/sse") as client:
            result = await client.call_tool("get_stock_price", {"symbol": "AAPL"})
            print(f"Stock Price Result: {result}")
    except Exception as e:
        print(f"Error testing stock price server: {e}")

async def test_company_info_server():
    """Test the company info server"""
    print("Testing Company Info Server...")
    try:
        async with Client("http://localhost:8001/sse") as client:
            result = await client.call_tool("get_company_info", {"symbol": "AAPL"})
            print(f"Company Info Result: {result}")
    except Exception as e:
        print(f"Error testing company info server: {e}")

async def test_llm_server():
    """Test the LLM server"""
    print("Testing LLM Server...")
    try:
        async with Client("http://localhost:8002/sse") as client:
            result = await client.call_tool("investment_analysis", {
                "stock_data": "Sample stock data", 
                "company_info": "Sample company info"
            })
            print(f"LLM Result: {result}")
    except Exception as e:
        print(f"Error testing LLM server: {e}")

async def main():
    """Test all servers"""
    print(" Testing MCP Stock Market Servers")
    print("=" * 50)
    
    await test_stock_price_server()
    print()
    await test_company_info_server()
    print()
    await test_llm_server()

if __name__ == "__main__":
    asyncio.run(main())
