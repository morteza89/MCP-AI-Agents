#!/usr/bin/env python3
"""
Quick test for the improved stock market system
Tests symbol conversion and error handling
"""

import asyncio
from fastmcp import Client

async def test_symbol_conversion():
    """Test company name to symbol conversion"""
    print(" Testing Symbol Conversion & Error Handling")
    print("=" * 50)
    
    test_cases = [
        "apple",      # Should convert to AAPL
        "AAPL",       # Should work as-is
        "tesla",      # Should convert to TSLA
        "INVALID123", # Should fail gracefully
        "microsoft"   # Should convert to MSFT
    ]
    
    for test_symbol in test_cases:
        print(f"\n Testing: '{test_symbol}'")
        try:
            async with Client("http://localhost:8000/sse") as client:
                result = await client.call_tool("get_stock_price", {"symbol": test_symbol})
                print(f" Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Small delay to avoid rate limiting
        await asyncio.sleep(1)

if __name__ == "__main__":
    print("Make sure the stock price server is running on port 8000")
    print("Start it with: uv run 1_stock_price_server.py")
    input("Press Enter to continue...")
    asyncio.run(test_symbol_conversion())
