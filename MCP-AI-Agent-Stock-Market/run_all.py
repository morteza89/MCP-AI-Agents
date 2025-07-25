#!/usr/bin/env python3
"""
Quick start script for the Stock Market MCP Assistant
This script starts all servers in the background and then runs the client
"""

import subprocess
import time
import sys
import os
import signal
import asyncio
from pathlib import Path

# Store process PIDs for cleanup
server_processes = []

def cleanup_processes():
    """Clean up all server processes"""
    print("\n🛑 Stopping all servers...")
    for process in server_processes:
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception:
            pass

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    cleanup_processes()
    sys.exit(0)

async def main():
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting Stock Market MCP Assistant")
    print("=" * 50)
    
    # Check if uv is available
    uv_path = Path.home() / ".local" / "bin" / "uv.exe"
    if not uv_path.exists():
        print("❌ uv not found. Please install uv first:")
        print("   powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        return
    
    try:
        # Start servers
        print("📈 Starting Stock Price Server (Port 8000)...")
        server1 = subprocess.Popen([str(uv_path), "run", "1_stock_price_server.py"])
        server_processes.append(server1)
        await asyncio.sleep(2)
        
        print("🏢 Starting Company Info Server (Port 8001)...")
        server2 = subprocess.Popen([str(uv_path), "run", "2_company_info_server.py"])
        server_processes.append(server2)
        await asyncio.sleep(2)
        
        print("🤖 Starting LLM Server (Port 8002)...")
        server3 = subprocess.Popen([str(uv_path), "run", "3_LLM_inference_server_OV.py"])
        server_processes.append(server3)
        
        print("⏳ Waiting for servers to start up...")
        await asyncio.sleep(5)
        
        print("🎯 Starting Stock Market Assistant...")
        print("=" * 50)
        
        # Run the client
        client = subprocess.run([str(uv_path), "run", "Stock_Market_MCP_Assistant.py"])
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cleanup_processes()

if __name__ == "__main__":
    if os.name == 'nt':  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
