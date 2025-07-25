@echo off
echo 🚀 Stock Market MCP Assistant Setup
echo ====================================
echo.

echo 📦 Installing dependencies...
"%USERPROFILE%\.local\bin\uv.exe" sync

echo.
echo ✅ Setup complete!
echo.
echo 📊 To run the Stock Market Assistant:
echo.
echo Option A - Easy Start (Recommended):
echo    "%USERPROFILE%\.local\bin\uv.exe" run run_all.py
echo.
echo Option B - Manual Start (4 separate terminals):
echo.
echo    Terminal 1: "%USERPROFILE%\.local\bin\uv.exe" run 1_stock_price_server.py
echo    Terminal 2: "%USERPROFILE%\.local\bin\uv.exe" run 2_company_info_server.py
echo    Terminal 3: "%USERPROFILE%\.local\bin\uv.exe" run 3_LLM_inference_server_OV.py
echo    Terminal 4: "%USERPROFILE%\.local\bin\uv.exe" run Stock_Market_MCP_Assistant.py
echo.
echo 💡 For faster inference, use 3_LLM_inference_server_OV.py (OpenVINO optimized)
echo.
echo 🔍 To test individual servers: "%USERPROFILE%\.local\bin\uv.exe" run client.py
echo.
pause
