# MCP AI Agents Collection

A comprehensive collection of Model Context Protocol (MCP) based AI agents and applications optimized for Intel Core Ultra Processors. This repository demonstrates the power and versatility of MCP in building sophisticated AI-powered applications.

##  What is Model Context Protocol (MCP)?

**Model Context Protocol (MCP)** is a revolutionary open standard that enables AI models to securely access and interact with external data sources, tools, and services in real-time. Think of MCP as the "API for AI" - it provides a standardized way for AI applications to connect with the world around them.

###  Key Benefits of MCP:

1. ** Universal Connectivity**: MCP allows AI models to seamlessly integrate with databases, APIs, file systems, and external services
2. ** Security First**: Built-in security mechanisms ensure safe and controlled access to external resources
3. ** Real-time Data**: Enable AI models to access live, up-to-date information rather than relying on static training data
4. ** Modular Architecture**: Build scalable, maintainable AI applications with clear separation of concerns
5. ** Interoperability**: Standardized protocol ensures compatibility across different AI models and platforms

###  Why MCP Matters for AI Development:

- **Beyond Static Knowledge**: Traditional AI models are limited to their training data. MCP enables dynamic, real-time knowledge acquisition
- **Tool Usage**: AI models can interact with external tools, databases, and APIs to perform complex tasks
- **Context Preservation**: Maintains context across multiple interactions and data sources
- **Scalable Integration**: Easily add new data sources and capabilities without rebuilding the entire system
- **Enterprise Ready**: Provides the security and reliability needed for production AI applications

##  Projects in This Collection

###  [MCP-AI-Agent-Stock-Market](./MCP-AI-Agent-Stock-Market/)
**AI-Powered Stock Market Investment Assistant**

A sophisticated financial analysis system that demonstrates MCP's power in real-time data integration:

**Key Features:**
-  Real-time stock prices, trading volumes, and market metrics
-  Comprehensive company profiles and financial insights
-  AI investment analysis using Qwen 2.5 models with OpenVINO optimization
-  Multi-API integration (Yahoo Finance, Alpha Vantage, Financial Modeling Prep)
-  Smart company name to stock symbol conversion
-  Intel hardware acceleration for fast AI inference

###  [MCP-AI-Agent-Weather](./MCP-AI-Agent-Weather/)
**Intelligent Weather Analysis System** *(Coming Soon)*

Advanced weather intelligence platform showcasing MCP's versatility:

**Planned Features:**
-  Real-time weather data and forecasting
-  Geospatial weather analysis
-  AI weather pattern insights
-  Multi-source weather API integration
-  Severe weather monitoring and alerts

##  Technology Stack

- **MCP Framework**: FastMCP for high-performance protocol implementation
- **AI Models**: Qwen 2.5-3B-Instruct with OpenVINO optimization
- **Hardware**: Intel Core Ultra Processors with GPU/NPU acceleration
- **Languages**: Python 3.13+ with async/await patterns
- **APIs**: Real-time data integration from multiple sources

##  Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/morteza89/MCP-AI-Agents.git
   cd MCP-AI-Agents
   ```

2. **Choose a project:**
   ```bash
   cd MCP-AI-Agent-Stock-Market
   .\setup.bat
   ```

3. **Run the application:**
   ```bash
   uv run run_all.py
   ```

##  Contributing

Contributions welcome! Each project demonstrates different MCP patterns and AI capabilities.

##  License

MIT License - See individual project licenses for details.

##  Intel Acceleration

Optimized for Intel Core Ultra Processors with OpenVINO acceleration.
