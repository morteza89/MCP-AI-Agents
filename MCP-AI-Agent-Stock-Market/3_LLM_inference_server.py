from fastmcp import FastMCP
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Initialize the MCP LLM Server
mcp = FastMCP("LLM-Inference", host="0.0.0.0", port=8002)


try:
    # Try loading Qwen model first
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    print(f"Attempting to load model: {model_name}")
    print("This may take several minutes for first-time download...")
    
    # Load tokenizer and model (will download if not cached)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True  # Required for Qwen models
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,  # Required for Qwen models
        torch_dtype="auto",      # Use appropriate dtype automatically
        device_map="cpu"         # Force CPU to avoid GPU issues
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,      # Reduced for faster processing
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    print(f"Successfully loaded Qwen model: {model_name}")
    
except Exception as qwen_error:
    print(f"Failed to load Qwen model: {qwen_error}")
    print("Falling back to smaller DistilGPT-2 model...")
    
    try:
        # Fallback to smaller model
        model_name = "distilgpt2"
        
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=-1,  # CPU only
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256
        )
        
        print(f"Successfully loaded fallback model: {model_name}")
        
    except Exception as fallback_error:
        print(f"Failed to load fallback model: {fallback_error}")
except Exception as e:
    print(
        f"Failed to load the text generation model: {str(e)} Please download it first using the CLI command( refer README)\n"
    )
    generator = None  # fallback, so your server doesn't crash


@mcp.tool()
async def investment_analysis(stock_data: str, company_info: str) -> str:
    """
    Generate personalized investment analysis and recommendations
    based on the provided stock price data and company information.

    This tool uses a language model to analyze the input data
    and provides investment insights and recommendations.

    Args:
        stock_data (str): Detailed stock price and trading data for the company.
        company_info (str): Company information including financials and business description.

    Returns:
        str: A formatted string containing the AI-generated investment analysis,
        covering stock performance, financial health, investment outlook, and
        recommendations for different investor types.
    """
    if generator is None:
        return "The language model could not be initialized. Please check your model path and device setup."

    prompt = f"""Based on the following stock and company data, provide investment analysis and recommendations:

Stock Price Data:
{stock_data}

Company Information:
{company_info}

Please provide:
1. Overall investment assessment (Bullish/Bearish/Neutral)
2. Key strengths and risks to consider
3. Financial health analysis
4. Short-term and long-term outlook
5. Investment recommendations for different risk profiles (Conservative, Moderate, Aggressive)
6. Potential price targets or key levels to watch

Investment Analysis:"""

    try:
        output = generator(
            prompt, do_sample=True, temperature=0.7, return_full_text=False
        )
        if not output or "generated_text" not in output[0]:
            return "Failed to generate investment analysis. The model returned no valid output."

        result = output[0]["generated_text"]
        return result.strip()

    except Exception as e:
        return f"Model pipeline error: {str(e)}"


if __name__ == "__main__":
    mcp.run("sse")