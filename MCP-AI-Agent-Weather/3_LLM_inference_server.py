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
async def safety_guidelines(weather_report: str, aqi_report: str) -> str:
    """
    Generate personalized outdoor safety and health guidelines
    based on the provided weather and air quality reports.

    This tool uses a language model to analyze the input data
    and provides an overall outdoor safety level.

    Args:
        weather_report (str): A detailed weather report for the location.
        aqi_report (str): The Air Quality Index (AQI) report for the same location.

    Returns:
        str: A formatted string containing the AI-generated safety advice,
        covering outdoor safety, health risks, precautions, and
        recommendations for sensitive groups.
    """
    if generator is None:
        return "The language model could not be initialized. Please check your model path and device setup."

    prompt = f"""Based on the following weather and air quality data, provide health and safety recommendations:

Weather Report:
{weather_report}

Air Quality Report:
{aqi_report}

Please provide:
1. Overall outdoor safety assessment
2. Health risks to be aware of  
3. Recommended precautions
4. Special advice for sensitive groups (children, elderly, people with respiratory conditions)

Recommendations:"""

    try:
        output = generator(
            prompt, do_sample=True, temperature=0.7, return_full_text=False
        )
        if not output or "generated_text" not in output[0]:
            return "Failed to generate safety guidelines. The model returned no valid output."

        result = output[0]["generated_text"]
        return result.strip()

    except Exception as e:
        return f"Model pipeline error: {str(e)}"
    except Exception as e:
        return f"Unexpected error during text generation: {str(e)}"


if __name__ == "__main__":
    mcp.run("sse")