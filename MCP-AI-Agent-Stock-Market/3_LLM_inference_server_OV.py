from fastmcp import FastMCP
from transformers import pipeline
import openvino_genai as ov_genai
import huggingface_hub as hf_hub
import os

# Initialize the MCP LLM Server
mcp = FastMCP("LLM-Inference", host="0.0.0.0", port=8002)

# Device detection for Intel GPU support
def detect_best_device():
    """Detect the best available device for OpenVINO inference"""
    try:
        import openvino as ov
        core = ov.Core()
        available_devices = core.available_devices
        print(f"Available OpenVINO devices: {available_devices}")
        
        # Prefer GPU (Intel iGPU/dGPU) if available, fallback to CPU
        if any("GPU" in device for device in available_devices):
            return "GPU"
        else:
            return "CPU"
    except Exception as e:
        print(f"Device detection failed: {e}, using CPU")
        return "CPU"

device = detect_best_device()
# device = "NPU"
print(f"Selected device: {device}")
generator = None
model_type = None

try:
    # Try loading OpenVINO optimized Qwen model first
    model_name = "OpenVINO/qwen2.5-1.5b-instruct-int8-ov"
    model_path = "qwen2.5-1.5b-instruct-int8-ov"
    
    print(f"Attempting to load OpenVINO model: {model_name}")
    print("This may take several minutes for first-time download...")
    
    # Download model if not present
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        hf_hub.snapshot_download(model_name, local_dir=model_path)
    
    print(f"Loading OpenVINO GenAI pipeline on {device}...")
    try:
        generator = ov_genai.LLMPipeline(model_path, device)
        print(f"Successfully loaded OpenVINO Qwen model on {device}: {model_name}")
        model_type = "openvino"
    except Exception as device_error:
        if device == "GPU":
            print(f"GPU loading failed: {device_error}")
            print("Falling back to CPU for OpenVINO...")
            generator = ov_genai.LLMPipeline(model_path, "CPU")
            print(f"Successfully loaded OpenVINO Qwen model on CPU: {model_name}")
            model_type = "openvino"
        else:
            raise device_error
    
except Exception as qwen_error:
    print(f"Failed to load OpenVINO Qwen model: {qwen_error}")
    print("Falling back to smaller DistilGPT-2 model...")
    
    try:
        # Fallback to smaller model using transformers
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
        model_type = "transformers"
        
    except Exception as fallback_error:
        print(f"Failed to load fallback model: {fallback_error}")
        generator = None
        model_type = None


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
        if model_type == "openvino":
            # OpenVINO GenAI API
            result = generator.generate(prompt, max_new_tokens=200, temperature=0.7)
            return result.strip()
        elif model_type == "transformers":
            # Transformers pipeline API
            output = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, return_full_text=False)
            
            if not output or len(output) == 0:
                return "Failed to generate investment analysis. The model returned no valid output."

            result = output[0]["generated_text"]
            return result.strip()
        else:
            return "No model available for text generation."

    except Exception as e:
        return f"Model pipeline error: {str(e)}"


if __name__ == "__main__":
    mcp.run("sse")
