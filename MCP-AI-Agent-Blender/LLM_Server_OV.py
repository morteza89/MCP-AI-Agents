from fastmcp import FastMCP
from transformers import pipeline
import openvino_genai as ov_genai
import huggingface_hub as hf_hub
import os

# Initialize the MCP LLM Server for Blender
mcp = FastMCP("LLM-Inference-Blender", host="0.0.0.0", port=8003)

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
async def analyze_blender_scene(scene_info: str) -> str:
    """
    Analyze the current Blender scene and provide insights about the 3D objects,
    materials, lighting, and overall composition.

    Args:
        scene_info (str): A detailed scene information report from Blender.

    Returns:
        str: AI-generated analysis and insights about the Blender scene.
    """
    if generator is None:
        return "The language model could not be initialized. Please check your model path and device setup."

    prompt = f"""Analyze the following Blender 3D scene information and provide insights:

Scene Information:
{scene_info}

Please provide:
1. Overview of the scene composition
2. Analysis of objects and their properties
3. Lighting and material assessment
4. Suggestions for improvements or enhancements
5. Potential use cases or applications

Analysis:"""

    try:
        if model_type == "openvino":
            # OpenVINO GenAI API
            result = generator.generate(prompt, max_new_tokens=300, temperature=0.7)
            return result.strip()
        elif model_type == "transformers":
            # Transformers pipeline API
            output = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7, return_full_text=False)
            
            if not output or len(output) == 0:
                return "Failed to analyze scene. The model returned no valid output."

            result = output[0]["generated_text"]
            return result.strip()
        else:
            return "No model available for text generation."

    except Exception as e:
        return f"Model pipeline error: {str(e)}"


@mcp.tool()
async def generate_blender_code(task_description: str, object_info: str = "") -> str:
    """
    Generate Python code for Blender based on a task description and optional object information.

    Args:
        task_description (str): Description of what you want to accomplish in Blender.
        object_info (str): Optional information about existing objects in the scene.

    Returns:
        str: AI-generated Python code for Blender.
    """
    if generator is None:
        return "The language model could not be initialized. Please check your model path and device setup."

    context = f"\nExisting Objects: {object_info}" if object_info else ""
    
    prompt = f"""Generate Python code for Blender to accomplish the following task:

Task: {task_description}{context}

Please provide clean, executable Python code that uses Blender's bpy module. 
Include comments explaining the key steps.

Python Code:
```python
import bpy

# """

    try:
        if model_type == "openvino":
            # OpenVINO GenAI API
            result = generator.generate(prompt, max_new_tokens=400, temperature=0.6)
            return result.strip()
        elif model_type == "transformers":
            # Transformers pipeline API
            output = generator(prompt, max_new_tokens=400, do_sample=True, temperature=0.6, return_full_text=False)
            
            if not output or len(output) == 0:
                return "Failed to generate code. The model returned no valid output."

            result = output[0]["generated_text"]
            return result.strip()
        else:
            return "No model available for code generation."

    except Exception as e:
        return f"Model pipeline error: {str(e)}"


@mcp.tool()
async def suggest_blender_workflow(project_goal: str, current_scene: str = "") -> str:
    """
    Suggest a step-by-step workflow for achieving a specific goal in Blender.

    Args:
        project_goal (str): The overall goal or project you want to accomplish.
        current_scene (str): Optional description of the current scene state.

    Returns:
        str: AI-generated workflow suggestions and best practices.
    """
    if generator is None:
        return "The language model could not be initialized. Please check your model path and device setup."

    scene_context = f"\nCurrent Scene: {current_scene}" if current_scene else ""
    
    prompt = f"""Suggest a detailed workflow for the following Blender project:

Project Goal: {project_goal}{scene_context}

Please provide:
1. Step-by-step workflow
2. Key techniques and tools to use
3. Best practices and tips
4. Potential challenges and solutions
5. Recommended Blender features or addons

Workflow:"""

    try:
        if model_type == "openvino":
            # OpenVINO GenAI API
            result = generator.generate(prompt, max_new_tokens=350, temperature=0.7)
            return result.strip()
        elif model_type == "transformers":
            # Transformers pipeline API
            output = generator(prompt, max_new_tokens=350, do_sample=True, temperature=0.7, return_full_text=False)
            
            if not output or len(output) == 0:
                return "Failed to generate workflow. The model returned no valid output."

            result = output[0]["generated_text"]
            return result.strip()
        else:
            return "No model available for text generation."

    except Exception as e:
        return f"Model pipeline error: {str(e)}"


if __name__ == "__main__":
    mcp.run("sse")
