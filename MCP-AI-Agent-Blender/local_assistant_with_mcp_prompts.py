import asyncio
import sys
import os
import json

# Fix for Windows asyncio issues
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Launch MCP server (using the server with integrated LLM)
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, ".venv", "Scripts", "python.exe")
server_script = os.path.join(script_dir, "local_mcp_server_with_llm.py")


print(f"Using Python: {venv_python}")
print(f"Using server script: {server_script}")

server_params = StdioServerParameters(
    command=venv_python, 
    args=[server_script],
    env=os.environ.copy()
)

async def get_mcp_prompts(session):
    """Retrieve all available prompts from the MCP server"""
    try:
        prompts = await session.list_prompts()
        print(f"📋 Available MCP prompts: {len(prompts.prompts)}")
        for prompt in prompts.prompts:
            print(f"  - {prompt.name}: {prompt.description}")
        return prompts.prompts
    except Exception as e:
        print(f"⚠️ Could not retrieve MCP prompts: {e}")
        return []

async def get_prompt_content(session, prompt_name: str):
    """Get the content of a specific MCP prompt"""
    try:
        result = await session.get_prompt(prompt_name, arguments={})
        if result.messages:
            return result.messages[0].content.text
        return None
    except Exception as e:
        print(f"⚠️ Could not retrieve prompt '{prompt_name}': {e}")
        return None

async def run_assistant():
    # Connect to the BlenderMCP server via stdio transport
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("✅ Connected to BlenderMCP server.")
            
            # Get available prompts from MCP server
            prompts = await get_mcp_prompts(session)
            
            # Try to get the asset_creation_strategy prompt
            system_prompt = None
            for prompt in prompts:
                if prompt.name == "asset_creation_strategy":
                    print(f"🎯 Found asset_creation_strategy prompt!")
                    strategy_content = await get_prompt_content(session, "asset_creation_strategy")
                    if strategy_content:
                        print("✅ Retrieved asset creation strategy from MCP server")
                        # Build a comprehensive system prompt combining strategy with tool instructions
                        system_prompt = f"""You are an AI assistant controlling Blender via the Model Context Protocol (MCP).
You have access to sophisticated 3D asset creation tools through MCP.

ASSET CREATION STRATEGY:
{strategy_content}

Available MCP tools:
- **create_from_prompt(prompt: str)** – Turn a natural language prompt into Blender Python and execute it.
- **execute_blender_code(code: str)** – Execute arbitrary Python code in Blender.
- **plan_and_execute(goal: str)** – Use LLM to plan and call tools iteratively until goal is met.
- **get_scene_info()** – Get detailed information about current scene.
- **get_polyhaven_status()** – Check if PolyHaven integration is available.
- **get_sketchfab_status()** – Check if Sketchfab integration is available.
- **get_hyper3d_status()** – Check if Hyper3D Rodin integration is available.
- **search_polyhaven_assets(asset_type, categories)** – Search PolyHaven for assets.
- **download_polyhaven_asset(asset_id, asset_type, resolution)** – Download PolyHaven assets.
- **search_sketchfab_models(query, categories, count)** – Search Sketchfab models.
- **download_sketchfab_model(uid)** – Download Sketchfab models.
- **generate_hyper3d_model_via_text(text_prompt, bbox_condition)** – Generate 3D models with AI.

IMPORTANT RULES:
- ALWAYS follow the asset creation strategy above
- Respond with ONLY a JSON object - no markdown, no extra text
- Always use double quotes in JSON
- For complex requests, use plan_and_execute to leverage the full MCP workflow
- For simple requests, use create_from_prompt

Example for "create a house":
{{"type": "plan_and_execute", "params": {{"goal": "Create a detailed house model"}}}}

Example for simple objects:
{{"type": "create_from_prompt", "params": {{"prompt": "Create a red cube"}}}}
"""
                        break
            
            # Fallback to simple prompt if MCP prompt not available
            if not system_prompt:
                print("⚠️ Using fallback system prompt")
                system_prompt = """\
You are an AI assistant controlling Blender via the Model Context Protocol (MCP). 
You have tools to manipulate the Blender scene by sending JSON commands. 

Available MCP tools:
- **create_from_prompt(prompt: str)** – Turn a natural language prompt into Blender Python and execute it.
- **execute_blender_code(code: str)** – Execute arbitrary Python code in Blender.
- **plan_and_execute(goal: str)** – Use LLM to plan and call tools iteratively until goal is met.

IMPORTANT RULES:
- Respond with ONLY a JSON object - no markdown, no extra text
- Always use double quotes in JSON
- For simple requests, use create_from_prompt with a clear description
- For code execution, use execute_blender_code with Python code

Example for "create a red cube":
{"type": "create_from_prompt", "params": {"prompt": "Create a red cube"}}
"""
            
            print("You can now enter instructions.")
            # REPL loop for user instructions
            while True:
                user_input = input("\nUser: ")
                if not user_input:
                    continue
                if user_input.lower() in {"quit", "exit"}:
                    print("Exiting assistant.")
                    break

                # ENHANCED: Use the server's LLM to suggest the best tool for this request
                print(f"🧠 Asking server LLM to suggest best tool for: '{user_input}'")
                
                try:
                    # First, ask the server which tool would be best for this request
                    tool_suggestion_result = await session.call_tool("llm_suggest_tool", arguments={
                        "user_request": user_input
                    })
                    
                    # Extract the tool suggestion
                    if tool_suggestion_result.content:
                        suggestion_text = tool_suggestion_result.content[0].text
                        print(f"💡 Server suggests: {suggestion_text}")
                        
                        try:
                            # Parse the JSON response from llm_suggest_tool
                            suggestion_data = json.loads(suggestion_text)
                            recommended_tool = suggestion_data.get("recommended_tool")
                            confidence = suggestion_data.get("confidence", 0.0)
                            reasoning = suggestion_data.get("reasoning", "")
                            
                            print(f"🎯 Recommended tool: {recommended_tool} (confidence: {confidence:.2f})")
                            print(f"📝 Reasoning: {reasoning}")
                            
                            # Use the suggested tool if confidence is high enough
                            if recommended_tool and confidence > 0.5:
                                # Determine parameters based on the tool type
                                if recommended_tool == "plan_and_execute":
                                    params = {"goal": user_input, "max_steps": 6, "temperature": 0.3}
                                elif recommended_tool == "create_from_prompt":
                                    params = {"prompt": user_input, "temperature": 0.2}
                                elif recommended_tool == "execute_blender_code":
                                    params = {"code": user_input}
                                elif recommended_tool == "get_scene_info":
                                    params = {}
                                elif recommended_tool == "get_viewport_screenshot":
                                    params = {"max_size": 800}
                                elif recommended_tool == "search_polyhaven_assets":
                                    # Extract asset type from user input
                                    asset_type = "textures"  # default
                                    if any(word in user_input.lower() for word in ["hdri", "environment", "sky"]):
                                        asset_type = "hdris"
                                    elif any(word in user_input.lower() for word in ["model", "object", "3d"]):
                                        asset_type = "models"
                                    params = {"asset_type": asset_type, "query": user_input, "limit": 10}
                                elif recommended_tool == "search_sketchfab_models":
                                    params = {"query": user_input, "count": 10}
                                elif recommended_tool == "generate_hyper3d_model_via_text":
                                    params = {"text_prompt": user_input}
                                else:
                                    # Generic parameters for other tools
                                    params = {"prompt": user_input} if "prompt" in user_input else {}
                                
                                tool_type = recommended_tool
                            else:
                                # Fallback to original logic if confidence is low
                                print(f"⚠️ Low confidence ({confidence:.2f}), using fallback logic")
                                if any(keyword in user_input.lower() for keyword in ["house", "building", "structure", "complex"]):
                                    tool_type = "plan_and_execute"
                                    params = {"goal": f"Create a detailed {user_input}"}
                                else:
                                    tool_type = "create_from_prompt" 
                                    params = {"prompt": user_input}
                                    
                        except json.JSONDecodeError:
                            print(f"⚠️ Could not parse tool suggestion, using fallback")
                            # Fallback to original simple logic
                            if any(keyword in user_input.lower() for keyword in ["house", "building", "structure", "complex"]):
                                tool_type = "plan_and_execute"
                                params = {"goal": f"Create a detailed {user_input}"}
                            else:
                                tool_type = "create_from_prompt" 
                                params = {"prompt": user_input}
                    else:
                        print(f"⚠️ No tool suggestion received, using fallback")
                        # Fallback to original simple logic
                        if any(keyword in user_input.lower() for keyword in ["house", "building", "structure", "complex"]):
                            tool_type = "plan_and_execute"
                            params = {"goal": f"Create a detailed {user_input}"}
                        else:
                            tool_type = "create_from_prompt" 
                            params = {"prompt": user_input}
                            
                except Exception as e:
                    print(f"⚠️ Error getting tool suggestion: {e}, using fallback")
                    # Fallback to original simple logic
                    if any(keyword in user_input.lower() for keyword in ["house", "building", "structure", "complex"]):
                        tool_type = "plan_and_execute"
                        params = {"goal": f"Create a detailed {user_input}"}
                    else:
                        tool_type = "create_from_prompt" 
                        params = {"prompt": user_input}
                
                print(f"🔧 Calling tool: {tool_type} with params {params}")
                
                try:
                    # Call the selected tool on the MCP server and get the result
                    result = await session.call_tool(tool_type, arguments=params)
                    
                    # Extract and display the result text
                    result_content = ""
                    if result.content:
                        content_block = result.content[0]
                        if hasattr(content_block, "text"):
                            result_content = content_block.text
                    
                    if not result_content and result.structuredContent:
                        result_content = str(result.structuredContent)
                    
                    if not result_content and hasattr(result, "result"):
                        result_content = str(result.result)

                    if result_content:
                        print(f"🟢 Tool result: {result_content}")
                    else:
                        print("🟢 Tool executed (no text result).")
                        
                except Exception as e:
                    print(f"❌ Tool call failed: {e}")
            # end while
        # end ClientSession
    # end stdio_client

# Run the assistant event loop
asyncio.run(run_assistant())
