import nest_asyncio
nest_asyncio.apply()

import asyncio
import os
from fastmcp import Client

class BlenderMCPAssistant:
    """
    BlenderMCPAssistant is responsible for interacting with Blender MCP server and LLM server.

    It provides methods to:
    - Execute Blender operations (scene manipulation, object creation, material application)
    - Get scene information and viewport screenshots
    - Generate AI-powered insights and code suggestions
    - Download and apply assets from various sources (PolyHaven, Sketchfab, Hyper3D)
    """

    def __init__(self, blender_server_url: str, llm_server_url: str):
        """
        Initialize the BlenderMCPAssistant with URLs for Blender MCP server and LLM server.

        Args:
            blender_server_url (str): URL of the Blender MCP server.
            llm_server_url (str): URL of the LLM MCP server.
        """
        self.blender_server_url = blender_server_url
        self.llm_server_url = llm_server_url

    async def get_scene_info(self) -> str:
        """
        Get detailed information about the current Blender scene.

        Returns:
            str: Scene information or error message.
        """
        try:
            async with Client(f"{self.blender_server_url}/sse") as client:
                result = await client.call_tool("get_scene_info", {})
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to get scene information: {str(e)}"

    async def get_object_info(self, object_name: str) -> str:
        """
        Get detailed information about a specific object in the Blender scene.

        Args:
            object_name (str): Name of the object to inspect.

        Returns:
            str: Object information or error message.
        """
        try:
            async with Client(f"{self.blender_server_url}/sse") as client:
                result = await client.call_tool("get_object_info", {"object_name": object_name})
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to get object information for '{object_name}': {str(e)}"

    async def execute_blender_code(self, code: str) -> str:
        """
        Execute Python code in Blender.

        Args:
            code (str): Python code to execute in Blender.

        Returns:
            str: Execution result or error message.
        """
        try:
            async with Client(f"{self.blender_server_url}/sse") as client:
                result = await client.call_tool("execute_blender_code", {"code": code})
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to execute Blender code: {str(e)}"

    async def get_viewport_screenshot(self, max_size: int = 800) -> str:
        """
        Take a screenshot of the Blender viewport.

        Args:
            max_size (int): Maximum size for the screenshot.

        Returns:
            str: Screenshot result or error message.
        """
        try:
            async with Client(f"{self.blender_server_url}/sse") as client:
                result = await client.call_tool("get_viewport_screenshot", {"max_size": max_size})
                return "Screenshot captured successfully" if result else "Failed to capture screenshot"
        except Exception as e:
            return f"Failed to capture viewport screenshot: {str(e)}"

    async def search_polyhaven_assets(self, query: str, asset_type: str = "hdris", limit: int = 10) -> str:
        """
        Search for assets on PolyHaven.

        Args:
            query (str): Search query.
            asset_type (str): Type of asset (hdris, textures, models).
            limit (int): Maximum number of results.

        Returns:
            str: Search results or error message.
        """
        try:
            async with Client(f"{self.blender_server_url}/sse") as client:
                result = await client.call_tool("search_polyhaven_assets", {
                    "query": query,
                    "asset_type": asset_type,
                    "limit": limit
                })
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to search PolyHaven assets: {str(e)}"

    async def download_polyhaven_asset(self, asset_id: str, asset_type: str = "hdris", resolution: str = "1k") -> str:
        """
        Download an asset from PolyHaven.

        Args:
            asset_id (str): ID of the asset to download.
            asset_type (str): Type of asset.
            resolution (str): Resolution to download.

        Returns:
            str: Download result or error message.
        """
        try:
            async with Client(f"{self.blender_server_url}/sse") as client:
                result = await client.call_tool("download_polyhaven_asset", {
                    "asset_id": asset_id,
                    "asset_type": asset_type,
                    "resolution": resolution
                })
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to download PolyHaven asset: {str(e)}"

    async def search_sketchfab_models(self, query: str, limit: int = 10) -> str:
        """
        Search for 3D models on Sketchfab.

        Args:
            query (str): Search query.
            limit (int): Maximum number of results.

        Returns:
            str: Search results or error message.
        """
        try:
            async with Client(f"{self.blender_server_url}/sse") as client:
                result = await client.call_tool("search_sketchfab_models", {
                    "query": query,
                    "limit": limit
                })
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to search Sketchfab models: {str(e)}"

    async def download_sketchfab_model(self, model_uid: str) -> str:
        """
        Download a 3D model from Sketchfab.

        Args:
            model_uid (str): Unique identifier of the model.

        Returns:
            str: Download result or error message.
        """
        try:
            async with Client(f"{self.blender_server_url}/sse") as client:
                result = await client.call_tool("download_sketchfab_model", {"model_uid": model_uid})
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to download Sketchfab model: {str(e)}"

    async def generate_hyper3d_model_via_text(self, prompt: str) -> str:
        """
        Generate a 3D model using Hyper3D via text prompt.

        Args:
            prompt (str): Text description of the 3D model to generate.

        Returns:
            str: Generation result or error message.
        """
        try:
            async with Client(f"{self.blender_server_url}/sse") as client:
                result = await client.call_tool("generate_hyper3d_model_via_text", {"prompt": prompt})
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to generate Hyper3D model via text: {str(e)}"

    async def analyze_blender_scene(self, scene_info: str) -> str:
        """
        Get AI analysis of the Blender scene using the LLM server.

        Args:
            scene_info (str): Scene information to analyze.

        Returns:
            str: AI analysis or error message.
        """
        try:
            async with Client(f"{self.llm_server_url}/sse") as client:
                result = await client.call_tool("analyze_blender_scene", {"scene_info": scene_info})
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to get scene analysis: {str(e)}"

    async def generate_blender_code(self, task_description: str, object_info: str = "") -> str:
        """
        Generate Python code for Blender tasks using the LLM server.

        Args:
            task_description (str): Description of the task.
            object_info (str): Optional object information.

        Returns:
            str: Generated code or error message.
        """
        try:
            async with Client(f"{self.llm_server_url}/sse") as client:
                result = await client.call_tool("generate_blender_code", {
                    "task_description": task_description,
                    "object_info": object_info
                })
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to generate Blender code: {str(e)}"

    async def suggest_blender_workflow(self, project_goal: str, current_scene: str = "") -> str:
        """
        Get workflow suggestions for Blender projects using the LLM server.

        Args:
            project_goal (str): The project goal.
            current_scene (str): Optional current scene description.

        Returns:
            str: Workflow suggestions or error message.
        """
        try:
            async with Client(f"{self.llm_server_url}/sse") as client:
                result = await client.call_tool("suggest_blender_workflow", {
                    "project_goal": project_goal,
                    "current_scene": current_scene
                })
                return self._extract_text(result)
        except Exception as e:
            return f"Failed to get workflow suggestions: {str(e)}"

    def _extract_text(self, result) -> str:
        """
        Helper method to extract plain text from MCP results.

        Args:
            result (Any): The result returned from an MCP tool call.

        Returns:
            str: Extracted text or string representation.
        """
        try:
            if isinstance(result, list):
                return "\n".join(block.text for block in result if hasattr(block, "text"))
            elif hasattr(result, "text"):
                return result.text
            return str(result)
        except Exception as e:
            return f"Failed to parse result: {str(e)}"

async def main():
    """
    Main entry point of the Blender MCP Assistant.

    Provides an interactive chat interface for controlling Blender through MCP servers.
    """
    assistant = BlenderMCPAssistant("http://localhost:8000", "http://localhost:8003")

    print("🎨 Blender MCP Assistant Started!")
    print("Available commands:")
    print("  - 'scene': Get current scene information")
    print("  - 'screenshot': Take viewport screenshot")
    print("  - 'object <name>': Get information about specific object")
    print("  - 'analyze': Get AI analysis of current scene")
    print("  - 'code <description>': Generate Python code for Blender")
    print("  - 'workflow <goal>': Get workflow suggestions")
    print("  - 'search poly <query>': Search PolyHaven assets")
    print("  - 'search sketch <query>': Search Sketchfab models")
    print("  - 'generate <prompt>': Generate 3D model with Hyper3D")
    print("  - 'execute <code>': Execute Python code in Blender")
    print("  - 'exit': Quit the assistant")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n🎯 Enter command: ").strip()
            
            if user_input.lower() == "exit":
                print("👋 Goodbye! Blender MCP Assistant shutting down.")
                break
            
            if user_input.lower() == "scene":
                print("📋 Getting scene information...")
                result = await assistant.get_scene_info()
                print(f"\n📋 Scene Info:\n{result}")
                
            elif user_input.lower() == "screenshot":
                print("📸 Taking viewport screenshot...")
                result = await assistant.get_viewport_screenshot()
                print(f"\n📸 Screenshot: {result}")
                
            elif user_input.lower().startswith("object "):
                object_name = user_input[7:].strip()
                print(f"🔍 Getting information for object '{object_name}'...")
                result = await assistant.get_object_info(object_name)
                print(f"\n🔍 Object Info:\n{result}")
                
            elif user_input.lower() == "analyze":
                print("🧠 Analyzing current scene...")
                scene_info = await assistant.get_scene_info()
                analysis = await assistant.analyze_blender_scene(scene_info)
                print(f"\n🧠 AI Analysis:\n{analysis}")
                
            elif user_input.lower().startswith("code "):
                description = user_input[5:].strip()
                print(f"⚡ Generating code for: {description}")
                code = await assistant.generate_blender_code(description)
                print(f"\n⚡ Generated Code:\n{code}")
                
            elif user_input.lower().startswith("workflow "):
                goal = user_input[9:].strip()
                print(f"📝 Getting workflow suggestions for: {goal}")
                scene_info = await assistant.get_scene_info()
                workflow = await assistant.suggest_blender_workflow(goal, scene_info)
                print(f"\n📝 Workflow Suggestions:\n{workflow}")
                
            elif user_input.lower().startswith("search poly "):
                query = user_input[12:].strip()
                print(f"🔍 Searching PolyHaven for: {query}")
                results = await assistant.search_polyhaven_assets(query)
                print(f"\n🔍 PolyHaven Results:\n{results}")
                
            elif user_input.lower().startswith("search sketch "):
                query = user_input[14:].strip()
                print(f"🔍 Searching Sketchfab for: {query}")
                results = await assistant.search_sketchfab_models(query)
                print(f"\n🔍 Sketchfab Results:\n{results}")
                
            elif user_input.lower().startswith("generate "):
                prompt = user_input[9:].strip()
                print(f"🎨 Generating 3D model: {prompt}")
                result = await assistant.generate_hyper3d_model_via_text(prompt)
                print(f"\n🎨 Generation Result:\n{result}")
                
            elif user_input.lower().startswith("execute "):
                code = user_input[8:].strip()
                print(f"⚡ Executing code in Blender...")
                result = await assistant.execute_blender_code(code)
                print(f"\n⚡ Execution Result:\n{result}")
                
            else:
                print("❓ Unknown command. Type 'exit' to quit or use one of the available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Blender MCP Assistant shutting down.")
            break
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
