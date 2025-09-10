# local_mcp_server_with_llm.py
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
import tempfile
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List
import os
from pathlib import Path
import base64
from urllib.parse import urlparse
import re
import threading
import time
from threading import Lock

# Logger setup
logger = logging.getLogger("BlenderMCPServer")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Optional OpenVINO import
try:
    import openvino_genai as ov_genai
    OPENVINO_AVAILABLE = True
except Exception:
    ov_genai = None
    OPENVINO_AVAILABLE = False


class BlenderConnection:
    """Persistent socket connection to the Blender addon JSON server."""

    def __init__(self, host: str = "localhost", port: int = 9876, timeout: float = 15.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.lock: Lock | None = None

    def connect(self) -> bool:
        try:
            if self.lock is None:
                self.lock = Lock()
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self.sock.settimeout(self.timeout)
            logger.info(f"Connected to Blender at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Blender: {e}")
            self.sock = None
            return False

    def disconnect(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def receive_full_response(self, sock: socket.socket) -> bytes:
        """Read chunks until a complete JSON object can be parsed or timeout occurs."""
        chunks: list[bytes] = []
        start = time.time()
        while True:
            if time.time() - start > self.timeout:
                raise socket.timeout()
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    # Connection closed by peer
                    break
                chunks.append(chunk)

                # Try to parse full buffer to detect completion
                try:
                    data = b"".join(chunks)
                    json.loads(data.decode("utf-8"))
                    logger.info(f"Received complete response ({len(data)} bytes)")
                    return data
                except json.JSONDecodeError:
                    # Not complete yet; continue accumulating
                    continue
            except socket.timeout:
                logger.warning("Socket timeout during chunked receive")
                break
            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Socket connection error during receive: {str(e)}")
                raise

        # Fallback: try to parse what we have
        if chunks:
            data = b"".join(chunks)
            logger.info(f"Returning data after receive completion ({len(data)} bytes)")
            try:
                json.loads(data.decode("utf-8"))
                return data
            except json.JSONDecodeError:
                raise Exception("Incomplete JSON response received")
        else:
            raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Blender and return the response"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Blender")
        
        command = {
            "type": command_type,
            "params": params or {}
        }
        
        try:
            # Log the command being sent
            logger.info(f"Sending command: {command_type} with params: {params}")
            # Send/receive under a lock so concurrent tool calls don't interleave
            if self.lock is None:
                self.lock = Lock()
            with self.lock:
                # Send the command
                self.sock.sendall(json.dumps(command).encode('utf-8'))
                logger.info(f"Command sent, waiting for response...")
                # Set a timeout for receiving - use configured timeout
                self.sock.settimeout(self.timeout)
                # Receive the response using the improved receive_full_response method
                response_data = self.receive_full_response(self.sock)
            logger.info(f"Received {len(response_data)} bytes of data")
            
            response = json.loads(response_data.decode('utf-8'))
            logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")
            
            if response.get("status") == "error":
                logger.error(f"Blender error: {response.get('message')}")
                raise Exception(response.get("message", "Unknown error from Blender"))
            
            return response.get("result", {})
        except socket.timeout:
            logger.error("Socket timeout while waiting for response from Blender")
            # Don't try to reconnect here - let the get_blender_connection handle reconnection
            # Just invalidate the current socket so it will be recreated next time
            self.sock = None
            raise Exception("Timeout waiting for Blender response - try simplifying your request")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.sock = None
            raise Exception(f"Connection to Blender lost: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Blender: {str(e)}")
            # Try to log what was received
            if 'response_data' in locals() and response_data:
                logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
            raise Exception(f"Invalid response from Blender: {str(e)}")
        except Exception as e:
            logger.error(f"Error communicating with Blender: {str(e)}")
            # Don't try to reconnect here - let the get_blender_connection handle reconnection
            self.sock = None
            raise Exception(f"Communication error with Blender: {str(e)}")

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    # We don't need to create a connection here since we're using the global connection
    # for resources and tools
    try:
        logger.info(" LocalLLM-Enhanced BlenderMCP server starting up")
        
        # Initialize local LLM
        llm_manager = get_llm_manager()
        if llm_manager.is_available():
            logger.info(" Local LLM integrated successfully")
        else:
            logger.warning("  Running without local LLM assistance")
        
        # Try to connect to Blender
        try:
            blender = get_blender_connection()
            logger.info("Successfully connected to Blender on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Blender on startup: {str(e)}")
            logger.warning("Make sure the Blender addon is running before using Blender resources or tools")
        
        # Return an empty context - we're using the global connection
        yield {}
    finally:
        # Clean up the global connection on shutdown
        global _blender_connection, _llm_manager
        if _blender_connection:
            logger.info("Disconnecting from Blender on shutdown")
            _blender_connection.disconnect()
            _blender_connection = None
        logger.info("BlenderMCP server shut down")

# Create the MCP server with local LLM integration
mcp = FastMCP(
    "BlenderMCP",
    lifespan=server_lifespan
)


class LocalLLMManager:
    """Manages the local LLM for intelligent MCP tool usage"""
    
    def __init__(self):
        self.pipe = None
        self.model_path = os.getenv("OV_GENAI_MODEL", "qwen2.5-1.5b-instruct-int8-ov")
        self.device = os.getenv("OV_DEVICE", "GPU")  # Default GPU on AI PC; CPU as fallback
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the OpenVINO LLM pipeline"""
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO not available - MCP server will run without LLM assistance")
            return
        
        try:
            logger.info(f" Initializing Local LLM for MCP server: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model path not found: {self.model_path}")
                return
            
            # Initialize OpenVINO pipeline with GPU acceleration
            self.pipe = ov_genai.LLMPipeline(self.model_path, self.device)
            logger.info(f" Local LLM successfully loaded on {self.device}")
            
            # Test the pipeline
            test_response = self.pipe.generate("Hello", max_new_tokens=10)
            logger.info(f" LLM test successful: {test_response[:50]}...")
            
        except Exception as e:
            logger.error(f" Failed to initialize local LLM: {e}")
            self.pipe = None
    
    def is_available(self) -> bool:
        """Check if local LLM is available"""
        return self.pipe is not None
    
    def enhance_tool_usage(self, user_request: str, available_tools: List[str]) -> Dict[str, Any]:
        """Use local LLM to suggest optimal tool usage for a user request"""
        if not self.is_available():
            return {"suggestion": "No LLM assistance available", "confidence": 0.0}
        
        try:
            # Create a prompt for tool selection
            prompt = f"""You are an expert assistant for Blender 3D creation using MCP tools.

Available MCP tools: {', '.join(available_tools)}

User request: "{user_request}"

Based on the user request, suggest the most appropriate single MCP tool to use.
Respond with the tool name only, no explanation.

Tool suggestion:"""

            response = self.pipe.generate(
                prompt,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True
            )
            
            # Print LLM output for debugging
            print(f"\n=== LLM TOOL SUGGESTION DEBUG ===")
            print(f"User request: {user_request}")
            print(f"Available tools: {available_tools}")
            print(f"LLM response: {response}")
            print(f"================================\n")
            
            # Extract tool suggestion
            suggestion = response.strip().split('\n')[0].strip()
            
            # Simple confidence scoring based on keyword matching
            confidence = 0.8 if any(tool in suggestion.lower() for tool in available_tools) else 0.3
            
            print(f"Parsed suggestion: {suggestion}")
            print(f"Confidence: {confidence}")
            
            return {
                "suggestion": suggestion,
                "confidence": confidence,
                "full_response": response
            }
            
        except Exception as e:
            logger.error(f"Error in LLM tool enhancement: {e}")
            return {"suggestion": "Error in LLM processing", "confidence": 0.0}

# Global instances
_blender_connection = None
_polyhaven_enabled = False  # Add this global variable
_llm_manager = None  # Add missing global for LLM manager

def get_blender_connection():
    """Get or create a persistent Blender connection"""
    global _blender_connection, _polyhaven_enabled  # Add _polyhaven_enabled to globals
    host = os.getenv("BLENDER_HOST", "localhost")
    port = int(os.getenv("BLENDER_PORT", "9876"))
    timeout = float(os.getenv("BLENDER_TIMEOUT", "15"))
    
    # If we have an existing connection, check if it's still valid
    if _blender_connection is not None:
        try:
            # First check if PolyHaven is enabled by sending a ping command
            result = _blender_connection.send_command("get_polyhaven_status")
            # Store the PolyHaven status globally
            _polyhaven_enabled = result.get("enabled", False)
            return _blender_connection
        except Exception as e:
            # Connection is dead, close it and create a new one
            logger.warning(f"Existing connection is no longer valid: {str(e)}")
            try:
                _blender_connection.disconnect()
            except:
                pass
            _blender_connection = None
    
    # Create a new connection if needed
    if _blender_connection is None:
        _blender_connection = BlenderConnection(host=host, port=port, timeout=timeout)
        if not _blender_connection.connect():
            logger.error("Failed to connect to Blender")
            _blender_connection = None
            raise Exception("Could not connect to Blender. Make sure the Blender addon is running.")
        logger.info("Created new persistent connection to Blender")
    
    return _blender_connection

def get_llm_manager():
    """Get or create the LLM manager"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LocalLLMManager()
    return _llm_manager


# -----------------------------------------------------------------------------
# Prompt→Code helpers and tool: create_from_prompt
# -----------------------------------------------------------------------------

def _extract_code_block(text: str) -> str:
    """Extract Python code from an LLM response. Supports fenced blocks; fallback to raw text."""
    if not text:
        return ""
    fence = "```"
    i = text.find(fence)
    if i != -1:
        j = text.find("\n", i + len(fence))
        # handle ```python
        if j != -1 and text[i:j].lower().startswith("```python"):
            start = j + 1
        else:
            start = i + len(fence)
        end = text.find(fence, start)
        if end != -1:
            return text[start:end].strip()
    return text.strip()

def _wrap_blender_code(code: str, clear_scene: bool = False, intent_house: bool = False) -> str:
    """Wrap generated code with safe imports, helpers, and scene update; avoid bpy.ops usage.
    intent_house: if True, auto-build a simple house when too few objects were created.
    """
    header = [
        "import bpy, bmesh",
        "# Track created objects",
        "_mcp_created = []",
        f"_mcp_intent_house = {str(bool(intent_house))}",
        "\n# Helper to create a cube via data API (no operators)",
        "def _mcp_create_cube(name, size=1.0, location=(0,0,0), scale=(1,1,1), *args, **kwargs):",
        "    mesh = bpy.data.meshes.new(name + '_mesh')",
        "    bm = bmesh.new()",
        "    bmesh.ops.create_cube(bm, size=size)",
        "    bm.to_mesh(mesh)",
        "    bm.free()",
        "    obj = bpy.data.objects.new(name, mesh)",
        "    obj.location = location",
        "    obj.scale = scale",
        "    bpy.context.scene.collection.objects.link(obj)",
        "    _mcp_created.append(obj)",
        "    return obj",
    "\n# Helper to create a box with non-uniform dimensions",
        "def _mcp_create_box(name, size=(1,1,1), location=(0,0,0), scale=None, *args, **kwargs):",
        "    sx, sy, sz = size if isinstance(size, (list, tuple)) and len(size)==3 else (size, size, size)",
        "    if scale is not None:",
        "        if isinstance(scale, (list, tuple)) and len(scale)==3:",
        "            sx, sy, sz = sx*scale[0], sy*scale[1], sz*scale[2]",
        "        elif isinstance(scale, (int, float)):",
        "            sx, sy, sz = sx*scale, sy*scale, sz*scale",
    "    return _mcp_create_cube(name, 1.0, location, (sx, sy, sz))",
    "\n# Convenience wrapper",
    "def _mcp_add_box(name, size=(1,1,1), location=(0,0,0)):",
    "    return _mcp_create_box(name, size=size, location=location)",
    "\n# Simple house builder using boxes (base, 4 walls, roof)",
    "def _mcp_build_simple_house(name='House', base_size=(4.0, 4.0), wall_height=2.5, wall_thickness=0.1, roof_type='flat', origin=(0,0,0)):",
    "    bx, by = base_size",
    "    # Base",
    "    base = _mcp_add_box(name + '_Base', (bx, by, 0.2), (origin[0], origin[1], origin[2] + 0.1))",
    "    # Walls (as thin boxes)",
    "    zc = origin[2] + 0.1 + wall_height/2",
    "    # +X wall",
    "    _mcp_add_box(name + '_Wall_E', (wall_thickness, by, wall_height), (origin[0] + bx/2, origin[1], zc))",
    "    # -X wall",
    "    _mcp_add_box(name + '_Wall_W', (wall_thickness, by, wall_height), (origin[0] - bx/2, origin[1], zc))",
    "    # +Y wall",
    "    _mcp_add_box(name + '_Wall_N', (bx, wall_thickness, wall_height), (origin[0], origin[1] + by/2, zc))",
    "    # -Y wall",
    "    _mcp_add_box(name + '_Wall_S', (bx, wall_thickness, wall_height), (origin[0], origin[1] - by/2, zc))",
    "    # Roof",
    "    rz = origin[2] + wall_height + 0.15",
    "    if str(roof_type).lower() in ('gabled', 'gable', 'tri', 'triangle'):",
    "        # Simple gabled effect: two thin boxes slightly tilted",
    "        r1 = _mcp_add_box(name + '_Roof_A', (bx, by*0.55, 0.1), (origin[0], origin[1] + by*0.225, rz))",
    "        r2 = _mcp_add_box(name + '_Roof_B', (bx, by*0.55, 0.1), (origin[0], origin[1] - by*0.225, rz))",
    "        try:",
    "            r1.rotation_euler[0] = 0.35",
    "            r2.rotation_euler[0] = -0.35",
    "        except Exception:",
    "            pass",
    "    else:",
    "        _mcp_add_box(name + '_Roof', (bx*1.05, by*1.05, 0.08), (origin[0], origin[1], rz))",
    "    return base",
    "\n# Create a dome-like roof using an icosphere flattened on Z",
    "def _mcp_add_dome_roof(base, bx, by, rz):",
    "    import math",
    "    # Create an icosphere and flatten it",
    "    mesh = bpy.data.meshes.new(base + '_Roof_Dome_mesh')",
    "    bm = bmesh.new()",
    "    try:",
    "        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)",
    "    except Exception:",
    "        # Fallback: cube as placeholder",
    "        bmesh.ops.create_cube(bm, size=1.0)",
    "    bm.to_mesh(mesh)",
    "    bm.free()",
    "    obj = bpy.data.objects.new(base + '_Roof_Dome', mesh)",
    "    bpy.context.scene.collection.objects.link(obj)",
    "    obj.location = (0, 0, rz)",
    "    obj.scale = (bx*0.55, by*0.55, 0.35)",
    "    _mcp_created.append(obj)",
    "    return obj",
    "\n# Guess existing house prefix from objects (based on *_Roof* or *_Wall_* names)",
    "def _mcp_guess_house_prefix():",
    "    scene = bpy.context.scene",
    "    for obj in scene.objects:",
    "        n = obj.name",
    "        if '_Roof' in n:",
    "            return n.split('_Roof')[0]",
    "        if '_Wall_' in n:",
    "            return n.split('_Wall_')[0]",
    "        if n.endswith('_Base'):",
    "            return n[:-5]",
    "    return 'House'",
    "\n# Delete objects safely by name contains",
    "def _mcp_delete_objects_with(sub):",
    "    scene = bpy.context.scene",
    "    to_del = [o for o in scene.objects if sub in o.name]",
    "    for o in to_del:",
    "        try:",
    "            bpy.data.objects.remove(o, do_unlink=True)",
    "        except Exception:",
    "            pass",
    "\n# Modify roof shape for the first detected house",
    "def _mcp_modify_roof(shape='gabled'):",
    "    base = _mcp_guess_house_prefix()",
    "    # Infer base size from walls if available",
    "    bx = by = 3.0",
    "    wall_h = 2.0",
    "    for o in bpy.context.scene.objects:",
    "        if o.name == base + '_Wall_E':",
    "            # scale encodes half-size; approximate dims from scale if possible",
    "            try:",
    "                by = abs(o.scale[1]) if hasattr(o, 'scale') else by",
    "                wall_h = abs(o.scale[2]) if hasattr(o, 'scale') else wall_h",
    "            except Exception:",
    "                pass",
    "        if o.name == base + '_Wall_N':",
    "            try:",
    "                bx = abs(o.scale[0]) if hasattr(o, 'scale') else bx",
    "            except Exception:",
    "                pass",
    "    # Remove existing roof parts",
    "    _mcp_delete_objects_with(base + '_Roof')",
    "    rz = 0.15 + wall_h",
    "    shape_l = str(shape).lower()",
    "    if shape_l in ('gabled', 'gable', 'tri', 'triangle'):",
    "        r1 = _mcp_add_box(base + '_Roof_A', (bx, by*0.55, 0.1), (0, by*0.225, rz))",
    "        r2 = _mcp_add_box(base + '_Roof_B', (bx, by*0.55, 0.1), (0, -by*0.225, rz))",
    "        try:",
    "            r1.rotation_euler[0] = 0.35",
    "            r2.rotation_euler[0] = -0.35",
    "        except Exception:",
    "            pass",
    "    elif shape_l in ('hip', 'pyramid', 'pyramidal'):",
    "        # Approximate hip roof with two perpendicular sloped planes",
    "        r1 = _mcp_add_box(base + '_Roof_X', (bx, by*0.55, 0.1), (0, 0, rz))",
    "        r2 = _mcp_add_box(base + '_Roof_Y', (by, bx*0.55, 0.1), (0, 0, rz))",
    "        try:",
    "            r1.rotation_euler[0] = 0.35",
    "            r2.rotation_euler[1] = 0.35",
    "        except Exception:",
    "            pass",
    "    elif shape_l in ('dome', 'curved', 'rounded', 'arch', 'arched'):",
    "        _mcp_add_dome_roof(base, bx, by, rz)",
    "    else:",
    "        _mcp_add_box(base + '_Roof', (bx*1.05, by*1.05, 0.08), (0, 0, rz))",
        "\n# Aliases",
        "def create_cube(*args, **kwargs): return _mcp_create_cube(*args, **kwargs)",
        "def create_box(*args, **kwargs): return _mcp_create_box(*args, **kwargs)",
        "\n# Safe .name printer",
        "def _mcp_try_print_name(x):\n    try:\n        print(x.name)\n    except Exception:\n        pass",
    ]
    if clear_scene:
        header.append("\n# Optional clear scene")
        header.append("for obj in list(bpy.context.scene.objects):\n    bpy.data.objects.remove(obj, do_unlink=True)")
    footer = [
        "\n# Ensure depsgraph updates",
        "bpy.context.view_layer.update()",
        "print('MCP: Prompt executed. Scene objects:', len(bpy.context.scene.objects))",
    ]
    safe_body = [
        "try:",
        "    def _mcp_user_code():",
    ] + ["        " + ln for ln in code.splitlines()] + [
        "    _mcp_user_code()",
        "except Exception as _mcp_e:",
        "    print('MCP: Error in prompt code:', _mcp_e)",
    ]
    ensure = [
        "\n# Ensure at least one object",
        "if not _mcp_created:",
        "    print('MCP: No objects created by prompt; adding a default cube')",
        "    _mcp_create_cube('PromptCube', 1.0, (0,0,0.5))",
        "# If house intent and too few objects, auto-build a simple house",
        "if _mcp_intent_house and len(_mcp_created) < 5:",
        "    print('MCP: House intent detected; building a simple house')",
        "    _mcp_build_simple_house('AutoHouse', base_size=(3.0, 2.5), wall_height=2.0, wall_thickness=0.12, roof_type='gabled', origin=(0,0,0))",
    ]
    parts = ["\n".join(header), "\n".join(safe_body), "\n".join(ensure), "\n".join(footer)]
    return "\n\n".join(parts)

def _sanitize_generated_code(code: str) -> str:
    """Strip fences, avoid MCP-side calls, comment out bpy.ops, and normalize helper calls."""
    if not code:
        return code
    code = code.replace("```python", "").replace("```", "")
    lines = code.splitlines()
    sanitized: list[str] = [] 
    blacklist = [
        re.compile(r"\bget_scene_info\("),
        re.compile(r"\bget_polyhaven_status\("),
        re.compile(r"\bget_sketchfab_status\("),
        re.compile(r"\bsearch_sketchfab_models\("),
        re.compile(r"\bdownload_sketchfab_model\("),
        re.compile(r"\bgenerate_hyper3d_model_via_\w+\("),
        re.compile(r"\bpoll_rodin_job_status\("),
        re.compile(r"\bimport_generated_asset\("),
    ]
    banned_keywords = (
        "bpy.context.scene.get(",
        "polyhaven", "sketchfab", "hyper3d", "rodin", "get_scene_info",
    )
    ops_pat = re.compile(r"\bbpy\.ops\.")
    skip_block = False
    skip_indent = 0
    for line in lines:
        if skip_block:
            leading = len(line) - len(line.lstrip(" \t"))
            if line.strip() == "":
                continue
            if leading <= skip_indent:
                skip_block = False
            else:
                continue
        # Drop any user redefinitions of helpers to avoid shadowing
        if (re.match(r"^\s*def\s+create_cube\s*\(", line)
            or re.match(r"^\s*def\s+create_box\s*\(", line)
            or re.match(r"^\s*def\s+_mcp_create_cube\s*\(", line)
            or re.match(r"^\s*def\s+_mcp_create_box\s*\(", line)
            or re.match(r"^\s*def\s+_mcp_add_box\s*\(", line)
            or re.match(r"^\s*def\s+_mcp_build_simple_house\s*\(", line)):
            skip_block = True
            skip_indent = len(line) - len(line.lstrip(" \t"))
            continue
        if re.match(r"^\s*_mcp_create_cube\s*=", line) or re.match(r"^\s*_mcp_create_box\s*=", line):
            continue
        if any(p.search(line) for p in blacklist):
            continue
        if ops_pat.search(line):
            sanitized.append("# [sanitized: avoid bpy.ops] " + line)
            continue
        if line.strip().startswith("```"):
            continue
        fixed = re.sub(r"\bcreate_cube\s*\(", "_mcp_create_cube(", line)
        fixed = re.sub(r"\bcreate_box\s*\(", "_mcp_create_box(", fixed)
        fixed = re.sub(r"print\(\s*([A-Za-z_]\w*)\.name\s*\)", r"_mcp_try_print_name(\1)", fixed)
        sanitized.append(fixed)

    # Repair orphan branches and obviously incomplete lines
    repaired: list[str] = []
    ctrl_stack: list[tuple[int, str]] = []  # (indent, tag)
    for line in sanitized:
        raw = line
        stripped = raw.lstrip(" \t")
        indent = len(raw) - len(stripped)
        # Pop stack for dedent
        while ctrl_stack and indent <= ctrl_stack[-1][0]:
            ctrl_stack.pop()
        # Orphan handling
        if re.match(r"^else\s*:\s*$", stripped):
            if not (ctrl_stack and ctrl_stack[-1][1] in ("if", "elif", "for", "while", "try") and ctrl_stack[-1][0] == indent):
                raw = (" " * indent) + "if True:"
        elif re.match(r"^elif\b.*:\s*$", stripped):
            if not (ctrl_stack and ctrl_stack[-1][1] in ("if", "elif") and ctrl_stack[-1][0] == indent):
                raw = (" " * indent) + re.sub(r"^elif\b", "if", stripped)
        elif re.match(r"^except\b.*:\s*$", stripped) or re.match(r"^finally\s*:\s*$", stripped):
            if not (ctrl_stack and ctrl_stack[-1][1] == "try" and ctrl_stack[-1][0] == indent):
                raw = (" " * indent) + "if False:"
        # Track starters
        stripped2 = raw.lstrip(" \t")
        if re.match(r"^(if|elif|for|while|try)\b.*:\s*$", stripped2):
            ctrl_tag = stripped2.split(":",1)[0].split()[0]
            ctrl_stack.append((indent, ctrl_tag))
        # Comment obviously incomplete paren lines
        if raw.count("(") > raw.count(")") and not stripped2.endswith(":"):
            raw = ("# [sanitized: possibly incomplete] " + raw)
        repaired.append(raw)

    sanitized = repaired
    # Ensure control blocks have a body (handles def/if/elif/else/for/while/try/except/with/class)
    out: list[str] = []
    i = 0
    block_re = re.compile(r"^(\s*)(def|if|elif|else|for|while|try|except|with|class)\b.*:\s*$")
    while i < len(sanitized):
        line = sanitized[i]
        m = block_re.match(line)
        if not m:
            out.append(line)
            i += 1
            continue
        indent = m.group(1)
        out.append(line)
        j = i + 1
        found_body = False
        while j < len(sanitized):
            l2 = sanitized[j]
            # skip empty lines
            if l2.strip() == "":
                j += 1
                continue
            leading = len(l2) - len(l2.lstrip(" \t"))
            # If dedented to same or less, block ended
            if leading <= len(indent):
                break
            # Consider non-comment as body; comments don't count
            if not l2.lstrip().startswith("#"):
                found_body = True
                break
            j += 1
        if not found_body:
            out.append(indent + "    pass")
        i += 1
    return "\n".join(out).strip()

def _detect_house_intent(prompt: str) -> bool:
    p = (prompt or "").lower()
    keywords = [
        "house", "home", "building", "room", "villa", "cottage", "hut",
        "cabin", "apartment", "flat", "bungalow", "residence", "dwelling",
        "shelter", "roof", "walls"
    ]
    return any(k in p for k in keywords)

def _detect_modify_intent(prompt: str) -> bool:
    p = (prompt or '').lower()
    mods = ['change', 'modify', 'adjust', 'update', 'replace', 'rotate', 'move', 'scale', 'resize', 'remake', 'switch', 'set']
    return any(m in p for m in mods)

def _infer_roof_type(prompt: str) -> str:
    p = (prompt or '').lower()
    if any(k in p for k in ['hip', 'pyramid', 'pyramidal']):
        return 'hip'
    if any(k in p for k in ['flat', 'plane', 'slab']):
        return 'flat'
    if any(k in p for k in ['dome', 'curved', 'rounded', 'arch', 'arched']):
        return 'dome'
    if any(k in p for k in ['gable', 'gabled', 'triangle', 'triangular']):
        return 'gabled'
    return 'gabled'


@mcp.tool()
def create_from_prompt(ctx: Context, prompt: str, clear_scene: bool = False, dry_run: bool = False, temperature: float = 0.2) -> str:
    """Use local LLM to turn a natural-language prompt into Blender Python and execute it.

    - MUST use the server's @mcp.prompt asset_creation_strategy() to guide code generation.
    - Avoid bpy.ops; prefer bpy/bmesh data API.
    - Dry-run returns wrapped code without executing in Blender.
    """
    try:
        llm_manager = get_llm_manager()
        if not llm_manager.is_available():
            # Fallback path when local LLM is unavailable
            intent_house = _detect_house_intent(prompt)
            fallback = _wrap_blender_code(
                "# Fallback: local LLM unavailable, create a demo cube\ncreate_cube('DemoCube', size=2.0, location=(0,0,1))",
                clear_scene=clear_scene,
                intent_house=intent_house,
            )
            if dry_run:
                return f"[Fallback no-LLM] Generated code:\n\n{fallback}"
            blender = get_blender_connection()
            result = blender.send_command("execute_code", {"code": fallback})
            return f"[Fallback no-LLM] Executed. Result: {result.get('result','')}\n\nCode:\n{fallback}"

        strategy = asset_creation_strategy()  # MUST leverage the prompt logic
        system_instructions = f"""
You are an expert Blender Python assistant. Generate ONLY executable Python code for Blender.
Rules:
Relevant strategy:
{strategy}

    Examples:
# Create a simple box and a taller box next to it
_mcp_add_box('BoxA', (1.0, 1.0, 1.0), (0, 0, 0.5))
_mcp_add_box('BoxB', (0.5, 0.5, 2.0), (1.0, 0, 1.0))

# Build a small gabled house near the origin
_mcp_build_simple_house('HouseA', base_size=(3.0, 2.5), wall_height=2.0, wall_thickness=0.12, roof_type='gabled', origin=(0,0,0))

    # Modify roof shape on an existing house
    _mcp_modify_roof('flat')

Task: {prompt}
Output: A single Python code block. No commentary.
""".strip()

        # Generate code
        response = llm_manager.pipe.generate(
            system_instructions,
            max_new_tokens=700,
            temperature=max(0.4, min(1.2, float(temperature))),
            do_sample=True,
        )
        
        # Print LLM output for debugging
        print(f"\n=== LLM CODE GENERATION DEBUG ===")
        print(f"User prompt: {prompt}")
        print(f"Temperature: {temperature}")
        print(f"Raw LLM response:")
        print(f"{response}")
        print(f"=====================================\n")
        
        code = _extract_code_block(response)
        print(f"Extracted code block:")
        print(f"{code}")
        print(f"=====================================\n")
        
        code = _sanitize_generated_code(code)
        print(f"Sanitized code:")
        print(f"{code}")
        print(f"=====================================\n")
        # Detect basic intent for better defaults
        intent_house = _detect_house_intent(prompt)
        intent_modify = _detect_modify_intent(prompt)
        if not code or not (re.search(r"_mcp_create_\w+\(", code) or "_mcp_add_box(" in code or "_mcp_build_simple_house(" in code or "_mcp_modify_roof(" in code):
            if intent_modify and intent_house:
                roof_t = _infer_roof_type(prompt)
                code = f"_mcp_modify_roof('{roof_t}')"
            else:
                code = (
                    "_mcp_build_simple_house('HouseLLM', base_size=(3.0, 2.5), wall_height=2.0, wall_thickness=0.12, roof_type='gabled', origin=(0,0,0))"
                    if intent_house else
                    "_mcp_add_box('AutoBox', (1.0, 1.0, 1.0), (0, 0, 0.5))"
                )
        wrapped = _wrap_blender_code(code, clear_scene=clear_scene, intent_house=intent_house)
        if dry_run:
            return f"Generated code (dry run):\n\n{wrapped}"

        blender = get_blender_connection()
        result = blender.send_command("execute_code", {"code": wrapped})
        return f"Executed generated code. Blender said:\n{result.get('result','')}\n\nCode:\n{wrapped}"
    except Exception as e:
        logger.error(f"Error in create_from_prompt: {e}")
        return f"Error creating from prompt: {e}"



@mcp.tool()
def plan_and_execute(
    ctx: Context,
    goal: str,
    max_steps: int = 6,
    temperature: float = 0.5,
) -> str:
    """Use the local LLM to plan and call MCP tools iteratively until the goal is met.

    The agent chooses among a small set of safe MCP tools (PolyHaven/Sketchfab/Hyper3D, create_from_prompt, execute_blender_code, scene/screenshot) and returns a concise transcript + result.
    """
    try:
        llm_manager = get_llm_manager()
        if not llm_manager.is_available():
            # Fallback: just generate Blender code from the goal
            return create_from_prompt(ctx, goal, clear_scene=False, dry_run=False, temperature=0.4)

        # Tool catalog: name -> (callable, arg spec string)
        tool_fns: Dict[str, Any] = {
            "get_scene_info": lambda kwargs: get_scene_info(None),
            "get_polyhaven_status": lambda kwargs: get_polyhaven_status(None),
            "get_sketchfab_status": lambda kwargs: get_sketchfab_status(None),
            "get_hyper3d_status": lambda kwargs: get_hyper3d_status(None),
            "get_polyhaven_categories": lambda kwargs: get_polyhaven_categories(None, kwargs.get("asset_type", "hdris")),
            "search_polyhaven_assets": lambda kwargs: search_polyhaven_assets(None, kwargs.get("asset_type", "all"), kwargs.get("categories")),
            "search_sketchfab_models": lambda kwargs: search_sketchfab_models(None, kwargs.get("query", ""), kwargs.get("max_results", 10), kwargs.get("downloadable_only", True)),
            "download_sketchfab_model": lambda kwargs: download_sketchfab_model(None, kwargs.get("uid", ""), kwargs.get("name", "SketchfabModel"), kwargs.get("max_polygons")),
            "generate_hyper3d_model_via_text": lambda kwargs: generate_hyper3d_model_via_text(None, kwargs.get("name", "RodinAsset"), kwargs.get("text_prompt", goal), kwargs.get("mode"), kwargs.get("api_key")),
            "generate_hyper3d_model_via_images": lambda kwargs: generate_hyper3d_model_via_images(None, kwargs.get("name", "RodinAsset"), kwargs.get("image_urls", []), kwargs.get("api_key")),
            "poll_rodin_job_status": lambda kwargs: poll_rodin_job_status(None, kwargs.get("subscription_key"), kwargs.get("request_id")),
            "import_generated_asset": lambda kwargs: import_generated_asset(None, kwargs.get("name", "RodinAsset"), kwargs.get("task_uuid"), kwargs.get("request_id")),
            "create_from_prompt": lambda kwargs: create_from_prompt(None, kwargs.get("prompt", goal), kwargs.get("clear_scene", False), kwargs.get("dry_run", False), kwargs.get("temperature", temperature)),
            "execute_blender_code": lambda kwargs: execute_blender_code(None, kwargs.get("code", "")),
            "get_viewport_screenshot": lambda kwargs: get_viewport_screenshot(None, kwargs.get("max_size", 800)),
        }
        tool_specs = [
            {
                "name": name,
                "args": "dynamic",
            }
            for name in tool_fns.keys()
        ]

        strategy = asset_creation_strategy()

        def summarize(obs: Any, limit: int = 600) -> str:
            try:
                if isinstance(obs, Image):
                    return f"[Image bytes: {len(obs.data)} bytes, format={obs.format}]"
                text = obs if isinstance(obs, str) else json.dumps(obs, indent=2)
            except Exception:
                text = str(obs)
            return text[:limit]

        def parse_json_block(txt: str) -> Dict[str, Any] | None:
            # Try to find a JSON object in the text
            m = re.search(r"\{[\s\S]*\}", txt)
            if not m:
                return None
            js = m.group(0)
            try:
                return json.loads(js)
            except Exception:
                return None

        transcript: List[Dict[str, Any]] = []

        for step in range(1, int(max_steps) + 1):
            last_obs = transcript[-1]["observation"] if transcript else "(none)"
            prompt = f"""
You are a helpful MCP agent that can use TOOLS to achieve the user's goal in Blender via a socket-connected addon.

TOOLS (call exactly one per step or finish):
{json.dumps(tool_specs, indent=2)}

Rules:
- Choose only from these tools and return STRICT JSON: {{"tool": "name", "args": {{...}}}} OR {{"final": "message"}}.
- Prefer libraries (PolyHaven/Sketchfab/Hyper3D) when available; otherwise generate Blender code.
- For Blender code, prefer tool "create_from_prompt" with a clear prompt.
- Keep steps small and observe results before continuing.
- Never include python code or commentary outside JSON.

Context strategy (guidance only):
{strategy}

User goal: {goal}
Step: {step}
Last observation: {last_obs}

Your response in JSON only:
""".strip()

            response = llm_manager.pipe.generate(
                prompt,
                max_new_tokens=256,
                temperature=max(0.2, min(1.0, float(temperature))),
                do_sample=True,
            )
            
            # Print LLM planning output for debugging
            print(f"\n=== LLM PLANNING DEBUG (Step {step}) ===")
            print(f"Goal: {goal}")
            print(f"Last observation: {last_obs}")
            print(f"LLM planning response:")
            print(f"{response}")
            print(f"==========================================\n")
            
            action = parse_json_block(response) or {}
            print(f"Parsed action: {action}")
            print(f"==========================================\n")

            # Finish condition
            if "final" in action and isinstance(action["final"], str):
                summary = action["final"]
                return json.dumps({
                    "status": "done",
                    "steps": transcript,
                    "summary": summary,
                }, indent=2)

            # Tool execution
            tool = action.get("tool")
            args = action.get("args", {})
            if tool not in tool_fns:
                # Fallback to Blender code generation
                tool = "create_from_prompt"
                args = {"prompt": goal, "clear_scene": False, "dry_run": False, "temperature": temperature}
            try:
                result = tool_fns[tool](args)
            except Exception as e:
                obs = f"Tool `{tool}` failed: {e}"
                transcript.append({"step": step, "tool": tool, "args": args, "observation": summarize(obs)})
                continue

            obs = summarize(result)
            transcript.append({"step": step, "tool": tool, "args": args, "observation": obs})

        # Max steps reached
        return json.dumps({
            "status": "max_steps",
            "steps": transcript,
            "summary": "Reached step limit",
        }, indent=2)
    except Exception as e:
        logger.error(f"Error in plan_and_execute: {e}")
        return f"Error in plan_and_execute: {e}"


@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Get detailed information about the current Blender scene"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_scene_info")
        
        # Optional: Use local LLM to enhance the response
        llm_manager = get_llm_manager()
        if llm_manager.is_available():
            logger.info("Enhanced with local LLM insights")

        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting scene info from Blender: {str(e)}")
        return f"Error getting scene info: {str(e)}"

@mcp.tool()
def get_object_info(ctx: Context, object_name: str) -> str:
    """
    Get detailed information about a specific object in the Blender scene.
    
    Parameters:
    - object_name: The name of the object to get information about
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_object_info", {"name": object_name})
        
        # Just return the JSON representation of what Blender sent us
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting object info from Blender: {str(e)}")
        return f"Error getting object info: {str(e)}"

@mcp.tool()
def get_viewport_screenshot(ctx: Context, max_size: int = 800) -> Image:
    """
    Capture a screenshot of the current Blender 3D viewport.
    
    Parameters:
    - max_size: Maximum size in pixels for the largest dimension (default: 800)
    
    Returns the screenshot as an Image.
    """
    try:
        blender = get_blender_connection()
        
        # Create temp file path
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"blender_screenshot_{os.getpid()}.png")
        
        result = blender.send_command("get_viewport_screenshot", {
            "max_size": max_size,
            "filepath": temp_path,
            "format": "png"
        })
        
        if "error" in result:
            raise Exception(result["error"])
        
        if not os.path.exists(temp_path):
            raise Exception("Screenshot file was not created")
        
        # Read the file
        with open(temp_path, 'rb') as f:
            image_bytes = f.read()
        
        # Delete the temp file
        os.remove(temp_path)
        
        return Image(data=image_bytes, format="png")
        
    except Exception as e:
        logger.error(f"Error capturing screenshot: {str(e)}")
        raise Exception(f"Screenshot failed: {str(e)}")


@mcp.tool()
def execute_blender_code(ctx: Context, code: str) -> str:
    """
    Execute arbitrary Python code in Blender. Make sure to do it step-by-step by breaking it into smaller chunks.
    
    Parameters:
    - code: The Python code to execute
    """
    try:
        # Get the global connection
        blender = get_blender_connection()
        result = blender.send_command("execute_code", {"code": code})
        return f"Code executed successfully: {result.get('result', '')}"
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return f"Error executing code: {str(e)}"

@mcp.tool()
def get_polyhaven_categories(ctx: Context, asset_type: str = "hdris") -> str:
    """
    Get a list of categories for a specific asset type on Polyhaven.
    
    Parameters:
    - asset_type: The type of asset to get categories for (hdris, textures, models, all)
    """
    try:
        blender = get_blender_connection()
        if not _polyhaven_enabled:
            return "PolyHaven integration is disabled. Select it in the sidebar in BlenderMCP, then run it again."
        result = blender.send_command("get_polyhaven_categories", {"asset_type": asset_type})
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Format the categories in a more readable way
        categories = result["categories"]
        formatted_output = f"Categories for {asset_type}:\n\n"
        
        # Sort categories by count (descending)
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories:
            formatted_output += f"- {category}: {count} assets\n"
        
        return formatted_output
    except Exception as e:
        logger.error(f"Error getting Polyhaven categories: {str(e)}")
        return f"Error getting Polyhaven categories: {str(e)}"

@mcp.tool()
def search_polyhaven_assets(
    ctx: Context,
    asset_type: str = "all",
    categories: str = None
) -> str:
    """
    Search for assets on Polyhaven with optional filtering.
    
    Parameters:
    - asset_type: Type of assets to search for (hdris, textures, models, all)
    - categories: Optional comma-separated list of categories to filter by
    
    Returns a list of matching assets with basic information.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("search_polyhaven_assets", {
            "asset_type": asset_type,
            "categories": categories
        })
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Format the assets in a more readable way
        assets = result["assets"]
        total_count = result["total_count"]
        returned_count = result["returned_count"]
        
        formatted_output = f"Found {total_count} assets"
        if categories:
            formatted_output += f" in categories: {categories}"
        formatted_output += f"\nShowing {returned_count} assets:\n\n"
        
        # Sort assets by download count (popularity)
        sorted_assets = sorted(assets.items(), key=lambda x: x[1].get("download_count", 0), reverse=True)
        
        for asset_id, asset_data in sorted_assets:
            formatted_output += f"- {asset_data.get('name', asset_id)} (ID: {asset_id})\n"
            formatted_output += f"  Type: {['HDRI', 'Texture', 'Model'][asset_data.get('type', 0)]}\n"
            formatted_output += f"  Categories: {', '.join(asset_data.get('categories', []))}\n"
            formatted_output += f"  Downloads: {asset_data.get('download_count', 'Unknown')}\n\n"
        
        return formatted_output
    except Exception as e:
        logger.error(f"Error searching Polyhaven assets: {str(e)}")
        return f"Error searching Polyhaven assets: {str(e)}"

@mcp.tool()
def download_polyhaven_asset(
    ctx: Context,
    asset_id: str,
    asset_type: str,
    resolution: str = "1k",
    file_format: str = None
) -> str:
    """
    Download and import a Polyhaven asset into Blender.
    
    Parameters:
    - asset_id: The ID of the asset to download
    - asset_type: The type of asset (hdris, textures, models)
    - resolution: The resolution to download (e.g., 1k, 2k, 4k)
    - file_format: Optional file format (e.g., hdr, exr for HDRIs; jpg, png for textures; gltf, fbx for models)
    
    Returns a message indicating success or failure.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("download_polyhaven_asset", {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "resolution": resolution,
            "file_format": file_format
        })
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            message = result.get("message", "Asset downloaded and imported successfully")
            
            # Add additional information based on asset type
            if asset_type == "hdris":
                return f"{message}. The HDRI has been set as the world environment."
            elif asset_type == "textures":
                material_name = result.get("material", "")
                maps = ", ".join(result.get("maps", []))
                return f"{message}. Created material '{material_name}' with maps: {maps}."
            elif asset_type == "models":
                return f"{message}. The model has been imported into the current scene."
            else:
                return message
        else:
            return f"Failed to download asset: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error downloading Polyhaven asset: {str(e)}")
        return f"Error downloading Polyhaven asset: {str(e)}"

@mcp.tool()
def set_texture(
    ctx: Context,
    object_name: str,
    texture_id: str
) -> str:
    """
    Apply a previously downloaded Polyhaven texture to an object.
    
    Parameters:
    - object_name: Name of the object to apply the texture to
    - texture_id: ID of the Polyhaven texture to apply (must be downloaded first)
    
    Returns a message indicating success or failure.
    """
    try:
        # Get the global connection
        blender = get_blender_connection()
        result = blender.send_command("set_texture", {
            "object_name": object_name,
            "texture_id": texture_id
        })
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            material_name = result.get("material", "")
            maps = ", ".join(result.get("maps", []))
            
            # Add detailed material info
            material_info = result.get("material_info", {})
            node_count = material_info.get("node_count", 0)
            has_nodes = material_info.get("has_nodes", False)
            texture_nodes = material_info.get("texture_nodes", [])
            
            output = f"Successfully applied texture '{texture_id}' to {object_name}.\n"
            output += f"Using material '{material_name}' with maps: {maps}.\n\n"
            output += f"Material has nodes: {has_nodes}\n"
            output += f"Total node count: {node_count}\n\n"
            
            if texture_nodes:
                output += "Texture nodes:\n"
                for node in texture_nodes:
                    output += f"- {node['name']} using image: {node['image']}\n"
                    if node['connections']:
                        output += "  Connections:\n"
                        for conn in node['connections']:
                            output += f"    {conn}\n"
            else:
                output += "No texture nodes found in the material.\n"
            
            return output
        else:
            return f"Failed to apply texture: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error applying texture: {str(e)}")
        return f"Error applying texture: {str(e)}"

@mcp.tool()
def get_polyhaven_status(ctx: Context) -> str:
    """
    Check if PolyHaven integration is enabled in Blender.
    Returns a message indicating whether PolyHaven features are available.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_polyhaven_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += "PolyHaven is good at Textures, and has a wider variety of textures than Sketchfab."
        return message
    except Exception as e:
        logger.error(f"Error checking PolyHaven status: {str(e)}")
        return f"Error checking PolyHaven status: {str(e)}"

@mcp.tool()
def get_hyper3d_status(ctx: Context) -> str:
    """
    Check if Hyper3D Rodin integration is enabled in Blender.
    Returns a message indicating whether Hyper3D Rodin features are available.

    Don't emphasize the key type in the returned message, but sliently remember it. 
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_hyper3d_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += ""
        return message
    except Exception as e:
        logger.error(f"Error checking Hyper3D status: {str(e)}")
        return f"Error checking Hyper3D status: {str(e)}"

@mcp.tool()
def get_sketchfab_status(ctx: Context) -> str:
    """
    Check if Sketchfab integration is enabled in Blender.
    Returns a message indicating whether Sketchfab features are available.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_sketchfab_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += "Sketchfab is good at Realistic models, and has a wider variety of models than PolyHaven."        
        return message
    except Exception as e:
        logger.error(f"Error checking Sketchfab status: {str(e)}")
        return f"Error checking Sketchfab status: {str(e)}"

@mcp.tool()
def search_sketchfab_models(
    ctx: Context,
    query: str,
    categories: str = None,
    count: int = 20,
    downloadable: bool = True
) -> str:
    """
    Search for models on Sketchfab with optional filtering.
    
    Parameters:
    - query: Text to search for
    - categories: Optional comma-separated list of categories
    - count: Maximum number of results to return (default 20)
    - downloadable: Whether to include only downloadable models (default True)
    
    Returns a formatted list of matching models.
    """
    try:
        
        blender = get_blender_connection()
        logger.info(f"Searching Sketchfab models with query: {query}, categories: {categories}, count: {count}, downloadable: {downloadable}")
        result = blender.send_command("search_sketchfab_models", {
            "query": query,
            "categories": categories,
            "count": count,
            "downloadable": downloadable
        })
        
        if "error" in result:
            logger.error(f"Error from Sketchfab search: {result['error']}")
            return f"Error: {result['error']}"
        
        # Safely get results with fallbacks for None
        if result is None:
            logger.error("Received None result from Sketchfab search")
            return "Error: Received no response from Sketchfab search"
            
        # Format the results
        models = result.get("results", []) or []
        if not models:
            return f"No models found matching '{query}'"
            
        formatted_output = f"Found {len(models)} models matching '{query}':\n\n"
        
        for model in models:
            if model is None:
                continue
                
            model_name = model.get("name", "Unnamed model")
            model_uid = model.get("uid", "Unknown ID")
            formatted_output += f"- {model_name} (UID: {model_uid})\n"
            
            # Get user info with safety checks
            user = model.get("user") or {}
            username = user.get("username", "Unknown author") if isinstance(user, dict) else "Unknown author"
            formatted_output += f"  Author: {username}\n"
            
            # Get license info with safety checks
            license_data = model.get("license") or {}
            license_label = license_data.get("label", "Unknown") if isinstance(license_data, dict) else "Unknown"
            formatted_output += f"  License: {license_label}\n"
            
            # Add face count and downloadable status
            face_count = model.get("faceCount", "Unknown")
            is_downloadable = "Yes" if model.get("isDownloadable") else "No"
            formatted_output += f"  Face count: {face_count}\n"
            formatted_output += f"  Downloadable: {is_downloadable}\n\n"
        
        return formatted_output
    except Exception as e:
        logger.error(f"Error searching Sketchfab models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error searching Sketchfab models: {str(e)}"

@mcp.tool()
def download_sketchfab_model(
    ctx: Context,
    uid: str
) -> str:
    """
    Download and import a Sketchfab model by its UID.
    
    Parameters:
    - uid: The unique identifier of the Sketchfab model
    
    Returns a message indicating success or failure.
    The model must be downloadable and you must have proper access rights.
    """
    try:
        
        blender = get_blender_connection()
        logger.info(f"Attempting to download Sketchfab model with UID: {uid}")
        
        result = blender.send_command("download_sketchfab_model", {
            "uid": uid
        })
        
        if result is None:
            logger.error("Received None result from Sketchfab download")
            return "Error: Received no response from Sketchfab download request"
            
        if "error" in result:
            logger.error(f"Error from Sketchfab download: {result['error']}")
            return f"Error: {result['error']}"
        
        if result.get("success"):
            imported_objects = result.get("imported_objects", [])
            object_names = ", ".join(imported_objects) if imported_objects else "none"
            return f"Successfully imported model. Created objects: {object_names}"
        else:
            return f"Failed to download model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error downloading Sketchfab model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error downloading Sketchfab model: {str(e)}"

def _process_bbox(original_bbox: list[float] | list[int] | None) -> list[int] | None:
    if original_bbox is None:
        return None
    if all(isinstance(i, int) for i in original_bbox):
        return original_bbox
    if any(i<=0 for i in original_bbox):
        raise ValueError("Incorrect number range: bbox must be bigger than zero!")
    return [int(float(i) / max(original_bbox) * 100) for i in original_bbox] if original_bbox else None

@mcp.tool()
def generate_hyper3d_model_via_text(
    ctx: Context,
    text_prompt: str,
    bbox_condition: list[float]=None
) -> str:
    """
    Generate 3D asset using Hyper3D by giving description of the desired asset, and import the asset into Blender.
    The 3D asset has built-in materials.
    The generated model has a normalized size, so re-scaling after generation can be useful.
    
    Parameters:
    - text_prompt: A short description of the desired model in **English**.
    - bbox_condition: Optional. If given, it has to be a list of floats of length 3. Controls the ratio between [Length, Width, Height] of the model.

    Returns a message indicating success or failure.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_rodin_job", {
            "text_prompt": text_prompt,
            "images": None,
            "bbox_condition": _process_bbox(bbox_condition),
        })
        succeed = result.get("submit_time", False)
        if succeed:
            return json.dumps({
                "task_uuid": result["uuid"],
                "subscription_key": result["jobs"]["subscription_key"],
            })
        else:
            return json.dumps(result)
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def generate_hyper3d_model_via_images(
    ctx: Context,
    input_image_paths: list[str]=None,
    input_image_urls: list[str]=None,
    bbox_condition: list[float]=None
) -> str:
    """
    Generate 3D asset using Hyper3D by giving images of the wanted asset, and import the generated asset into Blender.
    The 3D asset has built-in materials.
    The generated model has a normalized size, so re-scaling after generation can be useful.
    
    Parameters:
    - input_image_paths: The **absolute** paths of input images. Even if only one image is provided, wrap it into a list. Required if Hyper3D Rodin in MAIN_SITE mode.
    - input_image_urls: The URLs of input images. Even if only one image is provided, wrap it into a list. Required if Hyper3D Rodin in FAL_AI mode.
    - bbox_condition: Optional. If given, it has to be a list of ints of length 3. Controls the ratio between [Length, Width, Height] of the model.

    Only one of {input_image_paths, input_image_urls} should be given at a time, depending on the Hyper3D Rodin's current mode.
    Returns a message indicating success or failure.
    """
    if input_image_paths is not None and input_image_urls is not None:
        return f"Error: Conflict parameters given!"
    if input_image_paths is None and input_image_urls is None:
        return f"Error: No image given!"
    if input_image_paths is not None:
        if not all(os.path.exists(i) for i in input_image_paths):
            return "Error: not all image paths are valid!"
        images = []
        for path in input_image_paths:
            with open(path, "rb") as f:
                images.append(
                    (Path(path).suffix, base64.b64encode(f.read()).decode("ascii"))
                )
    elif input_image_urls is not None:
        if not all(urlparse(i) for i in input_image_urls):
            return "Error: not all image URLs are valid!"
        images = input_image_urls.copy()
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_rodin_job", {
            "text_prompt": None,
            "images": images,
            "bbox_condition": _process_bbox(bbox_condition),
        })
        succeed = result.get("submit_time", False)
        if succeed:
            return json.dumps({
                "task_uuid": result["uuid"],
                "subscription_key": result["jobs"]["subscription_key"],
            })
        else:
            return json.dumps(result)
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def poll_rodin_job_status(
    ctx: Context,
    subscription_key: str=None,
    request_id: str=None,
):
    """
    Check if the Hyper3D Rodin generation task is completed.

    For Hyper3D Rodin mode MAIN_SITE:
        Parameters:
        - subscription_key: The subscription_key given in the generate model step.

        Returns a list of status. The task is done if all status are "Done".
        If "Failed" showed up, the generating process failed.
        This is a polling API, so only proceed if the status are finally determined ("Done" or "Canceled").

    For Hyper3D Rodin mode FAL_AI:
        Parameters:
        - request_id: The request_id given in the generate model step.

        Returns the generation task status. The task is done if status is "COMPLETED".
        The task is in progress if status is "IN_PROGRESS".
        If status other than "COMPLETED", "IN_PROGRESS", "IN_QUEUE" showed up, the generating process might be failed.
        This is a polling API, so only proceed if the status are finally determined ("COMPLETED" or some failed state).
    """
    try:
        blender = get_blender_connection()
        kwargs = {}
        if subscription_key:
            kwargs = {
                "subscription_key": subscription_key,
            }
        elif request_id:
            kwargs = {
                "request_id": request_id,
            }
        result = blender.send_command("poll_rodin_job_status", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def import_generated_asset(
    ctx: Context,
    name: str,
    task_uuid: str=None,
    request_id: str=None,
):
    """
    Import the asset generated by Hyper3D Rodin after the generation task is completed.

    Parameters:
    - name: The name of the object in scene
    - task_uuid: For Hyper3D Rodin mode MAIN_SITE: The task_uuid given in the generate model step.
    - request_id: For Hyper3D Rodin mode FAL_AI: The request_id given in the generate model step.

    Only give one of {task_uuid, request_id} based on the Hyper3D Rodin Mode!
    Return if the asset has been imported successfully.
    """
    try:
        blender = get_blender_connection()
        kwargs = {
            "name": name
        }
        if task_uuid:
            kwargs["task_uuid"] = task_uuid
        elif request_id:
            kwargs["request_id"] = request_id
        result = blender.send_command("import_generated_asset", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def llm_suggest_tool(ctx: Context, user_request: str) -> str:
    """Suggest the most appropriate MCP tool name for a given request using the local LLM."""
    try:
        m = get_llm_manager()
        if not m.is_available():
            return "Local LLM unavailable."
        avail = [
            "get_scene_info", "get_object_info", "execute_blender_code", "get_viewport_screenshot",
            "get_polyhaven_status", "search_polyhaven_assets", "download_polyhaven_asset",
            "get_sketchfab_status", "search_sketchfab_models", "download_sketchfab_model",
        ]
        return json.dumps(m.enhance_tool_usage(user_request, avail))
    except Exception as e:
        return f"Error suggesting tool: {e}"

@mcp.tool()
def agent_execute(ctx: Context, user_request: str, max_steps: int = 6, temperature: float = 0.5) -> str:
    """Alias for plan_and_execute to run a local planning loop inside the server."""
    try:
        return plan_and_execute(ctx, goal=user_request, max_steps=max_steps, temperature=temperature)
    except Exception as e:
        return f"Error in agent_execute: {e}"

@mcp.prompt()
def asset_creation_strategy() -> str:
    """Defines the preferred strategy for creating assets in Blender"""
    return """When creating 3D content in Blender, always start by checking if integrations are available:

    0. Before anything, always check the scene from get_scene_info()
    1. First use the following tools to verify if the following integrations are enabled:
        1. PolyHaven
            Use get_polyhaven_status() to verify its status
            If PolyHaven is enabled:
            - For objects/models: Use download_polyhaven_asset() with asset_type="models"
            - For materials/textures: Use download_polyhaven_asset() with asset_type="textures"
            - For environment lighting: Use download_polyhaven_asset() with asset_type="hdris"
        2. Sketchfab
            Sketchfab is good at Realistic models, and has a wider variety of models than PolyHaven.
            Use get_sketchfab_status() to verify its status
            If Sketchfab is enabled:
            - For objects/models: First search using search_sketchfab_models() with your query
            - Then download specific models using download_sketchfab_model() with the UID
            - Note that only downloadable models can be accessed, and API key must be properly configured
            - Sketchfab has a wider variety of models than PolyHaven, especially for specific subjects
        3. Hyper3D(Rodin)
            Hyper3D Rodin is good at generating 3D models for single item.
            So don't try to:
            1. Generate the whole scene with one shot
            2. Generate ground using Hyper3D
            3. Generate parts of the items separately and put them together afterwards

            Use get_hyper3d_status() to verify its status
            If Hyper3D is enabled:
            - For objects/models, do the following steps:
                1. Create the model generation task
                    - Use generate_hyper3d_model_via_images() if image(s) is/are given
                    - Use generate_hyper3d_model_via_text() if generating 3D asset using text prompt
                    If key type is free_trial and insufficient balance error returned, tell the user that the free trial key can only generated limited models everyday, they can choose to:
                    - Wait for another day and try again
                    - Go to hyper3d.ai to find out how to get their own API key
                    - Go to fal.ai to get their own private API key
                2. Poll the status
                    - Use poll_rodin_job_status() to check if the generation task has completed or failed
                3. Import the asset
                    - Use import_generated_asset() to import the generated GLB model the asset
                4. After importing the asset, ALWAYS check the world_bounding_box of the imported mesh, and adjust the mesh's location and size
                    Adjust the imported mesh's location, scale, rotation, so that the mesh is on the right spot.

                You can reuse assets previous generated by running python code to duplicate the object, without creating another generation task.

    3. Always check the world_bounding_box for each item so that:
        - Ensure that all objects that should not be clipping are not clipping.
        - Items have right spatial relationship.
    
    4. Recommended asset source priority:
        - For specific existing objects: First try Sketchfab, then PolyHaven
        - For generic objects/furniture: First try PolyHaven, then Sketchfab
        - For custom or unique items not available in libraries: Use Hyper3D Rodin
        - For environment lighting: Use PolyHaven HDRIs
        - For materials/textures: Use PolyHaven textures

    Only fall back to scripting when:
    - PolyHaven, Sketchfab, and Hyper3D are all disabled
    - A simple primitive is explicitly requested
    - No suitable asset exists in any of the libraries
    - Hyper3D Rodin failed to generate the desired asset
    - The task specifically requires a basic material/color
    """

# Main execution

def main():
    """Run the MCP server with integrated local LLM"""
    logger.info("Framework: Actual FastMCP with OpenVINO local LLM")
    mcp.run()

if __name__ == "__main__":
    main()