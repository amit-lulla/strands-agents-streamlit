# Fix torch/asyncio conflicts
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set AWS configuration - users should configure their own AWS credentials
os.environ['AWS_PROFILE'] = 'default'  # Uncomment and set your AWS profile
os.environ['AWS_DEFAULT_REGION'] = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

# Disable torch JIT warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

import streamlit as st
import json
import inspect
import time
from typing import Dict, Any, List, Optional, Callable
import numpy as np

from strands import Agent, tool
from strands.models import BedrockModel
from strands.models.ollama import OllamaModel
from strands.types.tools import ToolResult, ToolUse

# Circuit breaker and semantic search imports
from pybreaker import CircuitBreaker
from txtai.embeddings import Embeddings
import boto3

import tools.list_appointments
import tools.update_appointment
import tools.create_appointment
from strands_tools import calculator, current_time

import logging
import re

# Configure logging
logging.getLogger("strands.tools.registry").setLevel(logging.WARNING)
logging.getLogger("strands.models").setLevel(logging.DEBUG)

file_handler = logging.FileHandler("./strands.log")
logging.getLogger("strands").addHandler(file_handler)

# Define all classes and functions first
class ToolRegistry:
    """Registry for managing tools with semantic search capabilities"""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.embeddings = None
        self.tool_descriptions = []
        self.tool_names = []
        
    def initialize_embeddings(self):
        """Initialize the txtai embeddings model"""
        self.embeddings = Embeddings({
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "content": True,
            # "scoring": {
            #     "method": "bm25",
            #     "terms": True
            # }
        })
        
    def _parse_docstring_parameters(self, docstring: str) -> Dict[str, str]:
        """Parse parameter descriptions from docstring Args section"""
        import re
        param_descriptions = {}
        
        if not docstring:
            return param_descriptions
            
        # Look for Args: section in docstring
        args_match = re.search(r'Args:\s*\n(.*?)(?:\n\s*\n|Returns:|Raises:|$)', docstring, re.DOTALL | re.IGNORECASE)
        if args_match:
            args_section = args_match.group(1)
            
            # Parse individual parameter descriptions
            # Pattern: param_name (type): description
            param_pattern = r'\s*(\w+)\s*\([^)]+\):\s*(.+?)(?=\n\s*\w+\s*\(|$)'
            matches = re.findall(param_pattern, args_section, re.DOTALL)
            
            for param_name, description in matches:
                # Clean up description
                description = re.sub(r'\s+', ' ', description.strip())
                param_descriptions[param_name] = description
                
        return param_descriptions
    
    def extract_tool_info(self, tool_func) -> Dict[str, Any]:
        """Extract tool information from a tool function"""
        # Handle different tool formats
        if hasattr(tool_func, '__wrapped__'):
            # This is a decorated tool
            actual_func = tool_func.__wrapped__
            tool_name = actual_func.__name__
            tool_description = actual_func.__doc__ or ""
            # Also try to get description from the wrapper
            if not tool_description and hasattr(tool_func, '__doc__'):
                tool_description = tool_func.__doc__ or ""
        elif hasattr(tool_func, 'TOOL_SPEC'):
            # This is a tool with TOOL_SPEC
            tool_spec = tool_func.TOOL_SPEC
            tool_name = tool_spec['name']
            tool_description = tool_spec['description']
            actual_func = getattr(tool_func, tool_name, tool_func)
        elif isinstance(tool_func, dict) and 'name' in tool_func:
            # This is a tool spec dict
            tool_name = tool_func['name']
            tool_description = tool_func.get('description', '')
            actual_func = None
        else:
            # Try to get function info directly
            tool_name = getattr(tool_func, '__name__', str(tool_func))
            # Remove module prefix if present
            if '.' in tool_name:
                tool_name = tool_name.split('.')[-1]
            tool_description = getattr(tool_func, '__doc__', '') or ""
            actual_func = tool_func
            
        # Enhance descriptions for common tools if they're empty
        if not tool_description.strip():
            if tool_name == 'current_time':
                tool_description = "Get the current date and time"
            elif tool_name == 'list_appointments':
                tool_description = "List all available appointments from the database"
            elif tool_name == 'create_appointment':
                tool_description = "Create a new personal appointment in the database"
            elif tool_name == 'update_appointment':
                tool_description = "Update an existing appointment based on the appointment ID"
            elif tool_name == 'calculator':
                tool_description = "Perform mathematical calculations and computations"
            
        # Extract parameters if possible
        params = {}
        if actual_func and callable(actual_func):
            try:
                sig = inspect.signature(actual_func)
                
                # Parse docstring for parameter descriptions
                param_descriptions = self._parse_docstring_parameters(tool_description)
                
                for param_name, param in sig.parameters.items():
                    if param_name not in ['self', 'tool', 'kwargs']:
                        param_info = {
                            'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                            'required': param.default == inspect.Parameter.empty
                        }
                        
                        # Add description from docstring if available
                        if param_name in param_descriptions:
                            param_info['description'] = param_descriptions[param_name]
                        
                        params[param_name] = param_info
            except:
                pass
                
        return {
            'name': tool_name,
            'description': tool_description.strip(),
            'function': tool_func,
            'parameters': params
        }
    
    def register_tools(self, tools: List[Any]):
        """Register multiple tools and build the semantic index"""
        self.initialize_embeddings()
        
        for tool in tools:
            tool_info = self.extract_tool_info(tool)
            self.tools[tool_info['name']] = tool_info
            
            # Create searchable text combining name and description
            search_text = f"{tool_info['name']} {tool_info['description']}"
            self.tool_descriptions.append(search_text)
            self.tool_names.append(tool_info['name'])
        
        # Build the index
        documents = [(i, text, None) for i, text in enumerate(self.tool_descriptions)]
        self.embeddings.index(documents)
        
    def search_tool(self, query: str, threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        """Search for the best matching tool using semantic search"""
        if not self.embeddings:
            return None
            
        # Search for the best match
        results = self.embeddings.search(query, 1)
        
        if results and len(results) > 0:
            score = results[0]['score']
            if score >= threshold:
                idx = int(results[0]['id'])  # Convert string id to integer
                tool_name = self.tool_names[idx]
                return self.tools[tool_name]
        
        return None
    
    def can_execute_directly(self, tool_info: Dict[str, Any], query: str) -> bool:
        """Determine if a tool can be executed directly without LLM"""
        # Check tool name without module prefix
        tool_name = tool_info['name']
        if '.' in tool_name:
            tool_name = tool_name.split('.')[-1]
            
        # Tools that can always be executed directly
        always_direct_tools = ['list_appointments', 'current_time']
        if tool_name in always_direct_tools:
            return True
            
        # Calculator can be executed directly if it contains math expressions
        if tool_name == 'calculator':
            import re
            # Look for mathematical expressions or calculation keywords
            math_patterns = [
                r'\d+[\+\-\*/]\d+',  # Basic math: 5+3, 10*2
                r'calculate|compute|math',  # Calculation keywords
                r'\d+\s*[\+\-\*/]\s*\d+',  # Math with spaces
                r'what\s+is\s+\d+',  # "what is 5+3"
            ]
            for pattern in math_patterns:
                if re.search(pattern, query.lower()):
                    return True
                    
        return False
    
    def extract_parameters(self, tool_info: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Extract parameters using Ollama gemma3:1b model"""
        try:
            # Use Ollama for universal parameter extraction
            params = self._extract_parameters_with_ollama(tool_info, query)
            
            # Store debug info for the semantic tool lookup tab
            if not hasattr(st.session_state, 'last_parameter_extraction'):
                st.session_state.last_parameter_extraction = {}
            
            st.session_state.last_parameter_extraction = {
                'query': query,
                'tool_name': tool_info['name'],
                'tool_description': tool_info['description'],
                'extracted_params': params,
                'extraction_method': 'ollama_gemma3_1b',
                'timestamp': time.time()
            }
            
            return params
            
        except Exception as e:
            logging.warning(f"Ollama parameter extraction failed: {e}")
            # Fallback to regex-based extraction
            return self._extract_parameters_fallback(tool_info, query)
    
    def _extract_parameters_with_ollama(self, tool_info: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Extract parameters using Ollama gemma3:1b model"""
        import requests
        import json
        
        # Create extraction prompt
        extraction_prompt = self._create_ollama_extraction_prompt(tool_info, query)
        
        try:
            # Call Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3:1b",
                    "prompt": extraction_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 200,
                        "stop": ["\n\n", "```", "---"]
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # Store raw response for debugging
                st.session_state.last_ollama_response = {
                    'prompt': extraction_prompt,
                    'raw_response': response_text,
                    'model': 'gemma3:1b'
                }
                
                return self._parse_ollama_json_response(response_text)
            else:
                logging.error(f"Ollama API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logging.error(f"Ollama parameter extraction error: {e}")
            raise
    
    def _create_ollama_extraction_prompt(self, tool_info: Dict[str, Any], query: str) -> str:
        """Create optimized prompt for Ollama parameter extraction"""
        tool_name = tool_info['name']
        tool_description = tool_info['description']
        
        # Build parameter schema
        param_schema = []
        if 'parameters' in tool_info and tool_info['parameters']:
            for param_name, param_info in tool_info['parameters'].items():
                param_type = param_info.get('type', 'string')
                required = param_info.get('required', False)
                description = param_info.get('description', f'{param_name} parameter')
                req_marker = " (REQUIRED)" if required else " (optional)"
                param_schema.append(f"  {param_name} ({param_type}){req_marker}: {description}")
        
        prompt = f"""Extract parameters from user query for tool execution.

TOOL: {tool_name}
DESCRIPTION: {tool_description}

PARAMETERS:
{chr(10).join(param_schema) if param_schema else '  No parameters required'}

USER QUERY: "{query}"

Extract only parameters clearly determined from the query. Respond with valid JSON only.

JSON:"""
        
        return prompt
    
    def _parse_ollama_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from Ollama response"""
        import json
        import re
        
        response_text = response_text.strip()
        
        # Try direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Extract JSON using regex
        json_patterns = [
            r'\{[^{}]*\}',
            r'\{.*?\}'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Fallback key-value parsing
        return self._parse_key_value_fallback(response_text)
    
    def _parse_key_value_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback parser for key-value pairs"""
        import re
        params = {}
        
        patterns = [
            r'"(\w+)":\s*"([^"]+)"',
            r'(\w+):\s*"([^"]+)"',
            r'(\w+):\s*([^,\n}]+)',
            r'"(\w+)":\s*([^,\n}]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                value = value.strip().strip('"').strip("'")
                if value and value.lower() not in ['null', 'none', '']:
                    params[key.strip()] = value
        
        return params
    
    def _extract_parameters_fallback(self, tool_info: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Regex-based fallback extraction"""
        import re
        from datetime import datetime, timedelta
        
        params = {}
        tool_name = tool_info['name']
        if '.' in tool_name:
            tool_name = tool_name.split('.')[-1]
            
        if tool_name == 'calculator':
            patterns = [
                r'calculate\s+([\d\+\-\*/\(\)\s\.]+)',
                r'compute\s+([\d\+\-\*/\(\)\s\.]+)',
                r'what\s+is\s+([\d\+\-\*/\(\)\s\.]+)',
                r'([\d\+\-\*/\(\)\s\.]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query.lower())
                if match:
                    expression = match.group(1) if len(match.groups()) > 0 else match.group(0)
                    expression = re.sub(r'[^\d\+\-\*/\(\)\s\.]', '', expression).strip()
                    if expression and any(op in expression for op in ['+', '-', '*', '/']):
                        params['expression'] = expression
                        break
        
        return params


class BedrockWithFallback:
    """Wrapper for Bedrock model with circuit breaker and Ollama fallback"""
    
    def __init__(self, bedrock_model, ollama_model="deepseek-r1:latest"):
        self.bedrock_model = bedrock_model
        self.ollama_model = ollama_model
        
        # Configure circuit breaker with 2-second timeout
        self.breaker = CircuitBreaker(
            fail_max=3,
            reset_timeout=60,
            exclude=[Exception],  # Only trip on timeouts
            name="BedrockCircuitBreaker"
        )
        
    def _call_bedrock_with_timeout(self, messages, tool_specs, system_prompt):
        """Call Bedrock with timeout and proper message format"""
        import threading
        result = [None]
        exception = [None]
        
        # Get dynamic timeout from session state
        timeout_seconds = st.session_state.get('circuit_timeout', 2)
        
        # Check if circuit test is forced
        if st.session_state.get('force_circuit_test', False):
            st.session_state.force_circuit_test = False  # Reset flag
            raise TimeoutError("Circuit breaker test - forced timeout")
        
        # Convert messages to Bedrock format
        bedrock_messages = self._convert_to_bedrock_format(messages)
        
        def target():
            try:
                result[0] = self.bedrock_model.converse(bedrock_messages, tool_specs, system_prompt)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=float(timeout_seconds))
        
        if thread.is_alive():
            raise TimeoutError(f"Bedrock call timed out after {timeout_seconds} seconds")
        
        if exception[0]:
            raise exception[0]
            
        return result[0]
    
    def _convert_to_bedrock_format(self, messages):
        """Convert simple string format messages to Bedrock format with content blocks"""
        bedrock_messages = []
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Skip empty messages
            if not content or (isinstance(content, str) and not content.strip()):
                continue
            
            # Convert to Bedrock format: content as list of content blocks
            if isinstance(content, str):
                bedrock_content = [{"text": content.strip()}]
            elif isinstance(content, list):
                # Already in Bedrock format, validate it
                bedrock_content = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        if text and isinstance(text, str) and text.strip():
                            bedrock_content.append({"text": text.strip()})
                if not bedrock_content:
                    continue  # Skip if no valid content
            else:
                continue  # Skip invalid content types
            
            bedrock_messages.append({
                "role": msg["role"],
                "content": bedrock_content
            })
        
        return bedrock_messages
    
    def _convert_to_ollama_format(self, messages):
        """Convert messages to simple string format for Ollama"""
        ollama_messages = []
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Skip empty messages
            if not content or (isinstance(content, str) and not content.strip()):
                continue
            
            # Convert to simple string format
            if isinstance(content, str):
                ollama_content = content.strip()
            elif isinstance(content, list):
                # Extract text from Bedrock format
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        if text and isinstance(text, str) and text.strip():
                            text_parts.append(text.strip())
                ollama_content = ' '.join(text_parts) if text_parts else ""
                if not ollama_content:
                    continue  # Skip if no valid content
            else:
                continue  # Skip invalid content types
            
            ollama_messages.append({
                "role": msg["role"],
                "content": ollama_content
            })
        
        return ollama_messages
    
    def _call_ollama_fallback(self, messages, tool_specs, system_prompt):
        """Fallback to Ollama when Bedrock fails using Strands OllamaModel"""
        try:
            # Get dynamic model from session state
            selected_model = st.session_state.get('selected_ollama_model', self.ollama_model)
            
            # Create Strands OllamaModel instance
            ollama_model = OllamaModel(
                host="http://localhost:11434",
                model_id=selected_model,
                temperature=0.7,
                keep_alive="5m"
            )
            
            # Ensure messages are in simple string format for Ollama
            ollama_messages = self._convert_to_ollama_format(messages)
            
            # First try without tools to avoid the error
            # Most models don't support tools, so we'll default to text-only mode
            try:
                # Try without tools first for better compatibility
                response_stream = ollama_model.converse(ollama_messages, [], system_prompt)
                
                # Return the actual stream from Ollama
                return response_stream
                
            except Exception as no_tool_error:
                # If even text-only fails, provide a helpful error message
                logging.error(f"Ollama model {selected_model} failed even without tools: {no_tool_error}")
                
                # Extract user message for context
                user_message = ""
                for msg in messages:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        content = msg.get('content', '')
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                            user_message = ' '.join(text_parts)
                        else:
                            user_message = content
                        break
                
                # Generate a helpful response explaining the limitation
                if "does not support tools" in str(no_tool_error).lower():
                    fallback_content = f"I'm currently running on {selected_model} which doesn't support tool usage. I can provide general assistance, but I cannot execute specific tools like calculator, appointments, or time functions. For full functionality, please ensure Bedrock is available or use a tool-compatible Ollama model like 'llama3'."
                else:
                    fallback_content = f"Ollama model {selected_model} encountered an error: {str(no_tool_error)}. Please check if the model is properly installed and running."
                
                # Create a proper streaming response that mimics Bedrock format
                def create_error_stream():
                    yield {'messageStart': {'role': 'assistant'}}
                    yield {'contentBlockStart': {'start': {'text': ''}}}
                    
                    # Stream the content in chunks for better UX
                    words = fallback_content.split(' ')
                    for i, word in enumerate(words):
                        chunk_text = word + (' ' if i < len(words) - 1 else '')
                        yield {'contentBlockDelta': {'delta': {'text': chunk_text}}}
                    
                    yield {'contentBlockStop': {}}
                    yield {'messageStop': {'stopReason': 'end_turn'}}
                
                return create_error_stream()
            
        except Exception as e:
            logging.error(f"Ollama fallback failed: {e}")
            error_content = f"Ollama fallback failed ({st.session_state.get('selected_ollama_model', 'unknown')}): {str(e)}. Please check if the model is available and try again."
            
            # Create a proper streaming response for the error
            def create_fallback_error_stream():
                yield {'messageStart': {'role': 'assistant'}}
                yield {'contentBlockStart': {'start': {'text': ''}}}
                
                # Stream the error content in chunks
                words = error_content.split(' ')
                for i, word in enumerate(words):
                    chunk_text = word + (' ' if i < len(words) - 1 else '')
                    yield {'contentBlockDelta': {'delta': {'text': chunk_text}}}
                
                yield {'contentBlockStop': {}}
                yield {'messageStop': {'stopReason': 'end_turn'}}
            
            return create_fallback_error_stream()
    
    def converse(self, messages, tool_specs, system_prompt):
        """Main converse method that implements the expected interface"""
        try:
            # Set current model status for sidebar display
            st.session_state.current_model = "Bedrock Claude-3.7-Sonnet"
            st.session_state.model_start_time = time.time()
            
            # Try Bedrock first with circuit breaker
            result = self.breaker(self._call_bedrock_with_timeout)(messages, tool_specs, system_prompt)
            
            # Keep model status visible for a few seconds after completion
            st.session_state.model_end_time = time.time()
            return result
            
        except Exception as e:
            logging.warning(f"Bedrock failed: {e}. Falling back to Ollama.")
            
            # Update model status for Ollama fallback
            selected_model = st.session_state.get('selected_ollama_model', self.ollama_model)
            st.session_state.current_model = f"Ollama {selected_model}"
            st.session_state.model_start_time = time.time()
            
            result = self._call_ollama_fallback(messages, tool_specs, system_prompt)
            
            # Keep model status visible for a few seconds after completion
            st.session_state.model_end_time = time.time()
            return result
    
    def __getattr__(self, name):
        """Delegate other attributes to the underlying bedrock model"""
        return getattr(self.bedrock_model, name)


def discover_all_tools():
    """Automatically discover all tools from various sources"""
    all_tools = []
    seen_functions = set()  # Track actual function objects to avoid duplicates
    
    def add_tool_if_unique(tool_obj, source="unknown"):
        """Add tool if it's not a duplicate based on the actual function"""
        # Get the actual function object (unwrap if decorated)
        actual_func = tool_obj
        if hasattr(tool_obj, '__wrapped__'):
            actual_func = tool_obj.__wrapped__
        
        # Use id() to uniquely identify the function object
        func_id = id(actual_func)
        
        if func_id not in seen_functions:
            seen_functions.add(func_id)
            all_tools.append(tool_obj)
            tool_name = getattr(tool_obj, '__name__', str(tool_obj))
            logging.info(f"Discovered tool from {source}: {tool_name}")
            return True
        return False
    
    # 1. Manually imported tools (from strands_tools)
    try:
        from strands_tools import current_time, calculator
        add_tool_if_unique(current_time, "strands_tools")
        add_tool_if_unique(calculator, "strands_tools")
    except ImportError:
        logging.warning("Could not import strands_tools")
    
    # 2. Tools from ./tools directory (both @tool decorated and TOOL_SPEC)
    tools_dir = os.path.join(os.path.dirname(__file__), 'tools')
    if os.path.exists(tools_dir):
        for filename in os.listdir(tools_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                try:
                    module = __import__(f'tools.{module_name}', fromlist=[module_name])
                    
                    # Look for @tool decorated functions
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        # Check if it's a tool (decorated with @tool or has TOOL_SPEC)
                        if (hasattr(attr, '__wrapped__') or 
                            hasattr(attr, 'TOOL_SPEC') or
                            (callable(attr) and hasattr(attr, '__name__') and 
                             attr.__name__ in ['create_appointment', 'list_appointments', 'update_appointment'])):
                            
                            add_tool_if_unique(attr, f"./tools/{filename}")
                                
                except Exception as e:
                    logging.warning(f"Failed to import tool from {filename}: {e}")
    
    return all_tools

def should_refresh_tools():
    """Check if tools should be refreshed (e.g., file changes)"""
    # Only refresh if not already initialized
    if not st.session_state.get('tools_initialized', False):
        return True
    
    # In production, you might want to check file modification times
    # For now, only refresh once per session
    return False

def parse_response_content(content: str) -> tuple[str, str]:
    """Parse response content to separate thinking and main content"""
    # Look for <think> tags
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, content, re.DOTALL)
    
    # Remove think tags from main content
    main_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
    
    # Combine all thinking content
    thinking_content = '\n\n'.join(think_matches) if think_matches else ''
    
    return main_content, thinking_content

def is_markdown_content(content: str) -> bool:
    """Check if content appears to be markdown"""
    markdown_indicators = [
        r'^#{1,6}\s',  # Headers
        r'\*\*.*?\*\*',  # Bold
        r'\*.*?\*',  # Italic
        r'`.*?`',  # Inline code
        r'^```',  # Code blocks
        r'^\*\s',  # Bullet points
        r'^\d+\.\s',  # Numbered lists
        r'\[.*?\]\(.*?\)',  # Links
    ]
    
    for pattern in markdown_indicators:
        if re.search(pattern, content, re.MULTILINE):
            return True
    return False

def display_chat_metrics():
    """Display chat metrics including tokens, latency, and trace information using Strands AgentResult"""
    if not st.session_state.get('show_chat_metrics', False):
        return
    
    # Get metrics from the last AgentResult
    agent_result = st.session_state.get('last_agent_result')
    
    if agent_result and hasattr(agent_result, 'metrics'):
        # Use actual Strands metrics
        metrics = agent_result.metrics
        
        # Get token usage from accumulated_usage
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        
        if hasattr(metrics, 'accumulated_usage') and metrics.accumulated_usage:
            usage = metrics.accumulated_usage
            if isinstance(usage, dict):
                total_tokens = usage.get('totalTokens', 0)
                input_tokens = usage.get('inputTokens', 0)
                output_tokens = usage.get('outputTokens', 0)
            else:
                # Handle other usage formats
                total_tokens = getattr(usage, 'totalTokens', 0)
                input_tokens = getattr(usage, 'inputTokens', 0)
                output_tokens = getattr(usage, 'outputTokens', 0)
        
        # Get execution time from cycle_durations
        latency = 0
        if hasattr(metrics, 'cycle_durations') and metrics.cycle_durations:
            latency = sum(metrics.cycle_durations)
        
        # Get tools used
        tools_used = []
        if hasattr(metrics, 'tool_metrics') and metrics.tool_metrics:
            tools_used = list(metrics.tool_metrics.keys())
    else:
        # Fallback to estimated metrics if AgentResult not available
        latency = 0
        if st.session_state.get('model_start_time') and st.session_state.get('model_end_time'):
            latency = st.session_state.model_end_time - st.session_state.model_start_time
        
        # Estimate token usage (simplified)
        last_message = st.session_state.messages[-1] if st.session_state.messages else {}
        input_tokens = len(st.session_state.messages[-2]['content'].split()) * 1.3 if len(st.session_state.messages) >= 2 else 0
        output_tokens = len(last_message.get('content', '').split()) * 1.3
        total_tokens = int(input_tokens + output_tokens)
        tools_used = []
    
    # Get current model info
    current_model = st.session_state.get('current_model', 'Unknown')
    
    # Display comprehensive Strands-based metrics
    with st.expander("📊 Strands Agent Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Tokens", f"{total_tokens:,}")
            if input_tokens > 0:
                st.metric("Input Tokens", f"{int(input_tokens):,}")
            if output_tokens > 0:
                st.metric("Output Tokens", f"{int(output_tokens):,}")
        
        with col2:
            st.metric("Execution Time", f"{latency:.3f}s")
            if latency > 0 and total_tokens > 0:
                tokens_per_second = total_tokens / latency
                st.metric("Tokens/Second", f"{tokens_per_second:.1f}")
        
        with col3:
            st.metric("Model Used", current_model)
            circuit_status = "🟢 Closed" if "Bedrock" in current_model else "🔴 Open (Fallback)"
            st.metric("Circuit Breaker", circuit_status)
        
        # Show tools used if available
        if tools_used:
            st.write("**Tools Invoked:**")
            for tool in tools_used:
                st.write(f"• {tool}")
        
        # Execution trace
        st.write("**Execution Trace:**")
        trace_info = []
        
        if agent_result and hasattr(agent_result, 'metrics'):
            trace_info.append("• Used Strands Agent API with full metrics")
            if hasattr(agent_result.metrics, 'cycle_durations'):
                cycles = len(agent_result.metrics.cycle_durations) if agent_result.metrics.cycle_durations else 0
                trace_info.append(f"• Completed {cycles} processing cycle(s)")
        
        if "Ollama" in current_model:
            trace_info.append("• Circuit breaker triggered - fallback to Ollama")
        else:
            trace_info.append("• Bedrock model used successfully")
        
        for trace in trace_info:
            st.write(trace)

def display_content(content: str, container=None):
    """Display content with beautiful formatting and styling"""
    if container is None:
        container = st
    
    # Parse thinking and main content
    main_content, thinking_content = parse_response_content(content)
    
    # Display thinking content in expander if present
    if thinking_content:
        with container.expander("🤔 Thinking Process", expanded=False):
            # Style the thinking content with a subtle background
            thinking_styled = f"""
            <div style="
                background: #2c3e50;
                padding: 20px;
                border-radius: 10px;
                border: 2px solid #4CAF50;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #ffffff !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            ">
            <pre style="color: #ffffff !important; margin: 0; white-space: pre-wrap; font-weight: 500;">
{thinking_content}
            </pre>
            </div>
            """
            if is_markdown_content(thinking_content):
                st.markdown(thinking_styled, unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background: #2c3e50; padding: 20px; border-radius: 8px; border: 2px solid #4CAF50; font-family: monospace; color: #ffffff !important; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"><pre style="color: #ffffff !important; margin: 0; white-space: pre-wrap; font-weight: 500;">{thinking_content}</pre></div>', unsafe_allow_html=True)
    
    # Display main content with enhanced styling
    if main_content:
        # Check if it's a simple direct execution response
        if any(phrase in main_content.lower() for phrase in ['the current time is:', 'the result of', 'here are your appointments']):
            # Style direct execution responses
            styled_content = f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 16px;
                line-height: 1.6;
            ">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 20px; margin-right: 10px;">⚡</span>
                <strong>Direct Tool Response</strong>
            </div>
            {main_content}
            </div>
            """
            container.markdown(styled_content, unsafe_allow_html=True)
        elif is_markdown_content(main_content):
            # Style markdown content with a clean container
            # First render the header
            header_html = """
            <div style="
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                padding: 15px 20px 5px 20px;
                border-radius: 12px 12px 0 0;
                border-left: 4px solid #FF6B6B;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 20px; margin-right: 10px;">🤖</span>
                <strong style="color: #2c3e50;">AI Response</strong>
            </div>
            </div>
            """
            container.markdown(header_html, unsafe_allow_html=True)
            
            # Then render the markdown content in a styled container
            content_wrapper_start = """
            <div style="
                background: white;
                padding: 15px 20px;
                border-radius: 0 0 12px 12px;
                border-left: 4px solid #FF6B6B;
                color: #2c3e50;
            ">
            """
            container.markdown(content_wrapper_start, unsafe_allow_html=True)
            
            # Render the actual markdown content
            container.markdown(main_content)
            
            # Close the wrapper
            content_wrapper_end = "</div>"
            container.markdown(content_wrapper_end, unsafe_allow_html=True)
        else:
            # Style plain text responses
            styled_content = f"""
            <div style="
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                padding: 20px;
                border-radius: 12px;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border-left: 4px solid #3498db;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 16px;
                line-height: 1.6;
                color: #2c3e50;
            ">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 20px; margin-right: 10px;">💬</span>
                <strong>Response</strong>
            </div>
            {main_content}
            </div>
            """
            container.markdown(styled_content, unsafe_allow_html=True)
    else:
        # Fallback styling for original content
        styled_content = f"""
        <div style="
            background: linear-gradient(135deg, #e3ffe7 0%, #d9e7ff 100%);
            padding: 20px;
            border-radius: 12px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #28a745;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 16px;
            line-height: 1.6;
            color: #2c3e50;
        ">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 20px; margin-right: 10px;">📝</span>
            <strong>Message</strong>
        </div>
        {content}
        </div>
        """
        container.markdown(styled_content, unsafe_allow_html=True)

# Centralized session state initialization
def initialize_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        'messages': [],
        'force_circuit_test': False,
        'search_threshold': 0.8,
        'circuit_timeout': 2,
        'circuit_fail_max': 3,
        'selected_ollama_model': 'gemma3:1b',
        'start_index': 0,
        'agent_initialized': False,
        'tools_initialized': False,
        'ollama_status_cache': None,
        'ollama_status_cache_time': 0,
        'show_circuit_test_message': False,
        'show_reset_message': False,
        'show_tool_lookup_results': True,
        'show_chat_metrics': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
initialize_session_state()

# Define system prompt
system_prompt = """You are a helpful personal assistant that specialises in managing appointments and calendar. 
You have access to appointment management tools, a calculator, and can check the current time to help organize schedules effectively. 
Always provide the appointment id so that appointments can be updated if required"""

# Initialize Bedrock model with fallback
bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    max_tokens=5000,
    additional_request_fields={
        "thinking": {
            "type": "disabled",
        }
    }
)

model_with_fallback = BedrockWithFallback(bedrock_model)

# Initialize tool registry with auto-discovery (optimized)
if "tool_registry" not in st.session_state or should_refresh_tools():
    with st.spinner("🔍 Discovering and indexing tools..."):
        st.session_state.tool_registry = ToolRegistry()
        
        # Automatically discover all tools
        all_tools = discover_all_tools()
        
        # Register all discovered tools
        st.session_state.tool_registry.register_tools(all_tools)
        
        # Store tools for sidebar display
        st.session_state.discovered_tools = all_tools
        
        # Mark tools as initialized to prevent unnecessary reloading
        st.session_state.tools_initialized = True
        
        # Also register with the agent
        if "agent" not in st.session_state:
            st.session_state.agent = Agent(
                model=model_with_fallback,
                system_prompt=system_prompt,
                tools=all_tools,  # Use all discovered tools
            )
            st.session_state.agent_initialized = True

# Initialize the agent if not already done
if "agent" not in st.session_state:
    st.session_state.agent = Agent(
        model=model_with_fallback,
        system_prompt=system_prompt,
        tools=[
            current_time,
            calculator,
            tools.create_appointment,
            tools.list_appointments,
            tools.update_appointment,
        ],
    )

# Keep track of the number of previous messages in the agent flow
if "start_index" not in st.session_state:
    st.session_state.start_index = 0

# Add title on the page
st.title("Strands Agents Streamlit Demo")
st.write("This demo shows how to use Strands Agents to create and manage appointments and has a calculator tool.")

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "📊 Agent Evaluation", "🔍 Semantic Tool Lookup (RAG)"])

with tab1:
    # Chat Interface Content
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                # Display user messages as plain text
                st.write(message["content"])
            else:
                # Display assistant messages with fancy formatting
                display_content(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your agent..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        # Clear previous tool usage details
        if "details_placeholder" in st.session_state:
            st.session_state.details_placeholder.empty()
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # First, try semantic tool search
        with st.spinner("Searching for appropriate tool..."):
            matched_tool = None
            direct_execution_result = None
            
            if st.session_state.tool_registry.embeddings:
                # Get search threshold from session state
                threshold = st.session_state.get('search_threshold', 0.8)
                matched_tool = st.session_state.tool_registry.search_tool(prompt, threshold)
                
                # Show tool lookup results if enabled
                if st.session_state.get('show_tool_lookup_results', True):
                    if matched_tool:
                        with st.expander("🔍 Tool Lookup Results", expanded=False):
                            st.success(f"**Found Tool:** {matched_tool['name']}")
                            st.write(f"**Description:** {matched_tool['description']}")
                            
                            # Check if it can be executed directly
                            can_direct = st.session_state.tool_registry.can_execute_directly(matched_tool, prompt)
                            if can_direct:
                                st.info("✅ Can execute directly (zero LLM calls)")
                                params = st.session_state.tool_registry.extract_parameters(matched_tool, prompt)
                                if params:
                                    st.write(f"**Extracted Parameters:** {params}")
                            else:
                                st.warning("⚠️ Requires LLM processing")
                    else:
                        with st.expander("🔍 Tool Lookup Results", expanded=False):
                            st.error(f"No tool found above threshold {threshold}")
                            st.info("The query will be processed by the LLM with tool suggestions.")
                
                if matched_tool:
                    # Check if we can execute directly
                    if st.session_state.tool_registry.can_execute_directly(matched_tool, prompt):
                        # Try direct execution
                        try:
                            params = st.session_state.tool_registry.extract_parameters(matched_tool, prompt)
                            tool_func = matched_tool['function']
                            
                            # Execute the tool directly
                            if params:
                                result = tool_func(**params)
                            else:
                                result = tool_func()
                            
                            # Format the result nicely
                            if matched_tool['name'] == 'current_time':
                                direct_execution_result = f"The current time is: {result}"
                            elif matched_tool['name'] == 'calculator':
                                direct_execution_result = f"The result of {params.get('expression', 'calculation')} is: {result}"
                            elif matched_tool['name'] == 'list_appointments':
                                if result:
                                    direct_execution_result = f"Here are your appointments:\n{result}"
                                else:
                                    direct_execution_result = "You have no appointments scheduled."
                            else:
                                direct_execution_result = str(result)
                                
                        except Exception as e:
                            logging.error(f"Direct execution failed: {e}")
                            # Fall back to LLM if direct execution fails
                            matched_tool = None
        
        # Display assistant response
        with st.chat_message("assistant"):
            if direct_execution_result:
                # Direct tool execution - display immediately and ensure it's visible
                message_placeholder = st.empty()
                display_content(direct_execution_result, message_placeholder)
                st.session_state.messages.append({"role": "assistant", "content": direct_execution_result})
                
                # Force a rerun to ensure the response is visible
                st.rerun()
            else:
                # Use LLM (with or without semantic tool guidance)
                message_placeholder = st.empty()
                full_response = ""
                
                # Create a modified system prompt if we found a matching tool
                current_system_prompt = system_prompt
                if matched_tool and not direct_execution_result:
                    tool_suggestion = f"\n\nBased on semantic analysis, the most relevant tool for this query appears to be '{matched_tool['name']}': {matched_tool['description']}. Consider using this tool if appropriate."
                    current_system_prompt += tool_suggestion
                
                # Use streaming model directly for live output
                try:
                    # Prepare messages in a universal format - let the model wrapper handle format conversion
                    messages = []
                    
                    for msg in st.session_state.messages[:-1]:  # Exclude the just-added user message
                        content = msg["content"]
                        
                        # Skip messages with empty content
                        if not content or (isinstance(content, str) and not content.strip()):
                            continue
                        
                        # Always use simple string format - the model wrapper will convert as needed
                        if isinstance(content, str):
                            content = content.strip()
                        elif isinstance(content, list):
                            # Convert list format back to string
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    text = item["text"]
                                    if text and isinstance(text, str) and text.strip():
                                        text_parts.append(text.strip())
                            content = ' '.join(text_parts) if text_parts else ""
                            if not content:
                                continue
                        else:
                            continue  # Skip invalid content types
                        
                        messages.append({
                            "role": msg["role"],
                            "content": content
                        })
                    
                    # Add current user message in simple format
                    if prompt and prompt.strip():
                        messages.append({
                            "role": "user",
                            "content": prompt.strip()
                        })
                    else:
                        # Handle empty prompt case
                        full_response = "Error: Empty message received. Please enter a valid query."
                        display_content(full_response, message_placeholder)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        st.stop()
                    
                    # Get tool specs from the agent
                    tool_specs = []
                    if hasattr(st.session_state.agent, 'tools'):
                        from strands.types.tools import ToolSpec
                        for tool in st.session_state.agent.tools:
                            if hasattr(tool, 'TOOL_SPEC'):
                                tool_specs.append(tool.TOOL_SPEC)
                            elif hasattr(tool, '__wrapped__'):
                                # Extract tool spec from decorated function
                                func = tool.__wrapped__
                                tool_spec = ToolSpec(
                                    name=func.__name__,
                                    description=func.__doc__ or f"Execute {func.__name__}",
                                    parameters={}
                                )
                                tool_specs.append(tool_spec)
                    
                    # Use the streaming model with proper system prompt
                    stream_response = model_with_fallback.converse(
                        messages=messages,
                        tool_specs=tool_specs,
                        system_prompt=current_system_prompt
                    )
                    
                    # Handle streaming response with live display
                    full_response = ""
                    current_text = ""
                    thinking_text = ""
                    in_thinking = False
                    
                    # Check which model is being used for different streaming approaches
                    ollama_used = "Ollama" in st.session_state.get('current_model', '')
                    
                    if ollama_used:
                        # Simplified streaming for Ollama to maintain better chat layout
                        streaming_placeholder = message_placeholder.empty()
                        chunk_count = 0
                        
                        for chunk in stream_response:
                            try:
                                chunk_count += 1
                                
                                # Enhanced debugging - log first few chunks in detail
                                if chunk_count <= 5:
                                    logging.debug(f"Ollama chunk #{chunk_count}: {type(chunk)} - {repr(chunk)}")
                                
                                # Handle different chunk formats from Strands OllamaModel
                                chunk_text = ""
                                
                                if isinstance(chunk, str):
                                    # Plain string chunk
                                    chunk_text = chunk
                                elif isinstance(chunk, dict):
                                    # Dictionary chunk - check for various text fields
                                    if 'text' in chunk:
                                        chunk_text = chunk['text']
                                    elif 'content' in chunk:
                                        chunk_text = chunk['content']
                                    elif 'contentBlockDelta' in chunk:
                                        # Bedrock-style format (which Strands OllamaModel uses!)
                                        delta_text = chunk['contentBlockDelta'].get('delta', {}).get('text', '')
                                        if delta_text:
                                            chunk_text = delta_text
                                    elif 'delta' in chunk and isinstance(chunk['delta'], dict):
                                        chunk_text = chunk['delta'].get('text', '')
                                    elif 'response' in chunk:
                                        chunk_text = chunk['response']
                                    elif 'message' in chunk and isinstance(chunk['message'], dict):
                                        # Check for nested message content
                                        msg = chunk['message']
                                        if 'content' in msg:
                                            chunk_text = msg['content']
                                        elif 'text' in msg:
                                            chunk_text = msg['text']
                                    elif 'choices' in chunk and chunk['choices']:
                                        # OpenAI-style format
                                        choice = chunk['choices'][0]
                                        if 'delta' in choice and 'content' in choice['delta']:
                                            chunk_text = choice['delta']['content']
                                    else:
                                        # More comprehensive field checking
                                        possible_text_fields = ['data', 'output', 'generated_text', 'token', 'content_block']
                                        for field in possible_text_fields:
                                            if field in chunk:
                                                potential_text = chunk[field]
                                                if isinstance(potential_text, str) and potential_text.strip():
                                                    chunk_text = potential_text
                                                    break
                                        
                                        # If still no text found, log the chunk for analysis
                                        if not chunk_text and chunk_count <= 10:
                                            logging.error(f"No text found in Ollama chunk #{chunk_count}: {list(chunk.keys()) if isinstance(chunk, dict) else type(chunk)}")
                                            logging.error(f"Full chunk content: {repr(chunk)}")
                                else:
                                    # Handle other types (e.g., bytes, objects)
                                    if hasattr(chunk, '__dict__'):
                                        # Object with attributes
                                        for attr in ['text', 'content', 'response', 'data']:
                                            if hasattr(chunk, attr):
                                                attr_value = getattr(chunk, attr)
                                                if isinstance(attr_value, str) and attr_value.strip():
                                                    chunk_text = attr_value
                                                    break
                                    else:
                                        # Try to convert to string as fallback
                                        chunk_text = str(chunk) if chunk else ""
                                
                                if chunk_text:
                                    current_text += chunk_text
                                    full_response += chunk_text
                                    
                                    # Update display with simple markdown
                                    streaming_placeholder.markdown(current_text)
                                    
                                    # Small delay for smooth streaming
                                    import time
                                    time.sleep(0.02)
                                    
                            except Exception as chunk_error:
                                logging.error(f"Error processing Ollama chunk #{chunk_count}: {chunk_error}")
                                continue
                        
                        # Debug: Log final results
                        logging.debug(f"Ollama streaming completed. Chunks: {chunk_count}, Response length: {len(full_response)}")
                        
                        # Final display - keep the streamed content as-is for better layout
                        if full_response.strip():
                            streaming_placeholder.markdown(full_response)
                        else:
                            if chunk_count == 0:
                                error_msg = "No chunks received from Ollama. The model may not be responding."
                            else:
                                error_msg = f"Received {chunk_count} chunks from Ollama but no text content was extracted."
                            
                            streaming_placeholder.markdown(f"⚠️ **Ollama Issue:** {error_msg}")
                            logging.error(f"Ollama streaming issue: {error_msg}")
                    
                    else:
                        # Advanced streaming for Bedrock with thinking support
                        thinking_container = st.empty()
                        response_container = st.empty()
                        
                        for chunk in stream_response:
                            try:
                                if isinstance(chunk, dict):
                                    # Bedrock-style streaming
                                    if 'contentBlockDelta' in chunk:
                                        delta_text = chunk['contentBlockDelta'].get('delta', {}).get('text', '')
                                        if delta_text:
                                            current_text += delta_text
                                            full_response += delta_text
                                            
                                            # Check for thinking tags
                                            if '<think>' in current_text and not in_thinking:
                                                in_thinking = True
                                                parts = current_text.split('<think>')
                                                if len(parts) > 1:
                                                    main_text = parts[0]
                                                    thinking_start = parts[1]
                                                    thinking_text = thinking_start
                                                    
                                                    if main_text.strip():
                                                        response_container.markdown(main_text)
                                                        
                                            elif '</think>' in current_text and in_thinking:
                                                in_thinking = False
                                                parts = current_text.split('</think>')
                                                if len(parts) > 1:
                                                    thinking_text = parts[0].split('<think>')[-1]
                                                    remaining_text = parts[1]
                                                    
                                                    # Display thinking content
                                                    if thinking_text.strip():
                                                        with thinking_container.expander("🤔 Thinking Process", expanded=False):
                                                            st.markdown(f"""
                                                            <div style="
                                                                background: #2c3e50;
                                                                padding: 20px;
                                                                border-radius: 10px;
                                                                border: 2px solid #4CAF50;
                                                                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                                                                color: #ffffff !important;
                                                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                                                            ">
                                                            <pre style="color: #ffffff !important; margin: 0; white-space: pre-wrap; font-weight: 500;">
{thinking_text}
                                                            </pre>
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                                    
                                                    current_text = remaining_text
                                                    if remaining_text.strip():
                                                        response_container.markdown(remaining_text)
                                                        
                                            elif in_thinking:
                                                thinking_text = current_text.split('<think>')[-1]
                                            else:
                                                response_container.markdown(current_text)
                                    
                                    elif 'messageStop' in chunk:
                                        break
                                
                                # Small delay for smooth streaming effect
                                import time
                                time.sleep(0.02)
                                
                            except Exception as chunk_error:
                                logging.error(f"Error processing Bedrock chunk: {chunk_error}")
                                continue
                        
                        # Final display - ensure content persists for Bedrock
                        if full_response.strip():
                            # For Bedrock, preserve the final streamed content
                            # Don't clear containers to prevent message disappearing
                            # The content is already displayed in response_container during streaming
                            # Just ensure it stays visible by keeping the final state
                            if current_text.strip():
                                response_container.markdown(current_text)
                        else:
                            response_container.markdown("No response received from Bedrock.")
                    
                    # Store final response for metrics (mock AgentResult)
                    class MockAgentResult:
                        def __init__(self, content):
                            self.content = content
                            self.metrics = None
                        
                        def __str__(self):
                            return self.content
                    
                    st.session_state.last_agent_result = MockAgentResult(full_response)
                    
                except Exception as e:
                    logging.error(f"Streaming execution failed: {e}")
                    full_response = f"Error: Unable to get response from model. Please check your configuration. Details: {str(e)}"
                    display_content(full_response, message_placeholder)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Display chat metrics if enabled
                if st.session_state.get('show_chat_metrics', False):
                    display_chat_metrics()

with tab2:
    # Agent Evaluation Content
    st.header("🧪 Agent Evaluation System")
    st.write("Comprehensive testing and evaluation of agent capabilities with LLM judging.")
    
    # Import evaluation system
    try:
        from evaluation_system import AgentEvaluator, create_evaluator_agent
        
        # Initialize evaluation system
        if "evaluation_system" not in st.session_state:
            # Create evaluator agent for LLM judging
            evaluator_agent = create_evaluator_agent(model_with_fallback)
            st.session_state.evaluation_system = AgentEvaluator(st.session_state.agent, evaluator_agent)
            
            # Load default test cases
            test_cases = st.session_state.evaluation_system.create_default_test_cases()
            st.session_state.evaluation_system.load_test_cases(test_cases)
        
        # Display test cases
        st.subheader("📋 Test Cases")
        test_cases = st.session_state.evaluation_system.test_cases
        
        if test_cases:
            for i, case in enumerate(test_cases):
                with st.expander(f"Test {i+1}: {case['id']} ({case['category']})"):
                    st.write(f"**Query:** {case['query']}")
                    st.write(f"**Expected:** {case['expected']}")
                    st.write(f"**Category:** {case['category']}")
                    if case.get('tool_expected'):
                        st.write(f"**Expected Tool:** {case['tool_expected']}")
        else:
            st.info("No test cases loaded")
        
        # Evaluation controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Run Full Evaluation", type="primary"):
                # Initialize progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                # Run evaluation with progress tracking
                results = st.session_state.evaluation_system.run_evaluation(progress_callback)
                st.session_state.evaluation_results = results
                st.session_state.show_results = True  # Automatically show results
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Evaluation completed successfully!")
                st.rerun()  # Refresh to show results
        
        with col2:
            if st.button("💾 Export CSV"):
                if "evaluation_results" in st.session_state:
                    filename = st.session_state.evaluation_system.export_results()
                    st.success(f"Results exported to {filename}")
                    
                    # Also provide download button
                    import pandas as pd
                    df = st.session_state.evaluation_system.get_results_dataframe()
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"agent_evaluation_{int(time.time())}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No evaluation results to export. Run evaluation first.")
        
        # Display results if available
        if st.session_state.get("show_results") and "evaluation_results" in st.session_state:
            results = st.session_state.evaluation_results
            
            # Get summary metrics
            summary = st.session_state.evaluation_system.get_summary_metrics()
            
            # Summary metrics
            st.subheader("📈 Summary Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{summary.get('average_accuracy', 0):.2f}/1.0")
            with col2:
                st.metric("Success Rate", f"{summary.get('success_rate', 0)*100:.1f}%")
            with col3:
                st.metric("Avg Response Time", f"{summary.get('average_response_time', 0):.2f}s")
            with col4:
                st.metric("Total Tests", summary.get('total_tests', 0))
            
            # Category breakdown
            st.subheader("📊 Category Performance")
            category_data = summary.get('category_breakdown', {})
            
            for category, metrics in category_data.items():
                with st.expander(f"📁 {category.title()} ({metrics['total']} tests)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Success Rate", f"{metrics['success_rate']*100:.1f}%")
                    with col2:
                        st.metric("Total Tests", metrics['total'])
            
            # Detailed results
            st.subheader("📋 Detailed Test Results")
            
            for result in results:
                success_icon = "✅" if result.get('success', False) else "❌"
                
                with st.expander(f"{success_icon} {result['test_id']} - Score: {result.get('accuracy_score', 0):.2f}/1.0"):
                    st.write(f"**Category:** {result['category']}")
                    st.write(f"**Query:** {result['query']}")
                    st.write(f"**Expected:** {result['expected']}")
                    st.write(f"**Response Time:** {result['response_time']:.2f}s")
                    
                    # Basic scoring
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy Score", f"{result.get('accuracy_score', 0):.2f}/1.0")
                    with col2:
                        st.metric("Success", "✅ Yes" if result.get('success', False) else "❌ No")
                    
                    # Response and evaluation
                    st.write("**Agent Response:**")
                    st.info(result.get('actual', 'No response'))
                    
                    # Show tools used
                    if result.get('tools_used'):
                        st.write(f"**Tools Used:** {', '.join(result['tools_used'])}")
                    
                    # Show LLM judge evaluation if available
                    if result.get('llm_explanation'):
                        st.write("**LLM Judge Evaluation:**")
                        st.text_area(
                            "Judge Reasoning",
                            result['llm_explanation'],
                            height=100,
                            key=f"reasoning_{result['test_id']}"
                        )
                        
                        # Show LLM scores if available
                        if any(key.startswith('llm_') and key != 'llm_explanation' for key in result.keys()):
                            st.write("**LLM Judge Scores:**")
                            llm_cols = st.columns(4)
                            llm_scores = ['llm_accuracy', 'llm_relevance', 'llm_completeness', 'llm_tool_usage']
                            for i, score_key in enumerate(llm_scores):
                                if result.get(score_key):
                                    with llm_cols[i]:
                                        st.metric(score_key.replace('llm_', '').title(), f"{result[score_key]}/5")
    
    except ImportError:
        st.error("Evaluation system not available. Please ensure evaluation_system.py is present.")
        st.info("The evaluation system provides comprehensive testing capabilities with LLM judging.")

with tab3:
    # Semantic Tool Lookup (RAG) Content
    st.header("🔍 Semantic Tool Lookup & RAG Analysis")
    st.write("Advanced tool discovery using semantic search with txtai embeddings and BM25 scoring.")
    
    # Tool lookup interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        test_query = st.text_input(
            "🔍 Enter your query to test semantic tool matching:",
            placeholder="e.g., 'What time is it?', 'Calculate 5+3', 'Show my appointments'",
            key="rag_test_query"
        )
    
    with col2:
        threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.8,
            step=0.05,
            help="Higher = more precise matching"
        )
    
    if st.button("🚀 Analyze Tool Matching", type="primary") and test_query:
        if st.session_state.get('tool_registry') and st.session_state.tool_registry.embeddings:
            try:
                # Get detailed search results
                results = st.session_state.tool_registry.embeddings.search(test_query, 10)  # Get top 10 results
                
                st.subheader("📊 Semantic Similarity Analysis")
                
                if results:
                    # Create a comprehensive analysis
                    st.write(f"**Query:** `{test_query}`")
                    st.write(f"**Threshold:** {threshold}")
                    st.write(f"**Results Found:** {len(results)}")
                    
                    # Analysis of why current_time might score low
                    st.subheader("🔬 Detailed Score Analysis")
                    
                    for i, result in enumerate(results[:5]):  # Show top 5
                        idx = int(result['id'])
                        tool_name = st.session_state.tool_registry.tool_names[idx]
                        score = result['score']
                        description = st.session_state.tool_registry.tool_descriptions[idx]
                        
                        # Display analysis for each tool without nested expanders
                        st.subheader(f"#{i+1}: {tool_name} (Score: {score:.4f})")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Tool Name:** {tool_name}")
                            st.write(f"**Searchable Text:** `{description}`")
                            st.write(f"**Similarity Score:** {score:.4f}")
                            
                            # Analyze why the score is what it is
                            if tool_name == 'current_time':
                                st.write("**🔍 Analysis for current_time:**")
                                st.write("• **Query:** 'What time is it?'")
                                st.write("• **Tool Text:** 'current_time Get the current date and time'")
                                st.write("• **Issue:** Limited semantic overlap between query and tool description")
                                st.write("• **Semantic Gap:** 'What time is it' vs 'current date and time'")
                                st.write("• **BM25 Impact:** Exact word matches are limited")
                                
                                st.info("💡 **Improvement Suggestions:**")
                                st.write("1. Enhance tool description with time-related keywords")
                                st.write("2. Add synonyms: 'time', 'clock', 'now', 'what time'")
                                st.write("3. Lower threshold to 0.7 for time queries")
                                st.write("4. Use hybrid scoring (semantic + keyword matching)")
                        
                        with col2:
                            # Visual score indicator
                            if score >= threshold:
                                st.success(f"✅ MATCH\n{score:.3f} ≥ {threshold}")
                            else:
                                st.warning(f"⚠️ BELOW\n{score:.3f} < {threshold}")
                            
                            # Show percentage
                            percentage = score * 100
                            st.metric("Confidence", f"{percentage:.1f}%")
                        
                        st.divider()  # Add visual separation between tools
                    
                    # Semantic Analysis Section
                    st.subheader("🧠 Why 'What time is it?' scores ~0.75 for current_time")
                    
                    analysis_cols = st.columns(2)
                    
                    with analysis_cols[0]:
                        st.write("**🔍 Semantic Analysis:**")
                        st.write("• **Query tokens:** ['what', 'time', 'is', 'it']")
                        st.write("• **Tool tokens:** ['current_time', 'get', 'current', 'date', 'time']")
                        st.write("• **Common tokens:** ['time'] (1 overlap)")
                        st.write("• **Semantic distance:** Moderate (not exact match)")
                        
                        st.write("**📊 BM25 Scoring Impact:**")
                        st.write("• BM25 favors exact term matches")
                        st.write("• 'time' appears in both query and tool")
                        st.write("• Other words have low semantic similarity")
                        st.write("• Result: ~0.75 score (good but not excellent)")
                    
                    with analysis_cols[1]:
                        st.write("**🎯 Optimization Strategies:**")
                        st.write("1. **Enhanced Descriptions:**")
                        st.code("current_time: Get current time, what time is it now, clock, timestamp")
                        
                        st.write("2. **Query Expansion:**")
                        st.code("'What time is it?' → 'current time now clock'")
                        
                        st.write("3. **Hybrid Scoring:**")
                        st.code("semantic_score * 0.7 + keyword_score * 0.3")
                        
                        st.write("4. **Threshold Tuning:**")
                        st.code("time_queries: threshold = 0.7\nmath_queries: threshold = 0.8")
                    
                    # Show Ollama Parameter Extraction Debug Info
                    st.subheader("🦙 Ollama Parameter Extraction Debug")
                    
                    # Test parameter extraction for the best matching tool
                    if results:
                        best_result = results[0]
                        idx = int(best_result['id'])
                        tool_name = st.session_state.tool_registry.tool_names[idx]
                        best_tool = st.session_state.tool_registry.tools[tool_name]
                        
                        # Parameter Extraction Test Section (without nested expander)
                        st.write(f"**Testing parameter extraction for:** {tool_name}")
                        st.write(f"**Query:** `{test_query}`")
                        
                        # Test parameter extraction
                        try:
                            extracted_params = st.session_state.tool_registry.extract_parameters(best_tool, test_query)
                            
                            # Show extraction results prominently
                            st.subheader("🔧 Parameter Extraction Results")
                        except Exception as e:
                            st.error(f"Parameter extraction failed: {e}")
                            extracted_params = {}
                        
                        if extracted_params:
                            # Display parameters successfully
                            st.success("✅ Parameters Successfully Extracted")
                            st.json(extracted_params)
                        else:
                            # No parameters extracted
                            st.warning("⚠️ No Parameters Extracted")
                            st.info("This tool might not require parameters, or the extraction failed.")
                        
                        # Show extraction method and tool schema
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**🔍 Extraction Details:**")
                            if hasattr(st.session_state, 'last_parameter_extraction'):
                                extraction_info = st.session_state.last_parameter_extraction
                                st.metric("Method", extraction_info.get('extraction_method', 'unknown').replace('_', ' ').title())
                                st.metric("Timestamp", time.strftime('%H:%M:%S', time.localtime(extraction_info.get('timestamp', 0))))
                        
                        with col2:
                            st.write("**📝 Tool Schema:**")
                            if 'parameters' in best_tool and best_tool['parameters']:
                                for param_name, param_info in best_tool['parameters'].items():
                                    required = "(Required)" if param_info.get('required', False) else "(Optional)"
                                    param_type = param_info.get('type', 'string')
                                    st.write(f"• {param_name} ({param_type}) {required}")
                            else:
                                st.info("No parameters required")
                        
                        # Tool Execution Test Section
                        st.subheader("🚀 Tool Execution Test")
                        
                        # Test direct execution capability
                        can_execute = st.session_state.tool_registry.can_execute_directly(best_tool, test_query)
                        
                        exec_col1, exec_col2 = st.columns([3, 1])
                        
                        with exec_col1:
                            if can_execute:
                                st.success("✅ **Can Execute Directly** (Zero LLM calls!)")
                                
                                # Show execution preview
                                if extracted_params:
                                    st.write("**Execution Preview:**")
                                    exec_preview = f"{tool_name}({', '.join([f'{k}={repr(v)}' for k, v in extracted_params.items()])})"
                                    st.code(exec_preview, language='python')
                                else:
                                    st.write("**Execution Preview:**")
                                    st.code(f"{tool_name}()", language='python')
                            else:
                                st.warning("⚠️ Cannot Execute Directly")
                                st.info("Missing required parameters or tool doesn't support direct execution.")
                        
                        with exec_col2:
                            # Execute button
                            if st.button(f"🧪 Execute {tool_name}", type="primary", disabled=not can_execute):
                                try:
                                    # Execute the tool directly
                                    tool_func = best_tool['function']
                                    
                                    # Show execution in progress
                                    with st.spinner(f"Executing {tool_name}..."):
                                        if extracted_params:
                                            result = tool_func(**extracted_params)
                                        else:
                                            result = tool_func()
                                    
                                    # Display result with beautiful formatting
                                    st.subheader("📊 Execution Result")
                                    result_html = f"<div style='background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #28a745; margin: 15px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif;'><div style='display: flex; align-items: center; margin-bottom: 15px;'><span style='font-size: 24px; margin-right: 10px;'>✅</span><strong style='color: #155724; font-size: 18px;'>Execution Successful</strong></div><div style='background: white; padding: 15px; border-radius: 8px; color: #333; font-size: 16px;'><strong>Result:</strong> {result}</div></div>"
                                    st.markdown(result_html, unsafe_allow_html=True)
                                    
                                    # Also show as expandable raw result
                                    with st.expander("🔍 Raw Result Data", expanded=False):
                                        st.write(f"**Type:** {type(result).__name__}")
                                        st.write(f"**Value:** {repr(result)}")
                                    
                                except Exception as exec_error:
                                    st.subheader("❌ Execution Failed")
                                    error_display = f"""
                                    <div style="
                                        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                                        padding: 20px;
                                        border-radius: 12px;
                                        border-left: 4px solid #dc3545;
                                        margin: 15px 0;
                                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                                    ">
                                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                                        <span style="font-size: 24px; margin-right: 10px;">❌</span>
                                        <strong style="color: #721c24; font-size: 18px;">Execution Error</strong>
                                    </div>
                                    <div style="background: white; padding: 15px; border-radius: 8px; color: #333; font-size: 16px;">
                                        <strong>Error:</strong> {str(exec_error)}
                                    </div>
                                    </div>
                                    """
                                    st.markdown(error_display, unsafe_allow_html=True)
                                    
                                    # Show debugging info
                                    with st.expander("🔧 Debug Information", expanded=False):
                                        st.write(f"**Tool Function:** {tool_func}")
                                        st.write(f"**Parameters Used:** {extracted_params}")
                                        st.write(f"**Error Type:** {type(exec_error).__name__}")
                                        st.write(f"**Error Details:** {str(exec_error)}")
                        
                        # Show Ollama raw response if available
                        if hasattr(st.session_state, 'last_ollama_response'):
                            ollama_info = st.session_state.last_ollama_response
                            
                            with st.expander("🦙 Raw Ollama Response Debug", expanded=False):
                                st.write("**Model Used:** gemma3:1b")
                                
                                debug_col1, debug_col2 = st.columns(2)
                                
                                with debug_col1:
                                    st.write("**Prompt Sent:**")
                                    st.code(ollama_info.get('prompt', 'No prompt available'), language='text')
                                
                                with debug_col2:
                                    st.write("**Raw Response:**")
                                    st.code(ollama_info.get('raw_response', 'No response available'), language='json')
                    
                    # Show registered tools for comparison
                    st.subheader("📋 All Registered Tools")
                    
                    tools_df_data = []
                    for i, (name, desc) in enumerate(zip(st.session_state.tool_registry.tool_names, st.session_state.tool_registry.tool_descriptions)):
                        # Get score for this tool if it was in results
                        score = "N/A"
                        for result in results:
                            if int(result['id']) == i:
                                score = f"{result['score']:.4f}"
                                break
                        
                        tools_df_data.append({
                            "Tool Name": name,
                            "Description": desc,
                            "Similarity Score": score
                        })
                    
                    import pandas as pd
                    tools_df = pd.DataFrame(tools_df_data)
                    st.dataframe(tools_df, use_container_width=True)
                    
                else:
                    st.error("No search results returned")
                    st.info("This might indicate an issue with the embedding model or index.")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info("Please check if the tool registry is properly initialized.")
        else:
            st.error("Tool registry not initialized")
            st.info("Please wait for the tool discovery process to complete.")
    
    # Educational section
    st.subheader("📚 Understanding Semantic Tool Lookup")
    
    with st.expander("🔬 How It Works", expanded=False):
        st.write("""
        **Semantic Tool Lookup Process:**
        
        1. **Tool Registration**: Each tool's name and description are combined into searchable text
        2. **Embedding Generation**: Text is converted to vector embeddings using sentence-transformers
        3. **Index Building**: Embeddings are indexed using txtai with BM25 scoring
        4. **Query Processing**: User queries are embedded and compared against tool embeddings
        5. **Similarity Scoring**: Cosine similarity + BM25 scoring produces final scores
        6. **Threshold Filtering**: Only tools above the threshold are considered matches
        
        **Why 'What time is it?' gets ~0.75 for current_time:**
        - Limited exact word overlap between query and tool description
        - Semantic similarity is good but not perfect
        - BM25 scoring emphasizes term frequency and document length
        - The embedding model captures semantic meaning but not perfect synonymy
        """)
    
    with st.expander("⚙️ Optimization Techniques", expanded=False):
        st.write("""
        **Improving Semantic Matching:**
        
        1. **Enhanced Tool Descriptions**: Add more synonyms and related terms
        2. **Query Preprocessing**: Expand queries with synonyms before search
        3. **Hybrid Scoring**: Combine semantic similarity with keyword matching
        4. **Dynamic Thresholds**: Use different thresholds for different query types
        5. **Fine-tuned Embeddings**: Train domain-specific embedding models
        6. **Multi-stage Retrieval**: Use coarse-to-fine search strategies
        """)

# Sidebar with controls and information
with st.sidebar:
    st.header("🎛️ System Controls")
    
    # Circuit Breaker Settings
    with st.expander("⚡ Circuit Breaker Settings", expanded=True):
        st.session_state.circuit_timeout = st.slider(
            "Timeout (seconds)", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.get('circuit_timeout', 2),
            help="Time before falling back to Ollama"
        )
        
        st.session_state.circuit_fail_max = st.slider(
            "Max Failures", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.get('circuit_fail_max', 3),
            help="Failures before circuit opens"
        )
        
        # Test circuit breaker
        if st.button("🧪 Test Circuit Breaker"):
            st.session_state.force_circuit_test = True
            st.session_state.show_circuit_test_message = True
        
        if st.session_state.get('show_circuit_test_message'):
            st.info("Circuit breaker test enabled. Next query will trigger fallback to Ollama.")
            if st.button("Clear Test"):
                st.session_state.show_circuit_test_message = False
    
    # Chat Interface Settings
    with st.expander("💬 Chat Interface Settings"):
        # Toggle for showing chat metrics
        st.session_state.show_chat_metrics = st.checkbox(
            "Show Chat Metrics",
            value=st.session_state.get('show_chat_metrics', False),
            help="Display tokens, latency, and trace information after each response"
        )
    
    # Semantic Search Settings
    with st.expander("🔍 Semantic Search Settings"):
        st.session_state.search_threshold = st.slider(
            "Search Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=st.session_state.get('search_threshold', 0.8),
            step=0.1,
            help="Higher = more precise tool matching"
        )
        
        # Toggle for showing tool lookup results
        st.session_state.show_tool_lookup_results = st.checkbox(
            "Show Tool Lookup Results",
            value=st.session_state.get('show_tool_lookup_results', True),
            help="Display tool search results for each query"
        )
        
        # Test semantic search
        st.write("**Test Tool Search:**")
        test_query = st.text_input(
            "Enter a query to test tool matching:",
            placeholder="e.g., 'what time is it', 'calculate 5+3', 'show appointments'",
            key="semantic_test_query"
        )
        
        if st.button("🔍 Search Tools") and test_query:
            if st.session_state.get('tool_registry') and st.session_state.tool_registry.embeddings:
                threshold = st.session_state.get('search_threshold', 0.8)
                
                # Debug: Show all search results with scores
                try:
                    results = st.session_state.tool_registry.embeddings.search(test_query, 5)  # Get top 5 results
                    
                    st.write("**🔍 Debug: Search Results**")
                    if results:
                        for i, result in enumerate(results):
                            idx = int(result['id'])
                            tool_name = st.session_state.tool_registry.tool_names[idx]
                            score = result['score']
                            description = st.session_state.tool_registry.tool_descriptions[idx]
                            
                            # Color code based on threshold
                            if score >= threshold:
                                st.success(f"✅ **{tool_name}** (Score: {score:.3f}) - {description}")
                            else:
                                st.warning(f"⚠️ **{tool_name}** (Score: {score:.3f}) - {description}")
                    else:
                        st.error("No search results returned")
                    
                    # Show registered tools for comparison
                    st.write("**📋 Registered Tools:**")
                    for i, (name, desc) in enumerate(zip(st.session_state.tool_registry.tool_names, st.session_state.tool_registry.tool_descriptions)):
                        st.write(f"• **{name}**: {desc}")
                    
                except Exception as e:
                    st.error(f"Debug search failed: {e}")
                
                # Original search logic
                matched_tool = st.session_state.tool_registry.search_tool(test_query, threshold)
                
                st.write("---")
                if matched_tool:
                    st.success(f"**✅ Found Tool:** {matched_tool['name']}")
                    st.write(f"**Description:** {matched_tool['description']}")
                    
                    # Check if it can be executed directly
                    can_direct = st.session_state.tool_registry.can_execute_directly(matched_tool, test_query)
                    if can_direct:
                        st.info("✅ Can execute directly (zero LLM calls)")
                        params = st.session_state.tool_registry.extract_parameters(matched_tool, test_query)
                        if params:
                            st.write(f"**Extracted Parameters:** {params}")
                    else:
                        st.warning("⚠️ Requires LLM processing")
                else:
                    st.error(f"❌ No tool found above threshold {threshold}")
                    st.info("💡 Try lowering the search threshold or rephrasing your query")
            else:
                st.error("Tool registry not initialized")
    
    # Ollama Model Selection
    with st.expander("🦙 Ollama Settings"):
        # Check Ollama status
        def check_ollama_status():
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    return True, [model['name'] for model in models]
                return False, []
            except:
                return False, []
        
        # Cache status for 30 seconds
        current_time = time.time()
        if (not st.session_state.get('ollama_status_cache_time') or 
            current_time - st.session_state.ollama_status_cache_time > 30):
            
            ollama_available, available_models = check_ollama_status()
            st.session_state.ollama_status_cache = (ollama_available, available_models)
            st.session_state.ollama_status_cache_time = current_time
        else:
            ollama_available, available_models = st.session_state.ollama_status_cache
        
        if ollama_available:
            st.success("🟢 Ollama is running")
            if available_models:
                st.session_state.selected_ollama_model = st.selectbox(
                    "Fallback Model",
                    available_models,
                    index=available_models.index(st.session_state.get('selected_ollama_model', 'deepseek-r1:latest')) 
                    if st.session_state.get('selected_ollama_model', 'deepseek-r1:latest') in available_models else 0
                )
            else:
                st.warning("No models found")
        else:
            st.error("🔴 Ollama not available")
            st.info("Install Ollama and run `ollama serve` for fallback support")
    
    # Current Model Status
    with st.expander("🤖 Current Model Status"):
        if st.session_state.get('current_model'):
            model_name = st.session_state.current_model
            
            # Show active status during processing
            if (st.session_state.get('model_start_time') and 
                not st.session_state.get('model_end_time') or
                (st.session_state.get('model_end_time') and 
                 time.time() - st.session_state.model_end_time < 3)):
                
                st.success(f"🟢 Active: {model_name}")
                
                if st.session_state.get('model_start_time'):
                    elapsed = time.time() - st.session_state.model_start_time
                    st.write(f"⏱️ Processing: {elapsed:.1f}s")
            else:
                st.info(f"💤 Last used: {model_name}")
        else:
            st.info("No model activity yet")
    
    # Tool Registry Information
    with st.expander("🛠️ Discovered Tools"):
        if st.session_state.get('discovered_tools'):
            st.write(f"**Total Tools:** {len(st.session_state.discovered_tools)}")
            
            for tool in st.session_state.discovered_tools:
                tool_info = st.session_state.tool_registry.extract_tool_info(tool)
                tool_name = tool_info['name']
                tool_desc = tool_info['description'][:50] + "..." if len(tool_info['description']) > 50 else tool_info['description']
                st.write(f"• **{tool_name}**: {tool_desc}")
        else:
            st.info("No tools discovered yet")
    
    # Reset Options
    with st.expander("🔄 Reset Options"):
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.start_index = 0
            st.rerun()
        
        if st.button("Reset All Settings"):
            # Keep only essential keys
            keys_to_keep = ['messages', 'agent', 'tool_registry', 'discovered_tools']
            keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
            for key in keys_to_remove:
                del st.session_state[key]
            st.session_state.show_reset_message = True
            st.rerun()
        
        if st.session_state.get('show_reset_message'):
            st.success("Settings reset successfully!")
            if st.button("Dismiss"):
                st.session_state.show_reset_message = False
                st.rerun()
