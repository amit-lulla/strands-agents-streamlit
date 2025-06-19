# strands-agents-streamlit

A comprehensive demo of [Strands Agents](https://strandsagents.com/) in [Streamlit](https://streamlit.io/) with advanced features including semantic tool lookup, circuit breaker pattern, agent evaluation, and multi-model support.

## Prerequisites

* python >= 3.10
* `anthropic.claude-3-7-sonnet` model enabled in Amazon Bedrock in your AWS account
* configure valid AWS credentials in your execution environment
* (Optional) Ollama installed locally for fallback support
* (Optional) Ollama models: `gemma3:1b`, `deepseek-r1:latest`, `llama3`

## Applications

This repository contains two applications:

### Basic Demo (`app.py`)
Simple Strands Agents demonstration with basic appointment management.

### Enhanced Demo (`app_enhanced.py`)
Advanced demo with multiple sophisticated features:

## Enhanced Demo Features

### 🤖 Multi-Model Support with Circuit Breaker
- **Primary**: Amazon Bedrock Claude 3.7 Sonnet
- **Fallback**: Ollama models (configurable)
- Circuit breaker pattern with configurable timeout and failure thresholds
- Automatic failover when Bedrock is unavailable or slow

### 🔍 Semantic Tool Lookup (RAG)
- Automatic tool discovery from multiple sources
- Semantic search using txtai embeddings with sentence-transformers
- BM25 scoring for improved tool matching
- Zero-LLM-call execution for simple queries
- Dynamic parameter extraction using Ollama

### 📊 Agent Evaluation System
- Comprehensive test case framework
- LLM-based evaluation with scoring
- Multi-dimensional assessment (accuracy, relevance, completeness)
- Performance metrics and detailed reporting
- CSV export functionality

### 💬 Enhanced Chat Interface
- Beautiful message formatting with thinking process visualization
- Real-time streaming responses
- Token usage and latency metrics
- Interactive tool lookup results
- Visual indicators for model status

### 🛠️ Advanced Tool Management
- Automatic tool discovery from `./tools` directory
- Tool registry with semantic indexing
- Direct tool execution capability
- Parameter extraction and validation
- Tool usage analytics

## Run

### Basic Demo
```bash
streamlit run app.py --server.port 8080
```

### Enhanced Demo
```bash
streamlit run app_enhanced.py --server.port 8080
```

## Installation

1. Clone this repo:
```bash
git clone https://github.com/amit-lulla/strands-agents-streamlit.git
cd strands-agents-streamlit
```

2. Create a new virtual env and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. (Optional) Install and run Ollama for fallback support:
```bash
# Install Ollama - see https://ollama.ai
ollama serve &
ollama pull gemma3:1b
ollama pull deepseek-r1:latest
```

## Usage Examples

### Basic Appointment Management
```bash
Create an appointment for tomorrow 2+2 hours later with Amit at AWS office in London.
```

### Advanced Queries (Enhanced Demo)
```bash
# Semantic tool lookup will automatically find the right tool
What time is it?
Calculate 15 * 23 + 45
Show me all my appointments
```

## Architecture

### Enhanced Demo Architecture
- **Tool Discovery**: Automatic scanning of tools directory and imports
- **Semantic Search**: txtai embeddings with sentence-transformers/all-MiniLM-L6-v2
- **Circuit Breaker**: pybreaker with configurable thresholds
- **Parameter Extraction**: Ollama-based with fallback to regex parsing
- **Evaluation**: LLM judging with multi-dimensional scoring
- **Streaming**: Real-time response display with thinking visualization

### Tool Registry
The enhanced demo automatically discovers tools from:
- `./tools/` directory (decorated with `@tool`)
- `strands_tools` library imports
- Functions with `TOOL_SPEC` attributes

## Configuration

### Circuit Breaker Settings
- **Timeout**: 1-10 seconds (default: 2s)
- **Max Failures**: 1-10 (default: 3)
- **Reset Timeout**: 60 seconds

### Semantic Search Settings
- **Threshold**: 0.1-1.0 (default: 0.8)
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Scoring**: Hybrid semantic + BM25

## Evaluation

The enhanced demo includes a comprehensive evaluation system:

### Test Categories
- **Time Queries**: Current time, date calculations
- **Math Operations**: Calculator usage, complex expressions
- **Appointment Management**: CRUD operations on appointments
- **Tool Selection**: Semantic matching accuracy

### Metrics
- **Accuracy Score**: 0-1.0 based on expected vs actual
- **LLM Judge Scores**: Multi-dimensional evaluation (1-5 scale)
- **Performance**: Token usage, response time, tool efficiency
- **Success Rate**: Percentage of successful executions

#### Input and execution metrics would be similar to this:
![Input & Token Metrics](assets/1.png "Input & Token Metrics")

#### Output will be similar to this:
![Output](assets/2.png "Output of Agent")

## Troubleshooting

### Common Issues
1. **Bedrock Access**: Ensure Claude 3.7 Sonnet is enabled in your AWS region
2. **Ollama Connection**: Check `ollama serve` is running on localhost:11434
3. **Model Loading**: Verify Ollama models are pulled: `ollama list`
4. **Dependencies**: Install missing packages: `pip install -r requirements.txt`

### Debug Features
- Tool lookup results display
- Raw Ollama response inspection
- Circuit breaker status monitoring
- Execution trace visualization

5. Deactivate the virtual env:
```bash
deactivate
```

## License

This application is licensed under Apache License 2.0 - See the [LICENSE](LICENSE) file for details.

## Author
[Amit Lulla](https://github.com/amit-lulla)
