# Atlas-PyTorch AI Agent Instructions

## Architecture Overview

This is a **hybrid AI system** combining:
1. **Custom PyTorch chatbot** (`src/model.py`, `src/chatbot.py`) - distilGPT2-based model trained on TIAA financial services data
2. **Smart agent system** (`src/atlas_agent.py`) - Multi-level agent with MySQL integration, conversation context, and OpenAI fallback
3. **Flask web interface** (`src/app.py`) - Customer service UI with database operations

**Key Integration Point**: `src/app.py` orchestrates the entire system - it initializes the `AtlasAgent` with database connections and serves the web interface.

## Critical Development Patterns

### Agent Architecture (src/atlas_agent.py)
- **Conversation Context**: Uses `@dataclass ConversationContext` to maintain user state, history, and pending actions
- **Tool Registry**: `AgentTools` class provides database operations (`get_customers`, `add_customer`, etc.)
- **Dual Processing**: Pattern matching for known queries + OpenAI integration for complex requests
- **Error Recovery**: Multi-level fallbacks from structured responses to LLM generation

### Database Integration
- **Connection Pattern**: MySQL connection established in `app.py`, passed to `AtlasAgent`
- **Customer Operations**: Direct SQL queries in `AgentTools` methods - no ORM used
- **Configuration**: Database settings in `.env` (Docker MySQL on port 3307)

### Model Training Workflow
```bash
# Standard development sequence
cd src
python train.py          # Train model with data/dataset.csv
python test_model.py     # Quick validation
python app.py           # Start Flask server with agent
```

### Data Format Conventions
- **Training Data**: CSV with `prompt,response,category` columns in `src/data/dataset.csv`
- **Response Categories**: Greeting, About TIAA, Retirement Products, Tools & Resources, Wealth Management
- **Name Extraction**: Complex regex patterns in `app.py` for contact information queries

## Development Environment

### Required Setup
1. **Python Environment**: Virtual environment in `venv/` (already configured)
2. **Dependencies**: `pip install -r requirements.txt` (includes PyTorch, transformers, Flask, MySQL connector)
3. **Environment Variables**: Copy `.env.example` to `.env` with OpenAI API key and database config
4. **Database**: Docker MySQL container expected on port 3307

### Key Commands
```bash
# Development server with live reload
python src/app.py

# Model training
python src/train.py

# Jupyter exploration
jupyter lab notebooks/exploration.ipynb
```

## Project-Specific Conventions

### File Organization
- **Core Logic**: Everything in `src/` directory
- **Static Assets**: `src/static/` and `src/templates/` for Flask
- **Data Pipeline**: `src/data/dataset.csv` → `train.py` → `model_weights.pth` → `chatbot.py`

### Code Patterns
- **Import Style**: Relative imports within `src/` (e.g., `from atlas_agent import AtlasAgent`)
- **Error Handling**: Agent methods return structured dictionaries with success/error states
- **Web Interface**: Bootstrap-styled chat interface with custom CSS for message bubbles
- **State Management**: Conversation context persisted in `AtlasAgent` instances

### Integration Points
- **Model Loading**: Both `chatbot.py` and `model.py` load distilGPT2 but with different interfaces
- **Agent Tools**: Database operations abstracted in `AgentTools` class for reusability
- **Environment Config**: Critical for OpenAI integration and database connectivity

## Testing & Debugging

- **Quick Model Test**: Use `src/test_model.py` for immediate model validation
- **Jupyter Exploration**: `notebooks/exploration.ipynb` for data analysis and model experimentation  
- **Flask Debug Mode**: Enabled by default in `.env.example` for hot reloading
- **Agent Testing**: Direct instantiation in `app.py` with success/error logging

When extending this system, maintain the dual-processing approach (pattern matching + LLM) and the conversation context threading throughout the agent system.