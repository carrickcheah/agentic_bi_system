# Agentic BI Chat Interface

## Overview

The `chat.py` provides an interactive command-line interface for the Agentic BI system using the new `AgenticBiFlow` from `main.py`.

## Key Features

- **Simple Import**: Uses `from main import AgenticBiFlow`
- **Clean Interface**: Just 3 lines to initialize and use
- **Streaming Progress**: Shows real-time investigation phases
- **Cache Statistics**: Track performance with `stats` command
- **Help System**: Built-in examples with `help` command

## Usage

### Running the Chat Interface

```bash
cd /Users/carrickcheah/Project/agentic_sql/app
source .venv/bin/activate  # Activate virtual environment
python chat.py
```

### Available Commands

- **Business Questions**: Ask any business intelligence question
- **`help`**: Show example questions
- **`stats`**: Display cache statistics
- **`exit`/`quit`**: Exit the chat

### Example Session

```
Welcome to Agentic BI - Business Intelligence Assistant
============================================================

Initializing services...
✅ All services initialized successfully!

💡 Tips:
  - Ask business questions like 'What were last month's sales?'
  - Type 'help' for examples
  - Type 'exit' to quit
  - Type 'stats' to see cache statistics

============================================================

🤖 You: What were yesterday's sales?

🔍 Processing your query...

📍 Phase 1/5: intelligence_planning
   [ 25%] Analyzing business question and planning investigation strategy...

📍 Phase 2/5: service_orchestration
   [ 50%] MCP database services ready for investigation...

📍 Phase 3/5: investigation_execution
   [ 80%] Executing autonomous investigation with 7-step framework...

📍 Phase 4/5: insight_synthesis
   [ 90%] Synthesizing strategic insights and generating recommendations...

✅ Investigation completed in 12.3s

📋 Executive Summary:
Yesterday's total sales were $47,832, representing a 12% increase...

💡 Strategic Insights (3):
  1. Strong Performance in Electronics Category
     Electronics sales exceeded targets by 18%, driven by new product launches...

🎯 Key Recommendations (2):
  1. Increase Electronics Inventory
     Priority: High
  2. Optimize Marketing for Top Performers
     Priority: Medium
```

## Architecture

The chat interface demonstrates the hybrid approach benefits:

1. **Simple Import**: Just imports `AgenticBiFlow` from main.py
2. **Clean Flow**: All complex logic hidden in business_analyst.py
3. **Easy to Use**: Initialize and call methods
4. **Progress Tracking**: Real-time updates during investigation

## Code Structure

```python
# Import the high-level interface
from main import AgenticBiFlow

# Initialize
flow = AgenticBiFlow()
await flow.initialize()

# Use for queries
async for update in flow.investigate_query(question, user_context, org_context):
    # Handle progress updates and results
```

This is exactly what the hybrid approach achieves - simple interface with sophisticated implementation!