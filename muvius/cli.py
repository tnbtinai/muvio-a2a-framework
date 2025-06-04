import os
import click
from pathlib import Path
import shutil
import subprocess
import sys
import json
import time
from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from rich.text import Text
from rich.table import Table
from rich.prompt import Confirm

# Initialize Rich console
console = Console()

# ASCII Art for Muvius
MUVIUS_BANNER = r"""
███╗   ███╗██╗   ██╗██╗   ██╗██╗██╗   ██╗███████╗
████╗ ████║██║   ██║██║   ██║██║██║   ██║██╔════╝
██╔████╔██║██║   ██║██║   ██║██║██║   ██║███████╗
██║╚██╔╝██║██║   ██║██║   ██║██║██║   ██║╚════██║
██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║╚██████╔╝███████║
╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝ ╚═════╝ ╚══════╝

A G E N T - T O - A G E N T   I N T E L L I G E N C E
                by MUVIO AI
"""

# Framework-level templates (created during muvius init)
FRAMEWORK_TEMPLATES = {
    "orchestrator/__init__.py": "",
    
    "orchestrator/main.py": """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from .orchestrator import Orchestrator
from .agent_protocol import AgentMessage, AgentResponse

app = FastAPI(title="Muvius Orchestrator")
orchestrator = Orchestrator()

class UserInput(BaseModel):
    message: str
    user_id: str

@app.post("/process-message")
async def process_message(input_data: UserInput):
    \"\"\"Process incoming messages and route to appropriate agent.\"\"\"
    try:
        # Create agent message
        message = AgentMessage(
            sender=input_data.user_id,
            content=input_data.message
        )
        
        # Process through orchestrator
        response = await orchestrator.process_message(message)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.error)
            
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",

    "orchestrator/orchestrator.py": """from typing import Dict, Optional
from .dispatcher import MessageDispatcher
from .intent_classifier import IntentClassifier
from .memory_manager import MemoryManager
from .prompt_builder import PromptBuilder
from .agent_protocol import AgentMessage, AgentResponse

class Orchestrator:
    def __init__(self):
        self.dispatcher = MessageDispatcher()
        self.intent_classifier = IntentClassifier()
        self.memory_manager = MemoryManager()
        self.prompt_builder = PromptBuilder()
        self.agent_types = []  # Will be populated dynamically
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        \"\"\"Process a message through the orchestrator pipeline.\"\"\"
        try:
            # Get conversation history
            history = self.memory_manager.get_conversation_history(message.sender)
            
            # Classify intent
            intent = await self.intent_classifier.classify(message.content, history)
            
            # Build prompt with context
            prompt = self.prompt_builder.build_prompt(
                message.content,
                intent,
                history
            )
            
            # Determine target agent
            target_agent = self.dispatcher.get_target_agent(intent)
            if not target_agent:
                return AgentResponse(
                    success=False,
                    error=f"No suitable agent found for intent: {intent}"
                )
            
            # Update message with prompt and target
            message.recipient = target_agent
            message.content = prompt
            
            # Dispatch to agent
            response = await self.dispatcher.dispatch_message(message)
            
            # Store in memory
            self.memory_manager.store_interaction(
                message.sender,
                message.content,
                response.data
            )
            
            return response
            
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Orchestration error: {str(e)}"
            )
    
    def register_agent(self, agent_name: str, port: int):
        \"\"\"Register a new agent with the orchestrator.\"\"\"
        self.agent_types.append(agent_name)
        self.dispatcher.register_agent(agent_name, port)
        self.intent_classifier.update_agent_types(self.agent_types)
""",

    "orchestrator/dispatcher.py": """from typing import Dict, Optional
import aiohttp
from .agent_protocol import AgentMessage, AgentResponse

class MessageDispatcher:
    def __init__(self):
        self.agent_urls: Dict[str, str] = {}
        self.agent_capabilities: Dict[str, list] = {}
    
    def register_agent(self, agent_name: str, port: int, capabilities: list = None):
        \"\"\"Register a new agent with its URL and capabilities.\"\"\"
        self.agent_urls[agent_name] = f"http://localhost:{port}/interact"
        self.agent_capabilities[agent_name] = capabilities or []
    
    def get_target_agent(self, intent: str) -> Optional[str]:
        \"\"\"Get the most suitable agent for a given intent.\"\"\"
        # Simple implementation - can be enhanced with ML-based routing
        for agent, capabilities in self.agent_capabilities.items():
            if intent in capabilities:
                return agent
        return None
    
    async def dispatch_message(self, message: AgentMessage) -> AgentResponse:
        \"\"\"Dispatch a message to the appropriate agent.\"\"\"
        target_url = self.agent_urls.get(message.recipient)
        if not target_url:
            return AgentResponse(
                success=False,
                error=f"Unknown agent: {message.recipient}"
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(target_url, json=message.dict()) as response:
                    if response.status == 200:
                        data = await response.json()
                        return AgentResponse(success=True, data=data)
                    else:
                        return AgentResponse(
                            success=False,
                            error=f"Agent returned status {response.status}"
                        )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Failed to dispatch message: {str(e)}"
            )
""",

    "orchestrator/intent_classifier.py": """from typing import List, Dict
from shared.llm import call_llm

class IntentClassifier:
    def __init__(self):
        self.agent_types = []
        self.intent_patterns = {
            "greeting": ["hello", "hi", "hey", "greetings"],
            "farewell": ["bye", "goodbye", "see you", "farewell"],
            "question": ["what", "how", "why", "when", "where", "who"],
            "request": ["can you", "could you", "please", "help me"],
            "confirmation": ["yes", "no", "sure", "okay", "fine"],
            "clarification": ["what do you mean", "can you explain", "i don't understand"]
        }
    
    def update_agent_types(self, agent_types: List[str]):
        \"\"\"Update the list of available agent types.\"\"\"
        self.agent_types = agent_types
    
    async def classify(self, message: str, context: List[Dict] = None) -> str:
        \"\"\"Classify the intent of a message.\"\"\"
        # First try pattern matching
        message_lower = message.lower()
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                return intent
        
        # If no pattern match, use LLM
        prompt = f\"\"\"Classify the intent of this message:
Message: {message}

Context: {context if context else 'No context available'}

Available intents: {', '.join(self.intent_patterns.keys())}

Respond with only the intent name from the available intents list.\"\"\"

        response = await call_llm(
            prompt=prompt,
            system_prompt="You are an intent classifier. Respond with only the intent name from the available intents list."
        )
        return response.strip().lower()
""",

    "orchestrator/memory_manager.py": """from typing import List, Dict, Any
from datetime import datetime
import json
from pathlib import Path

class MemoryManager:
    def __init__(self):
        self.memory_dir = Path("memory")
        self.memory_dir.mkdir(exist_ok=True)
        self.conversations_file = self.memory_dir / "conversations.json"
        self._init_storage()
    
    def _init_storage(self):
        \"\"\"Initialize the storage for conversations.\"\"\"
        if not self.conversations_file.exists():
            with open(self.conversations_file, "w") as f:
                json.dump({}, f)
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        \"\"\"Get the conversation history for a user.\"\"\"
        with open(self.conversations_file, "r") as f:
            conversations = json.load(f)
        
        user_history = conversations.get(user_id, [])
        return user_history[-limit:] if limit else user_history
    
    def store_interaction(self, user_id: str, message: str, response: Dict[str, Any]):
        \"\"\"Store an interaction in the conversation history.\"\"\"
        with open(self.conversations_file, "r") as f:
            conversations = json.load(f)
        
        if user_id not in conversations:
            conversations[user_id] = []
        
        conversations[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response
        })
        
        with open(self.conversations_file, "w") as f:
            json.dump(conversations, f)
    
    def clear_history(self, user_id: str):
        \"\"\"Clear the conversation history for a user.\"\"\"
        with open(self.conversations_file, "r") as f:
            conversations = json.load(f)
        
        if user_id in conversations:
            del conversations[user_id]
        
        with open(self.conversations_file, "w") as f:
            json.dump(conversations, f)
""",

    "orchestrator/prompt_builder.py": """from typing import List, Dict, Any
from shared.llm import call_llm

class PromptBuilder:
    def __init__(self):
        self.prompt_templates = {
            "greeting": "You are a friendly assistant. Respond to: {message}",
            "farewell": "You are a polite assistant. Respond to: {message}",
            "question": "You are a helpful assistant. Answer this question: {message}",
            "request": "You are a capable assistant. Handle this request: {message}",
            "confirmation": "You are a clear assistant. Respond to: {message}",
            "clarification": "You are a patient assistant. Clarify: {message}"
        }
    
    def build_prompt(self, message: str, intent: str, context: List[Dict[str, Any]] = None) -> str:
        \"\"\"Build a prompt with context for the LLM.\"\"\"
        # Get base template
        template = self.prompt_templates.get(intent, self.prompt_templates["question"])
        
        # Add context if available
        if context:
            context_str = "\\n".join([
                f"Previous: {item['message']}\\nResponse: {item['response']}"
                for item in context[-3:]  # Last 3 interactions
            ])
            template = f\"\"\"Context from previous interactions:
{context_str}

{template}\"\"\"
        
        return template.format(message=message)
    
    async def enhance_prompt(self, prompt: str) -> str:
        \"\"\"Enhance the prompt using the LLM.\"\"\"
        enhancement_prompt = f\"\"\"Improve this prompt to be more effective:
{prompt}

Respond with only the improved prompt.\"\"\"
        
        return await call_llm(
            prompt=enhancement_prompt,
            system_prompt="You are a prompt engineering expert. Improve the given prompt to be more effective."
        )
""",

    "orchestrator/agent_protocol.py": """from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class AgentMessage(BaseModel):
    sender: str = Field(..., description="ID of the message sender")
    recipient: Optional[str] = Field(None, description="ID of the target agent")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if any")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentCapability(BaseModel):
    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of the capability")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required: bool = Field(default=True)

class AgentInfo(BaseModel):
    name: str = Field(..., description="Name of the agent")
    version: str = Field(..., description="Version of the agent")
    capabilities: List[AgentCapability] = Field(default_factory=list)
    description: Optional[str] = Field(None, description="Description of the agent")
    metadata: Dict[str, Any] = Field(default_factory=dict)
""",

    "shared/__init__.py": "",
    
    "shared/llm.py": """from typing import Optional
import os
import requests

def call_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    \"\"\"Call the language model with the given prompt.\"\"\"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]
""",

    "agents/__init__.py": "",
    
    "agents/README.md": """# Agents Directory

This directory contains all the AI agents created using the Muvius framework.

## Available Agents

{agent_list}

## Agent Structure

Each agent has the following structure:
- `main.py` - Main FastAPI application
- `routes.py` - API route definitions
- `memory/` - Memory systems (episodic, procedural, vector store)
- `README.md` - Agent-specific documentation

## Running Agents

To run an agent:
```bash
python -m agents.<agent-name>-agent.main
```

Example:
```bash
python -m agents.user-agent.main
```
""",

    "requirements.txt": """# Core dependencies
click>=8.1.7
fastapi>=0.104.1
uvicorn>=0.24.0
requests>=2.31.0
pydantic>=2.5.2
python-dotenv>=1.0.0
pyyaml>=6.0.1
rich>=13.7.0
typer>=0.9.0
openai>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
python-multipart>=0.0.6
aiohttp>=3.9.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.25.0
""",

    ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Environment variables
.env

# Memory files
memory/
*.db
*.json

# Logs
*.log
""",

    ".env": """# OpenAI API Key
OPENAI_API_KEY=your_api_key_here

# Orchestrator
ORCHESTRATOR_URL=http://localhost:8000

# Database
DATABASE_URL=sqlite:///memory/episodic.db
""",

    "README.md": """# Muvius Framework

A framework for building and managing AI agents with memory systems and API endpoints.

## Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\\venv\\Scripts\\activate
```

2. Install Muvius CLI:
```bash
# Install from source
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/tnbtinai/muvius-a2a-framework.git
```

## CLI Commands

### Initialize Framework
Initialize the Muvius framework with basic structure:
```bash
muvius init
```

Options:
- `--force`: Force overwrite existing files
- `--skip-install`: Skip dependency installation

### Agent Management

#### Create Agent
Create a new agent with memory systems and API endpoints:
```bash
muvius create-agent <agent-name>
```

Example:
```bash
# Create a new agent
muvius create-agent my-agent
```

#### Delete Agent
Delete an existing agent:
```bash
muvius delete-agent <agent-name>
```

Example:
```bash
# Delete an agent
muvius delete-agent my-agent
```

## Project Structure

```
.
├── agents/           # Agent modules
│   └── README.md    # Agent documentation
├── orchestrator/    # Message routing and coordination
└── shared/         # Shared utilities and services
```

## Development

### Prerequisites
- Python 3.8+
- pip
- virtualenv

### Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_api_key_here
```

### Running Agents
1. Start the orchestrator:
```bash
python -m orchestrator.main
```

2. Start individual agents:
```bash
python -m agents.<agent-name>-agent.main
```

### API Endpoints
- Orchestrator: http://localhost:8000
- Agent ports are assigned automatically starting from 8001 (first agent gets 8001, second gets 8002, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
"""
}

# Agent-level templates (created during muvius create-agent)
AGENT_TEMPLATES = {
    "__init__.py": "",
    
    "main.py": """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from .memory.episodic import EpisodicMemory
from .memory.procedural import ProceduralMemory
from .memory.vector_store import VectorStore
from .routes import router

app = FastAPI(title="{agent_name} Agent")
app.include_router(router)

# Initialize memory systems
episodic_memory = EpisodicMemory("{agent_role}")
procedural_memory = ProceduralMemory("{agent_role}")
vector_store = VectorStore("{agent_role}")

class UserInput(BaseModel):
    message: str
    user_id: str

@app.post("/interact")
async def interact(input_data: UserInput):
    \"\"\"Handle user interactions.\"\"\"
    try:
        # Store in episodic memory
        episodic_memory.log_episode(
            user_id=input_data.user_id,
            role="user",
            message=input_data.message
        )
        
        # Get conversation history
        history = episodic_memory.get_conversation_history(input_data.user_id)
        
        # Get relevant procedures
        procedures = procedural_memory.load_data()
        
        # Process the message (implement your agent's logic here)
        response = {
            "message": f"Hello! I am the {agent_name} agent. I received your message: {input_data.message}",
            "history": history,
            "procedures": procedures
        }
        
        # Store response in episodic memory
        episodic_memory.log_episode(
            user_id=input_data.user_id,
            role="agent",
            message=response["message"]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
    \"\"\"Health check endpoint.\"\"\"
    return {"status": "healthy", "agent": "{agent_name}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={port})
""",

    "routes.py": """from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from .memory.episodic import EpisodicMemory
from .memory.procedural import ProceduralMemory
from .memory.vector_store import VectorStore

router = APIRouter()

# Initialize memory systems
episodic_memory = EpisodicMemory("{agent_role}")
procedural_memory = ProceduralMemory("{agent_role}")
vector_store = VectorStore("{agent_role}")

class ProcedureInput(BaseModel):
    name: str
    steps: List[Dict[str, str]]

@router.post("/procedures")
async def add_procedure(input_data: ProcedureInput):
    \"\"\"Add a new procedure.\"\"\"
    try:
        procedural_memory.add_procedure(input_data.name, input_data.steps)
        return {"message": f"Procedure '{input_data.name}' added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/procedures")
async def get_procedures():
    \"\"\"Get all procedures.\"\"\"
    try:
        return procedural_memory.load_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 10):
    \"\"\"Get conversation history for a user.\"\"\"
    try:
        return episodic_memory.get_conversation_history(user_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/{user_id}")
async def clear_history(user_id: str):
    \"\"\"Clear conversation history for a user.\"\"\"
    try:
        episodic_memory.clear_user_history(user_id)
        return {"message": f"History cleared for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
""",

    "memory/__init__.py": "",
    
    "memory/episodic.py": """from typing import List, Dict, Any
import sqlite3
from datetime import datetime
import uuid
from pathlib import Path

class EpisodicMemory:
    def __init__(self, agent_role: str):
        self.agent_role = agent_role
        # Use the current directory as base for memory files
        self.memory_dir = Path("memory")
        self.memory_dir.mkdir(exist_ok=True)
        self.db_path = self.memory_dir / f"{agent_role}_episodic.db"
        self._init_db()
    
    def _init_db(self):
        \"\"\"Initialize the SQLite database for episodic memory.\"\"\"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    user_id TEXT,
                    role TEXT,
                    message TEXT,
                    agent TEXT
                )
            ''')
    
    def log_episode(self, user_id: str, role: str, message: str):
        \"\"\"Log an episode to the memory.\"\"\"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO episodes (id, timestamp, user_id, role, message, agent) VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), datetime.now().isoformat(), user_id, role, message, self.agent_role)
            )
    
    def get_last_episodes(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        \"\"\"Get the last N episodes for a user.\"\"\"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM episodes WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        \"\"\"Get the complete conversation history for a user.\"\"\"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM episodes WHERE user_id = ? ORDER BY timestamp ASC",
                (user_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def clear_user_history(self, user_id: str):
        \"\"\"Clear all episodes for a user.\"\"\"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM episodes WHERE user_id = ?", (user_id,))
""",

    "memory/procedural.py": """from typing import Dict, Any
import yaml
from pathlib import Path

class ProceduralMemory:
    def __init__(self, agent_role: str):
        self.agent_role = agent_role
        # Use the current directory as base for memory files
        self.memory_dir = Path("memory")
        self.memory_dir.mkdir(exist_ok=True)
        self.yaml_path = self.memory_dir / f"{agent_role}_procedural.yaml"
        self._init_yaml()
    
    def _init_yaml(self):
        \"\"\"Initialize the YAML file for procedural memory.\"\"\"
        if not self.yaml_path.exists():
            default_data = {
                "procedures": {},
                "rules": {},
                "preferences": {}
            }
            self.save_data(default_data)
    
    def save_data(self, data: Dict[str, Any]):
        \"\"\"Save data to the YAML file.\"\"\"
        with open(self.yaml_path, 'w') as f:
            yaml.dump(data, f)
    
    def load_data(self) -> Dict[str, Any]:
        \"\"\"Load data from the YAML file.\"\"\"
        with open(self.yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    def add_procedure(self, name: str, steps: list):
        \"\"\"Add a new procedure.\"\"\"
        data = self.load_data()
        data["procedures"][name] = steps
        self.save_data(data)
    
    def get_procedure(self, name: str) -> list:
        \"\"\"Get a procedure by name.\"\"\"
        data = self.load_data()
        return data["procedures"].get(name, [])
    
    def add_rule(self, name: str, rule: str):
        \"\"\"Add a new rule.\"\"\"
        data = self.load_data()
        data["rules"][name] = rule
        self.save_data(data)
    
    def get_rule(self, name: str) -> str:
        \"\"\"Get a rule by name.\"\"\"
        data = self.load_data()
        return data["rules"].get(name, "")
""",

    "memory/vector_store.py": """from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import json

class VectorStore:
    def __init__(self, agent_role: str):
        self.agent_role = agent_role
        # Use the current directory as base for memory files
        self.memory_dir = Path("memory")
        self.memory_dir.mkdir(exist_ok=True)
        self.store_path = self.memory_dir / f"{agent_role}_vectors.json"
        self._init_store()
    
    def _init_store(self):
        \"\"\"Initialize the vector store.\"\"\"
        if not self.store_path.exists():
            self.save_vectors({})
    
    def save_vectors(self, vectors: Dict[str, List[float]]):
        \"\"\"Save vectors to the store.\"\"\"
        with open(self.store_path, 'w') as f:
            json.dump(vectors, f)
    
    def load_vectors(self) -> Dict[str, List[float]]:
        \"\"\"Load vectors from the store.\"\"\"
        with open(self.store_path, 'r') as f:
            return json.load(f)
    
    def add_vector(self, key: str, vector: List[float]):
        \"\"\"Add a vector to the store.\"\"\"
        vectors = self.load_vectors()
        vectors[key] = vector
        self.save_vectors(vectors)
    
    def get_vector(self, key: str) -> List[float]:
        \"\"\"Get a vector by key.\"\"\"
        vectors = self.load_vectors()
        return vectors.get(key, [])
    
    def find_similar(self, query_vector: List[float], top_k: int = 5) -> List[str]:
        \"\"\"Find similar vectors using cosine similarity.\"\"\"
        vectors = self.load_vectors()
        if not vectors:
            return []
        
        similarities = {}
        query_norm = np.linalg.norm(query_vector)
        
        for key, vector in vectors.items():
            vector_norm = np.linalg.norm(vector)
            if vector_norm == 0:
                continue
            similarity = np.dot(query_vector, vector) / (query_norm * vector_norm)
            similarities[key] = similarity
        
        return sorted(similarities.keys(), key=lambda k: similarities[k], reverse=True)[:top_k]
""",

    "README.md": """# {agent_name} Agent

This is a Muvius agent that handles {agent_role} interactions.

## Features

- Episodic memory for conversation history
- Procedural memory for rules and procedures
- Vector store for semantic search
- FastAPI endpoints for interaction

## API Endpoints

- `POST /interact`: Handle user interactions
- `GET /ping`: Health check endpoint

## Memory Systems

### Episodic Memory
Stores conversation history and user interactions.

### Procedural Memory
Stores rules, procedures, and preferences in YAML format.

### Vector Store
Provides semantic search capabilities for similar content.

## Development

To run this agent:
```bash
python -m agents.{agent_role}-agent.main
```

The agent will be available at http://localhost:{port}
"""
}

def install_dependencies():
    """Install required dependencies."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Installing dependencies...", total=None)
        
        try:
            # First create requirements.txt
            requirements_content = """# Core dependencies
click>=8.1.7
fastapi>=0.104.1
uvicorn>=0.24.0
requests>=2.31.0
pydantic>=2.5.2
python-dotenv>=1.0.0
pyyaml>=6.0.1
rich>=13.7.0
typer>=0.9.0
openai>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
python-multipart>=0.0.6
aiohttp>=3.9.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.25.0"""
            
            with open("requirements.txt", "w") as f:
                f.write(requirements_content)
            
            # Then install dependencies
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            progress.update(task, description="[green]Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            progress.update(task, description="[red]Failed to install dependencies!")
            console.print(f"[red]Error: {str(e)}")
            sys.exit(1)
        except Exception as e:
            progress.update(task, description="[red]Failed to create requirements.txt!")
            console.print(f"[red]Error: {str(e)}")
            sys.exit(1)

def create_directory_structure():
    """Create the basic directory structure for the Muvius framework."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Creating directory structure...", total=None)
        
        directories = [
            "agents",
            "orchestrator",
            "shared",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        progress.update(task, description="[green]Directory structure created!")

def create_template_files():
    """Create template files for the Muvius framework."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Creating template files...", total=None)
        
        for file_path, content in FRAMEWORK_TEMPLATES.items():
            full_path = Path(file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
        
        progress.update(task, description="[green]Template files created!")

def display_success_message():
    """Display success message with next steps."""
    console.print("\n")
    console.print(Panel.fit(
        Text("Muvius Framework Setup Complete!", style="bold green"),
        title="Success",
        border_style="green"
    ))
    
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. Configure your environment variables in [yellow].env[/yellow]")
    console.print("2. Create your agents using [yellow]muvius create-agent <agent-name>[/yellow]")
    console.print("3. Start the orchestrator using [yellow]python -m orchestrator.main[/yellow]")
    
    console.print("\n[bold]Available Services:[/bold]")
    console.print("• Orchestrator: [yellow]http://localhost:8001[/yellow]")
    console.print("• Agent ports will be assigned automatically starting from 8001")

def update_agents_readme():
    """Update the agents README with the current list of agents."""
    agents_dir = Path("agents")
    agent_list = []
    
    # Get list of existing agents
    for item in agents_dir.iterdir():
        if item.is_dir() and item.name.endswith("-agent"):
            agent_name = item.name.replace("-agent", "")
            agent_list.append(f"- `{agent_name}`")
    
    # Update README with agent list
    readme_path = agents_dir / "README.md"
    if readme_path.exists():
        with open(readme_path, "r") as f:
            content = f.read()
        
        # Replace the agent list section
        if "{agent_list}" in content:
            agent_list_text = "\n".join(agent_list) if agent_list else "No agents created yet."
            content = content.replace("{agent_list}", agent_list_text)
            
            with open(readme_path, "w") as f:
                f.write(content)

def safe_format_template(content: str, replacements: dict) -> str:
    """Safely replace template placeholders without using .format()"""
    result = content
    for key, value in replacements.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result

def create_agent(agent_name: str):
    """Create a new agent with all necessary files and structure."""
    agent_role = agent_name.lower()
    agent_dir = Path(f"agents/{agent_role}-agent")
    
    # Assign port dynamically based on existing agents, starting from 8001
    base_port = 8001
    existing_agents = [d for d in Path("agents").iterdir() if d.is_dir() and d.name.endswith("-agent")]
    port = base_port + len(existing_agents)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Create agent directory
        task = progress.add_task(f"Creating {agent_name} agent...", total=None)
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Create memory directory inside agent directory
        memory_dir = agent_dir / "memory"
        memory_dir.mkdir(exist_ok=True)
        
        # Create all template files
        for template_path, content in AGENT_TEMPLATES.items():
            full_path = agent_dir / template_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Format template content with agent-specific values
            if template_path == "agents/README.md":
                # Skip formatting the agents README as it's handled separately
                formatted_content = content
            else:
                formatted_content = safe_format_template(content, {
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                    "port": port
                })
            
            with open(full_path, "w") as f:
                f.write(formatted_content)
        
        # Update agents README
        update_agents_readme()
        
        progress.update(task, description=f"[green]{agent_name} agent created successfully!")

def delete_agent(agent_name: str):
    """Delete an existing agent."""
    agent_role = agent_name.lower()
    agent_dir = Path(f"agents/{agent_role}-agent")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Deleting {agent_name} agent...", total=None)
        
        if not agent_dir.exists():
            progress.update(task, description=f"[red]Error: {agent_name} agent does not exist!")
            return
        
        try:
            shutil.rmtree(agent_dir)
            # Update agents README
            update_agents_readme()
            progress.update(task, description=f"[green]{agent_name} agent deleted successfully!")
        except Exception as e:
            progress.update(task, description=f"[red]Error deleting {agent_name} agent: {str(e)}")

@click.group()
def cli():
    """Muvius Framework CLI"""
    pass

@cli.command()
@click.option('--force', is_flag=True, help='Force overwrite existing files')
@click.option('--skip-install', is_flag=True, help='Skip dependency installation')
def init(force, skip_install):
    """Initialize the Muvius framework with basic structure and templates."""
    # Display banner
    console.print(Panel.fit(
        Text(MUVIUS_BANNER, style="bold blue"),
        title="Muvius Framework",
        subtitle="v0.1.0",
        border_style="blue"
    ))
    
    if not force and Path("agents").exists():
        console.print("[red]Error: Framework already exists. Use --force to overwrite.")
        return

    try:
        # Install dependencies if not skipped
        if not skip_install:
            install_dependencies()
        
        # Create structure
        create_directory_structure()
        create_template_files()
        
        # Display success message
        display_success_message()
        
    except Exception as e:
        console.print(f"[red]Error during initialization: {str(e)}")
        sys.exit(1)

@cli.command(name='create-agent')
@click.argument('agent_name')
def create_agent_command(agent_name):
    """Create a new agent with all necessary files and structure."""
    create_agent(agent_name)

@cli.command(name='delete-agent')
@click.argument('agent_name')
def delete_agent_command(agent_name):
    """Delete an existing agent."""
    delete_agent(agent_name)

if __name__ == '__main__':
    cli() 