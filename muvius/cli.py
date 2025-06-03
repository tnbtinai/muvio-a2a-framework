import os
import click
from pathlib import Path
import shutil
import subprocess
import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from rich.text import Text
from rich.syntax import Syntax

# Initialize Rich console
console = Console()

# ASCII Art for Muvius
MUVIUS_BANNER = """
███╗   ███╗██╗   ██╗██╗   ██╗██╗██╗   ██╗███████╗
████╗ ████║██║   ██║██║   ██║██║██║   ██║██╔════╝
██╔████╔██║██║   ██║██║   ██║██║██║   ██║███████╗
██║╚██╔╝██║██║   ██║██║   ██║██║██║   ██║╚════██║
██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║╚██████╔╝███████║
╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝ ╚═════╝ ╚══════╝

A G E N T - T O - A G E N T   I N T E L L I G E N C E
                by MUVIO AI
"""

# Agent-specific templates
AGENT_TEMPLATES = {
    "memory/episodic.py": """from typing import List, Dict, Any
import sqlite3
from datetime import datetime
import uuid
from pathlib import Path

class EpisodicMemory:
    def __init__(self, agent_role: str):
        self.agent_role = agent_role
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
            cursor = conn.execute(
                "SELECT * FROM episodes WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        \"\"\"Get the complete conversation history for a user.\"\"\"
        with sqlite3.connect(self.db_path) as conn:
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

    "memory/procedural.yaml": """# Procedural memory for {agent_role}
procedures:
  # Add your procedures here
  example_procedure:
    - step1: "First step description"
    - step2: "Second step description"

rules:
  # Add your rules here
  example_rule: "Rule description"

preferences:
  # Add your preferences here
  example_preference: "Preference value"
""",

    "memory/vector_store.py": """from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import json

class VectorStore:
    def __init__(self, agent_role: str):
        self.agent_role = agent_role
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

    "main.py": """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from memory.episodic import EpisodicMemory
from memory.procedural import ProceduralMemory
from memory.vector_store import VectorStore
from routes import router

# Initialize FastAPI app
app = FastAPI(title="{agent_name} Agent")

# Initialize memory systems
episodic_memory = EpisodicMemory("{agent_role}")
procedural_memory = ProceduralMemory("{agent_role}")
vector_store = VectorStore("{agent_role}")

# Include routes
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
""",

    "routes.py": """from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from memory.episodic import EpisodicMemory
from memory.procedural import ProceduralMemory
from memory.vector_store import VectorStore

router = APIRouter()

class Message(BaseModel):
    user_id: str
    content: str

@router.post("/interact")
async def interact(message: Message) -> Dict[str, Any]:
    \"\"\"Handle interaction with the agent.\"\"\"
    try:
        # Log the user message
        episodic_memory.log_episode(message.user_id, "user", message.content)
        
        # Get conversation history
        history = episodic_memory.get_last_episodes(message.user_id)
        
        # Process the message (implement your agent's logic here)
        response = "I am the {agent_name} agent. I received your message: " + message.content
        
        # Log the agent's response
        episodic_memory.log_episode(message.user_id, "agent", response)
        
        return {{"response": response}}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ping")
async def ping() -> Dict[str, str]:
    \"\"\"Health check endpoint.\"\"\"
    return {{"status": "pong"}}
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
.\venv\Scripts\activate
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
muvius/
├── agents/           # Agent modules
│   └── README.md    # Agent documentation
├── orchestrator/    # Message routing and coordination
└── shared/         # Shared utilities and services
```

## Agent Structure

Each agent contains:
- `main.py`: Main agent logic and FastAPI server
- `routes.py`: API endpoints
- `memory/`: Memory systems
  - `episodic.py`: Conversation history
  - `procedural.py`: Rules and procedures
  - `procedural.yaml`: Configuration
  - `vector_store.py`: Semantic search

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
python -m muvius.orchestrator.main
```

2. Start individual agents:
```bash
python -m muvius.agents.<agent-name>-agent.main
```

### API Endpoints
- Orchestrator: http://localhost:8001
- Agent ports are assigned automatically starting from 8000

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
""",

    "agents/README.md": """# Muvius Agents

This directory contains all the agents in the Muvius framework. Each agent is a separate module with its own memory systems and API endpoints.

## Creating a New Agent

To create a new agent, use the CLI command:
```bash
muvius create-agent <agent-name>
```

This will create a new agent with:
- Memory systems (episodic, procedural, vector store)
- API endpoints
- Documentation
- Proper directory structure

Example:
```bash
muvius create-agent buyer
muvius create-agent seller
```

## Deleting an Agent

To delete an existing agent, use:
```bash
muvius delete-agent <agent-name>
```

Example:
```bash
muvius delete-agent buyer
```

## Agent Structure

Each agent directory contains:
- `main.py`: Main agent logic and FastAPI server
- `routes.py`: API endpoints
- `memory/`: Memory systems
  - `episodic.py`: Conversation history
  - `procedural.py`: Rules and procedures
  - `procedural.yaml`: Configuration
  - `vector_store.py`: Semantic search

## Available Agents

List of currently available agents:
{agent_list}
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

# Development dependencies
pytest>=7.4.3
black>=23.11.0
flake8>=6.1.0
mypy>=1.7.1
""",

    "orchestrator/classifier.py": """from typing import Dict
from .protocol import AgentMessage
from shared.llm import call_llm

class MessageClassifier:
    def __init__(self):
        self.agent_types = []  # Will be populated dynamically based on available agents

    def classify_message(self, message: str) -> str:
        \"\"\"Classify a message to determine the appropriate agent.\"\"\"
        prompt = f\"\"\"Classify this message to determine which agent should handle it:
Message: {message}

Available agents: {', '.join(self.agent_types)}

Respond with only the agent name from the available agents list.\"\"\"

        response = call_llm(
            prompt=prompt,
            system_prompt="You are a message classifier. Respond with only the agent name from the available agents list."
        )
        return response.strip().lower()
""",

    "orchestrator/dispatcher.py": """from typing import Dict
import requests
from .protocol import AgentMessage, AgentResponse

class MessageDispatcher:
    def __init__(self):
        self.agent_urls: Dict[str, str] = {}  # Will be populated dynamically

    def register_agent(self, agent_name: str, port: int):
        \"\"\"Register a new agent with its URL.\"\"\"
        self.agent_urls[agent_name] = f"http://localhost:{port}/interact"

    def dispatch_message(self, message: AgentMessage) -> AgentResponse:
        \"\"\"Dispatch a message to the appropriate agent.\"\"\"
        target_url = self.agent_urls.get(message.recipient)
        if not target_url:
            return AgentResponse(
                success=False,
                error=f"Unknown agent: {message.recipient}"
            )

        try:
            response = requests.post(target_url, json=message.dict())
            response.raise_for_status()
            return AgentResponse(success=True, data=response.json())
        except requests.RequestException as e:
            return AgentResponse(
                success=False,
                error=f"Failed to dispatch message: {str(e)}"
            )
""",

    "orchestrator/main.py": """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from .classifier import MessageClassifier
from .dispatcher import MessageDispatcher
from .protocol import AgentMessage, AgentResponse

app = FastAPI()
classifier = MessageClassifier()
dispatcher = MessageDispatcher()

class UserInput(BaseModel):
    message: str
    user_id: str

@app.post("/process-message")
async def process_message(input_data: UserInput):
    \"\"\"Process incoming messages and route to appropriate agent.\"\"\"
    try:
        # Classify the message
        target_agent = classifier.classify_message(input_data.message)
        
        # Create agent message
        message = AgentMessage(
            sender=input_data.user_id,
            recipient=target_agent,
            content=input_data.message
        )
        
        # Dispatch to appropriate agent
        response = dispatcher.dispatch_message(message)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.error)
            
        return response.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
""",

    "orchestrator/memory_manager.py": """from typing import Dict, Any
import json
from pathlib import Path

class MemoryManager:
    def __init__(self):
        self.memory_dir = Path("memory")
        self.memory_dir.mkdir(exist_ok=True)
        
    def save_memory(self, key: str, data: Dict[str, Any]):
        \"\"\"Save memory data to file.\"\"\"
        file_path = self.memory_dir / f"{key}.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
            
    def load_memory(self, key: str) -> Dict[str, Any]:
        \"\"\"Load memory data from file.\"\"\"
        file_path = self.memory_dir / f"{key}.json"
        if not file_path.exists():
            return {}
        with open(file_path, "r") as f:
            return json.load(f)
""",

    "orchestrator/orchestrator.py": """from typing import Dict, Any
from .classifier import MessageClassifier
from .dispatcher import MessageDispatcher
from .protocol import AgentMessage, AgentResponse

class Orchestrator:
    def __init__(self):
        self.classifier = MessageClassifier()
        self.dispatcher = MessageDispatcher()
        
    def process_message(self, message: str, user_id: str) -> AgentResponse:
        \"\"\"Process a message through the orchestrator.\"\"\"
        # Classify message
        target_agent = self.classifier.classify_message(message)
        
        # Create agent message
        agent_message = AgentMessage(
            sender=user_id,
            recipient=target_agent,
            content=message
        )
        
        # Dispatch message
        return self.dispatcher.dispatch_message(agent_message)
""",

    "orchestrator/prompt_builder.py": """from typing import List, Dict, Any

class PromptBuilder:
    def __init__(self):
        self.templates = {{
            "default": "You are an AI agent. Your goals are:\n{{goals}}\n\nYour policies are:\n{{policies}}"
        }}
        
    def build_prompt(self, agent_type: str, context: Dict[str, Any]) -> str:
        \"\"\"Build a prompt for the specified agent type.\"\"\"
        template = self.templates.get(agent_type, self.templates["default"])
        return template.format(**context)
""",

    "orchestrator/protocol.py": """from pydantic import BaseModel
from typing import Optional, Dict, Any

class AgentMessage(BaseModel):
    sender: str
    recipient: str
    content: str

class AgentResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
""",

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
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt} if system_prompt else None,
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"]
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

    "docker-compose.yaml": """version: '3.8'

services:
  orchestrator:
    build: .
    ports:
      - "8001:8001"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./:/app
    command: python -m muvius.orchestrator.main

  # Add your agent services here
  # Example:
  # my-agent:
  #   build: .
  #   ports:
  #     - "8000:8000"
  #   environment:
  #     - OPENAI_API_KEY=${OPENAI_API_KEY}
  #   volumes:
  #     - ./:/app
  #   command: python -m muvius.agents.my-agent.main
""",

    ".env": """# OpenAI API Key
OPENAI_API_KEY=your_api_key_here

# Orchestrator
ORCHESTRATOR_URL=http://localhost:8001

# Database
DATABASE_URL=sqlite:///memory/episodic.db

# Add your agent URLs here
# Example:
# MY_AGENT_URL=http://localhost:8000
""",
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
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            progress.update(task, description="[green]Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            progress.update(task, description="[red]Failed to install dependencies!")
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
            "muvius/agents",
            "muvius/orchestrator",
            "muvius/shared",
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
        
        for file_path, content in TEMPLATES.items():
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
    console.print("3. Start the orchestrator using [yellow]docker-compose up[/yellow]")
    
    console.print("\n[bold]Available Services:[/bold]")
    console.print("• Orchestrator: [yellow]http://localhost:8001[/yellow]")
    console.print("• Agent ports will be assigned automatically starting from 8000")

def update_agents_readme():
    """Update the agents README with the current list of agents."""
    agents_dir = Path("muvius/agents")
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

def create_agent(agent_name: str):
    """Create a new agent with all necessary files and structure."""
    agent_role = agent_name.lower()
    agent_dir = Path(f"muvius/agents/{agent_role}-agent")
    
    # Assign port dynamically based on existing agents
    base_port = 8000
    existing_agents = [d for d in Path("muvius/agents").iterdir() if d.is_dir() and d.name.endswith("-agent")]
    port = base_port + len(existing_agents)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Create agent directory
        task = progress.add_task(f"Creating {agent_name} agent...", total=None)
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Create memory directory
        memory_dir = agent_dir / "memory"
        memory_dir.mkdir(exist_ok=True)
        
        # Create all template files
        for template_path, content in AGENT_TEMPLATES.items():
            full_path = agent_dir / template_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Format template content with agent-specific values
            formatted_content = content.format(
                agent_name=agent_name,
                agent_role=agent_role,
                port=port
            )
            
            with open(full_path, "w") as f:
                f.write(formatted_content)
        
        # Update agents README
        update_agents_readme()
        
        progress.update(task, description=f"[green]{agent_name} agent created successfully!")

def delete_agent(agent_name: str):
    """Delete an existing agent."""
    agent_role = agent_name.lower()
    agent_dir = Path(f"muvius/agents/{agent_role}-agent")
    
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
    
    if not force and Path("muvius").exists():
        console.print("[red]Error: Muvius directory already exists. Use --force to overwrite.")
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

@cli.command()
@click.argument('agent_name')
def create_agent_command(agent_name):
    """Create a new agent with all necessary files and structure."""
    create_agent(agent_name)

@cli.command()
@click.argument('agent_name')
def delete_agent_command(agent_name):
    """Delete an existing agent."""
    delete_agent(agent_name)

if __name__ == '__main__':
    cli() 