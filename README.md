# Muvius Framework

<p align="center">
  <img src="https://github.com/user-attachments/assets/5b53c0e6-67c5-431e-a7a4-704b243c9588" 
       alt="Muvius Framework Logo" 
       width="300" 
       style="padding: 10px; border-radius: 8px;"/>
</p>

<p align="center">
  <b>A G E N T - T O - A G E N T 
  <br>I N T E L L I G E N C E</b><br>
  <i>by MUVIO AI</i>
</p>

## What is Muvius?

Muvius is a powerful framework for building and managing AI agents with advanced memory systems and seamless inter-agent communication. It provides a structured approach to creating intelligent agents that can:

- Maintain context through multiple memory systems
- Communicate with other agents using standardized protocols
- Handle complex tasks through role-based interactions
- Scale from simple to complex multi-agent systems

## Key Features

### 1. Memory Systems
- **Episodic Memory**: Tracks conversation history and interactions
- **Procedural Memory**: Stores agent roles, rules, and procedures
- **Vector Store**: Enables semantic search and context understanding

### 2. Agent Communication
- Standardized message protocols
- Dynamic message routing
- Inter-agent collaboration support

### 3. Easy Integration
- FastAPI-based REST endpoints
- Docker support for easy deployment
- CLI tools for quick setup and management

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                     Muvius Framework                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Orchestrator│    │   Agents    │    │   Shared    │  │
│  │             │    │             │    │ Services    │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │         │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐  │
│  │  Message    │    │  Memory     │    │  Utilities  │  │
│  │  Routing    │    │  Systems    │    │  & Tools    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install Muvius
pip install muvius
```

### 2. Initialize Your Project

```bash
# Create a new Muvius project
muvius init

# Create your first agent
muvius create-agent my-agent
```

### 3. Start Development

```bash
# Start the orchestrator
python -m muvius.orchestrator.main

# Start your agent (in a new terminal)
python -m muvius.agents.my-agent.main
```

## Project Structure

```
muvius/
├── agents/              # Agent modules
│   └── my-agent/       # Your custom agent
│       ├── memory/     # Memory systems
│       ├── main.py     # Agent logic
│       └── routes.py   # API endpoints
├── orchestrator/       # Message routing
└── shared/            # Common utilities
```

## CLI Commands

```bash
# Initialize framework
muvius init

# Create new agent
muvius create-agent <agent-name>

# Delete agent
muvius delete-agent <agent-name>
```

## Memory Systems

### Episodic Memory
- SQLite-based conversation history
- Automatic message logging
- Context retrieval for conversations

### Procedural Memory
- YAML-based configuration
- Role and policy definitions
- Procedure storage and retrieval

### Vector Store
- Semantic search capabilities
- Context understanding
- Long-term memory storage

## API Endpoints

### Orchestrator
- `POST /process-message`: Route messages to agents
- `GET /ping`: Health check

### Agents
- `POST /interact`: Handle agent interactions
- `GET /ping`: Health check

## Development

### Prerequisites
- Python 3.8+
- pip
- virtualenv

### Environment Setup
1. Create `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
```

### Running Tests
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Support

For support, please:
1. Check the documentation
2. Open an issue on GitHub
3. Contact the Muvio AI team

---

<p align="center">
  <i>Built with ❤️ by <a href="https://github.com/varun-singhh">@varun-singhh</a></i>
</p>
