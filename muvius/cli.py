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

# ... (rest of the code remains the same until the cli definition) ...

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
        subtitle="v0.1.3",
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

# Export the cli function
__all__ = ['cli'] 