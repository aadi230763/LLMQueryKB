#!/usr/bin/env python3
"""
Semantic Router for LLM Gateway
================================
A production-grade query router that dynamically selects the optimal LLM
based on semantic complexity analysis of user queries.

Author: Senior AI Engineer
Date: January 2026

Note: Uses sentence-transformers directly for Python 3.9 compatibility.
"""

import random
import time
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich import box


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Tier(Enum):
    """Model tier classification based on capability and cost."""
    TIER_1_COMPLEX = "tier_1"      # Expensive - Complex Reasoning
    TIER_2_MID = "tier_2"          # Moderate - Summarization
    TIER_3_SIMPLE = "tier_3"       # Cheap - Chitchat


@dataclass
class ModelConfig:
    """Configuration for an LLM provider."""
    name: str
    provider: str
    tier: Tier
    cost_per_1k_tokens: float  # Approximate cost for display
    color: str  # Rich color for UI


# Model Registry - Organized by Tiers
MODEL_REGISTRY: Dict[str, List[ModelConfig]] = {
    # Tier 1: Complex/Reasoning (Expensive - Red)
    "tier_1": [
        ModelConfig("GPT-4o", "OpenAI", Tier.TIER_1_COMPLEX, 0.015, "red"),
        ModelConfig("Claude 3.5 Sonnet", "Anthropic", Tier.TIER_1_COMPLEX, 0.015, "red"),
    ],
    # Tier 2: Summarization/Mid (Moderate - Yellow)
    "tier_2": [
        ModelConfig("Gemini Pro", "Google", Tier.TIER_2_MID, 0.00125, "yellow"),
        ModelConfig("Gemini Flash", "Google", Tier.TIER_2_MID, 0.00075, "yellow"),
        ModelConfig("Mistral Large", "Mistral AI", Tier.TIER_2_MID, 0.008, "yellow"),
    ],
    # Tier 3: Chitchat/Simple (Cheap - Green)
    "tier_3": [
        ModelConfig("Llama 3 70B", "OpenRouter", Tier.TIER_3_SIMPLE, 0.0005, "green"),
        ModelConfig("Llama 3 8B", "OpenRouter", Tier.TIER_3_SIMPLE, 0.0001, "green"),
    ],
}

# Route to Tier mapping
ROUTE_TIER_MAP: Dict[Optional[str], str] = {
    "complex_reasoning": "tier_1",
    "summarization": "tier_2",
    "chitchat": "tier_3",
    None: "tier_2",  # Default fallback
}


# ============================================================================
# SEMANTIC ROUTES DEFINITION
# ============================================================================

@dataclass
class Route:
    """A semantic route with training utterances."""
    name: str
    utterances: List[str]


def create_semantic_routes() -> List[Route]:
    """
    Create semantic routes with diverse training utterances.
    These utterances train the router to classify queries semantically,
    not based on keywords.
    """
    
    # Route 1: Complex Reasoning - Requires deep analysis, multi-step logic
    complex_reasoning = Route(
        name="complex_reasoning",
        utterances=[
            "Analyze the causal relationship between climate policy and economic growth in developing nations",
            "Compare and contrast the architectural decisions in microservices vs monolithic systems with trade-offs",
            "What are the second-order effects of implementing this authentication strategy?",
            "Explain the mathematical proof behind transformer attention mechanisms",
            "Debug this race condition and explain why the mutex implementation fails under high concurrency",
            "Evaluate the ethical implications of using AI in healthcare diagnostics with regulatory considerations",
        ],
    )
    
    # Route 2: Summarization - Condensing, extracting key points, reformatting
    summarization = Route(
        name="summarization",
        utterances=[
            "Give me a brief summary of this research paper",
            "What are the main points discussed in this document?",
            "Can you condense this article into 3 bullet points?",
            "Extract the key takeaways from this meeting transcript",
            "Summarize the findings from chapters 3 through 5",
            "Create an executive summary of this quarterly report",
        ],
    )
    
    # Route 3: Chitchat - Simple queries, greetings, basic facts
    chitchat = Route(
        name="chitchat",
        utterances=[
            "Hello, how are you today?",
            "What's the capital of France?",
            "Thanks for your help!",
            "Can you tell me a fun fact?",
            "What time is it in Tokyo?",
            "Who wrote Romeo and Juliet?",
        ],
    )
    
    return [complex_reasoning, summarization, chitchat]


# ============================================================================
# SEMANTIC ROUTER CLASS
# ============================================================================

class SemanticQueryRouter:
    """
    Production-grade semantic router for LLM query classification.
    Uses embedding-based similarity to route queries to appropriate model tiers.
    """
    
    def __init__(self, console: Console):
        self.console = console
        self.encoder: Optional[SentenceTransformer] = None
        self.routes: List[Route] = []
        self.route_embeddings: Dict[str, np.ndarray] = {}
        self.similarity_threshold: float = 0.4
        self._initialize_router()
    
    def _initialize_router(self) -> None:
        """Initialize the encoder and compute route embeddings."""
        with Live(
            Spinner("dots", text="[cyan]Initializing Semantic Router...[/cyan]"),
            console=self.console,
            refresh_per_second=10,
        ) as live:
            # Load the sentence transformer model
            live.update(Spinner("dots", text="[cyan]Loading encoder model (all-MiniLM-L6-v2)...[/cyan]"))
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create routes
            live.update(Spinner("dots", text="[cyan]Building semantic routes...[/cyan]"))
            self.routes = create_semantic_routes()
            
            # Pre-compute embeddings for all route utterances
            live.update(Spinner("dots", text="[cyan]Vectorizing training utterances...[/cyan]"))
            for route in self.routes:
                embeddings = self.encoder.encode(route.utterances, convert_to_numpy=True)
                # Store the mean embedding for each route (centroid approach)
                self.route_embeddings[route.name] = np.mean(embeddings, axis=0)
            
            time.sleep(0.3)  # Brief pause for visual effect
        
        self.console.print("[green]✓[/green] Semantic Router initialized successfully!\n")
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def classify_query(self, query: str) -> Tuple[Optional[str], ModelConfig, float]:
        """
        Classify a query and return the route name and selected model.
        
        Args:
            query: The user's input query
            
        Returns:
            Tuple of (route_name, selected_model_config, confidence_score)
        """
        # Encode the query
        query_embedding = self.encoder.encode(query, convert_to_numpy=True)
        
        # Find the most similar route
        best_route: Optional[str] = None
        best_similarity: float = -1.0
        
        for route_name, route_embedding in self.route_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, route_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_route = route_name
        
        # Apply threshold - if similarity is too low, use default
        if best_similarity < self.similarity_threshold:
            best_route = None
        
        # Map route to tier and select a model
        tier_key = ROUTE_TIER_MAP.get(best_route, "tier_2")
        available_models = MODEL_REGISTRY[tier_key]
        selected_model = random.choice(available_models)
        
        return best_route, selected_model, best_similarity
    
    def mock_llm_response(self, query: str, model: ModelConfig) -> str:
        """Generate a mock LLM response for demonstration."""
        responses = {
            "tier_1": (
                f"[Deep Analysis from {model.name}]\n\n"
                f"This is a complex query that requires careful reasoning. "
                f"After analyzing the multi-faceted aspects of your question, "
                f"I've identified several key considerations...\n\n"
                f"• Primary insight: The relationship you're asking about involves "
                f"multiple interdependent variables.\n"
                f"• Secondary consideration: Historical context suggests a nuanced interpretation.\n"
                f"• Recommendation: Further analysis with specific domain data would "
                f"strengthen these conclusions."
            ),
            "tier_2": (
                f"[Summary from {model.name}]\n\n"
                f"**Key Points:**\n"
                f"1. The main topic centers around the core subject matter\n"
                f"2. Important findings include several actionable insights\n"
                f"3. The conclusion supports the initial hypothesis\n\n"
                f"*Generated summary based on document analysis.*"
            ),
            "tier_3": (
                f"[Quick Response from {model.name}]\n\n"
                f"Hello! Happy to help with that. Here's a straightforward answer "
                f"to your question. Let me know if you need anything else! 😊"
            ),
        }
        return responses.get(model.tier.value, responses["tier_2"])


# ============================================================================
# TERMINAL UI COMPONENTS
# ============================================================================

class DashboardUI:
    """Rich terminal UI for the Semantic Router dashboard."""
    
    def __init__(self, console: Console):
        self.console = console
    
    def print_header(self) -> None:
        """Print the application header."""
        header_text = Text()
        header_text.append("🚀 ", style="bold")
        header_text.append("SEMANTIC ROUTER", style="bold cyan")
        header_text.append(" | ", style="dim")
        header_text.append("LLM Gateway v1.0", style="italic dim")
        
        self.console.print(Panel(
            Align.center(header_text),
            box=box.DOUBLE,
            style="cyan",
            padding=(1, 2),
        ))
        self.console.print()
    
    def print_tier_legend(self) -> None:
        """Print the tier color legend."""
        legend = Table(show_header=False, box=None, padding=(0, 2))
        legend.add_column(justify="center")
        legend.add_column(justify="center")
        legend.add_column(justify="center")
        legend.add_row(
            "[red]● Tier 1: Complex[/red]",
            "[yellow]● Tier 2: Mid[/yellow]",
            "[green]● Tier 3: Simple[/green]",
        )
        self.console.print(Align.center(legend))
        self.console.print()
    
    def display_query_panel(self, query: str) -> None:
        """Display the user query in a styled panel."""
        self.console.print(Panel(
            f"[white]{query}[/white]",
            title="[bold blue]📝 User Query[/bold blue]",
            title_align="left",
            border_style="blue",
            padding=(1, 2),
        ))
    
    def display_routing_result(
        self, 
        route_name: Optional[str], 
        model: ModelConfig,
        confidence: float
    ) -> None:
        """Display the routing decision in a color-coded table."""
        
        # Create results table
        table = Table(
            title="[bold]🎯 Routing Decision[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
        )
        
        table.add_column("Property", style="cyan", width=20)
        table.add_column("Value", width=40)
        table.add_column("Status", justify="center", width=15)
        
        # Route row
        route_display = route_name or "default (summarization)"
        route_color = {
            "complex_reasoning": "red",
            "summarization": "yellow",
            "chitchat": "green",
        }.get(route_name, "yellow")
        
        table.add_row(
            "Selected Route",
            f"[{route_color}]{route_display}[/{route_color}]",
            f"[{route_color}]●[/{route_color}]",
        )
        
        # Model row
        table.add_row(
            "Selected Model",
            f"[{model.color} bold]{model.name}[/{model.color} bold]",
            f"[{model.color}]●[/{model.color}]",
        )
        
        # Provider row
        table.add_row(
            "Provider",
            f"[white]{model.provider}[/white]",
            "━",
        )
        
        # Confidence score row
        conf_color = "green" if confidence > 0.6 else "yellow" if confidence > 0.4 else "red"
        table.add_row(
            "Confidence",
            f"[{conf_color}]{confidence:.1%}[/{conf_color}]",
            f"[{conf_color}]●[/{conf_color}]",
        )
        
        # Cost indicator row
        cost_indicator = {
            "red": "[red]$$$ Expensive[/red]",
            "yellow": "[yellow]$$ Moderate[/yellow]",
            "green": "[green]$ Cheap[/green]",
        }.get(model.color, "[yellow]$$ Moderate[/yellow]")
        
        table.add_row(
            "Cost Tier",
            cost_indicator,
            f"~${model.cost_per_1k_tokens}/1K tokens",
        )
        
        self.console.print()
        self.console.print(table)
    
    def display_response(self, response: str, model: ModelConfig) -> None:
        """Display the mock LLM response."""
        self.console.print()
        self.console.print(Panel(
            f"[white]{response}[/white]",
            title=f"[bold {model.color}]🤖 Response from {model.name}[/bold {model.color}]",
            title_align="left",
            border_style=model.color,
            padding=(1, 2),
        ))
    
    def display_thinking_spinner(self) -> Live:
        """Return a Live context manager with thinking spinner."""
        return Live(
            Spinner("dots12", text="[cyan]Analyzing query semantics...[/cyan]"),
            console=self.console,
            refresh_per_second=10,
        )
    
    def print_separator(self) -> None:
        """Print a visual separator between queries."""
        self.console.print()
        self.console.rule("[dim]─[/dim]", style="dim")
        self.console.print()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main() -> None:
    """Main application entry point."""
    console = Console()
    
    # Clear screen and print header
    console.clear()
    
    # Initialize UI
    ui = DashboardUI(console)
    ui.print_header()
    ui.print_tier_legend()
    
    # Initialize router
    router = SemanticQueryRouter(console)
    
    console.print("[dim]Type your query and press Enter. Type 'quit' or 'exit' to stop.[/dim]\n")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            console.print("[bold cyan]>[/bold cyan] ", end="")
            query = input().strip()
            
            # Check for exit commands
            if query.lower() in ("quit", "exit", "q"):
                console.print("\n[yellow]👋 Goodbye! Thanks for using Semantic Router.[/yellow]\n")
                break
            
            # Skip empty queries
            if not query:
                console.print("[dim]Please enter a query.[/dim]\n")
                continue
            
            console.print()
            
            # Display the query
            ui.display_query_panel(query)
            
            # Classify with spinner
            with ui.display_thinking_spinner() as live:
                time.sleep(0.5)  # Simulated processing time for effect
                live.update(Spinner("dots12", text="[cyan]Vectorizing query...[/cyan]"))
                time.sleep(0.3)
                live.update(Spinner("dots12", text="[cyan]Computing semantic similarity...[/cyan]"))
                route_name, selected_model, confidence = router.classify_query(query)
                time.sleep(0.2)
                live.update(Spinner("dots12", text=f"[cyan]Selected: {selected_model.name}[/cyan]"))
                time.sleep(0.2)
            
            # Display routing result
            ui.display_routing_result(route_name, selected_model, confidence)
            
            # Generate and display mock response
            with Live(
                Spinner("dots", text=f"[{selected_model.color}]Generating response from {selected_model.name}...[/{selected_model.color}]"),
                console=console,
                refresh_per_second=10,
            ):
                time.sleep(0.8)  # Simulated API latency
                response = router.mock_llm_response(query, selected_model)
            
            ui.display_response(response, selected_model)
            ui.print_separator()
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]👋 Interrupted. Goodbye![/yellow]\n")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            continue


if __name__ == "__main__":
    main()
