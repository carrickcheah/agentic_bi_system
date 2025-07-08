"""
LangGraph assembly for model orchestration.
Combines nodes and edges into a complete workflow graph.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ModelState
from .nodes import (
    initialize_node,
    validate_input_node,
    check_model_health_node,
    select_model_node,
    generate_response_node,
    handle_error_node,
    collect_metrics_node
)
from .edges import (
    route_after_initialization,
    route_after_validation,
    route_after_health_check,
    route_after_model_selection,
    route_after_generation,
    route_after_error_handling
)


def create_model_graph(with_checkpointing: bool = False):
    """
    Create the LangGraph for model orchestration.
    
    Args:
        with_checkpointing: Enable state checkpointing for debugging
        
    Returns:
        Compiled LangGraph workflow
    """
    
    # Create the graph with ModelState
    workflow = StateGraph(ModelState)
    
    # Add all nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("check_health", check_model_health_node)
    workflow.add_node("select_model", select_model_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("handle_error", handle_error_node)
    workflow.add_node("collect_metrics", collect_metrics_node)
    
    # Define the flow with conditional edges
    
    # Entry point
    workflow.set_entry_point("initialize")
    
    # From initialize
    workflow.add_conditional_edges(
        "initialize",
        route_after_initialization,
        {
            "validate_input": "validate_input",
            "collect_metrics": "collect_metrics"
        }
    )
    
    # From validate_input
    workflow.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "check_health": "check_health",
            "collect_metrics": "collect_metrics"
        }
    )
    
    # From check_health
    workflow.add_conditional_edges(
        "check_health",
        route_after_health_check,
        {
            "select_model": "select_model",
            "collect_metrics": "collect_metrics"
        }
    )
    
    # From select_model
    workflow.add_conditional_edges(
        "select_model",
        route_after_model_selection,
        {
            "generate_response": "generate_response",
            "collect_metrics": "collect_metrics"
        }
    )
    
    # From generate_response
    workflow.add_conditional_edges(
        "generate_response",
        route_after_generation,
        {
            "handle_error": "handle_error",
            "collect_metrics": "collect_metrics"
        }
    )
    
    # From handle_error
    workflow.add_conditional_edges(
        "handle_error",
        route_after_error_handling,
        {
            "generate_response": "generate_response",
            "select_model": "select_model",
            "collect_metrics": "collect_metrics"
        }
    )
    
    # collect_metrics always goes to END
    workflow.add_edge("collect_metrics", END)
    
    # Compile the graph
    if with_checkpointing:
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()
    
    return app


def create_simple_model_graph():
    """
    Create a simplified model graph for basic use cases.
    No health checks or complex error handling.
    """
    
    workflow = StateGraph(ModelState)
    
    # Simplified flow: initialize -> select -> generate -> metrics
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("select_model", select_model_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("collect_metrics", collect_metrics_node)
    
    # Simple linear flow
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "select_model")
    workflow.add_edge("select_model", "generate_response")
    workflow.add_edge("generate_response", "collect_metrics")
    workflow.add_edge("collect_metrics", END)
    
    return workflow.compile()


# Helper function to visualize the graph
def visualize_graph(save_path: str = "model_graph.png"):
    """
    Generate a visual representation of the model graph.
    
    Args:
        save_path: Path to save the graph image
    """
    try:
        from IPython.display import Image
        
        graph = create_model_graph()
        img = graph.get_graph().draw_png()
        
        with open(save_path, "wb") as f:
            f.write(img)
        
        return Image(img)
    except Exception as e:
        print(f"Could not visualize graph: {e}")
        return None