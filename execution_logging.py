import uuid
import time
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from agent import agent_executor
from llm import flush_memory

# 1. Structure the Tracing Layers using immutable dataclass architecture
@dataclass
class StepLog:
    """Captures granular information for an individual state update or tool invocation."""
    step_id: str
    node_name: str
    timestamp: str
    latency_ms: float
    message_snapshot: str

@dataclass
class ExecutionTrace:
    """Maintains the full execution metadata for an entire agent run lifecycle."""
    trace_id: str
    user_query: str
    started_at: str
    steps: List[StepLog] = field(default_factory=list)
    finished_at: Optional[str] = None
    total_duration_ms: float = 0.0
    status: str = "PENDING"
    final_response: str = ""

# 2. Instrumented Execution Wrapper
def run_observed_agent(user_query: str) -> Dict[str, Any]:
    """
    Runs the admissions agent graph while explicitly tracing the message history,
    measuring node latency, and preparing a structured JSON-ready log.
    """
    trace = ExecutionTrace(
        trace_id=str(uuid.uuid4())[:8],
        user_query=user_query,
        started_at=datetime.utcnow().isoformat()
    )
    
    inputs = {"messages": [("user", user_query)]}
    t_start = time.perf_counter()
    
    try:
        last_step_time = time.perf_counter()
        
        # Stream the full execution values
        for output in agent_executor.stream(inputs, stream_mode="values"):
            if "messages" in output and output["messages"]:
                current_time = time.perf_counter()
                latency = (current_time - last_step_time) * 1000
                last_step_time = current_time
                
                latest_msg = output["messages"][-1]
                
                # Determine originating structure
                node_origin = "orchestrator" if latest_msg.type == "ai" else "tools_node"
                if latest_msg.type == "user":
                    node_origin = "user_input"
                
                # Log step details
                step = StepLog(
                    step_id=str(uuid.uuid4())[:6],
                    node_name=node_origin,
                    timestamp=datetime.utcnow().isoformat(),
                    latency_ms=round(latency, 2),
                    message_snapshot=latest_msg.content if latest_msg.content else f"Tool Call Invocation: {getattr(latest_msg, 'tool_calls', [])}"
                )
                trace.steps.append(step)
                trace.final_response = latest_msg.content
        
        trace.status = "SUCCESS"
        
    except Exception as execution_error:
        trace.status = "FAILED"
        trace.final_response = f"Traced System Exception: {str(execution_error)}"
        
    finally:
        # Conclude timestamps and performance calculations
        trace.finished_at = datetime.utcnow().isoformat()
        trace.total_duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
        flush_memory()
        
    # Serialize the complete telemetry trace objects to raw dictionary format
    return {
        "trace_id": trace.trace_id,
        "query": trace.user_query,
        "status": trace.status,
        "execution_time_ms": trace.total_duration_ms,
        "timestamps": {"start": trace.started_at, "end": trace.finished_at},
        "steps_logged": [
            {
                "id": s.step_id,
                "node": s.node_name,
                "latency_ms": s.latency_ms,
                "snapshot": s.message_snapshot[:150] + "..." if len(s.message_snapshot) > 150 else s.message_snapshot
            } for s in trace.steps
        ],
        "final_output": trace.final_response
    }

# Execution Test
if __name__ == "__main__":
    test_query = "What documents do I need to send for the Agentic AI Bootcamp, and check ID 101 status?"
    print(f"Executing and tracing query: '{test_query}'")
    
    telemetry_log = run_observed_agent(test_query)
    print("\n--- STRUCTURED TRACE JSON OUTPUT ---")
    print(json.dumps(telemetry_log, indent=2))
