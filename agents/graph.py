"""
LangGraph state machine for the Care Coordination Agent.

Workflow:
    ┌──────────────┐     ┌─────────────────────┐     ┌────────────────┐
    │ analyze_risk │ ──▶ │ generate_intervention│ ──▶ │ compile_report │ ──▶ END
    └──────────────┘     └─────────────────────┘     └────────────────┘

Each node reads from and writes to a shared CareCoordinationState dict.
If any node encounters an error, it writes to state["error"] and the
pipeline continues (graceful degradation, not hard crash).
"""

from langgraph.graph import StateGraph, END
from agents.state import CareCoordinationState
from agents.nodes import analyze_risk, generate_intervention, compile_report


def build_graph():
    """Build and compile the LangGraph state machine."""

    workflow = StateGraph(CareCoordinationState)

    # Register nodes
    workflow.add_node("analyze_risk", analyze_risk)
    workflow.add_node("generate_intervention", generate_intervention)
    workflow.add_node("compile_report", compile_report)

    # Define edges (sequential pipeline)
    workflow.set_entry_point("analyze_risk")
    workflow.add_edge("analyze_risk", "generate_intervention")
    workflow.add_edge("generate_intervention", "compile_report")
    workflow.add_edge("compile_report", END)

    return workflow.compile()


def generate_care_plan(
    patient_data: dict,
    risk_score: float,
    risk_level: str,
    feature_importances: dict,
) -> dict:
    """
    Main entry point — called by the Streamlit app (Task 4).

    Args:
        patient_data: dict of patient features
            e.g. {"Age": 34, "Gender": 1, "WaitDays": 45, ...}
        risk_score: float, no-show probability as percentage (0-100)
        risk_level: str, "High Risk" or "Low Risk"
        feature_importances: dict mapping feature names to importance scores
            e.g. {"WaitDays": 0.42, "Age": 0.18, ...}

    Returns:
        dict with keys:
            - "final_report": dict with all report sections
            - "error": str or None
    """

    graph = build_graph()

    initial_state: CareCoordinationState = {
        "patient_data": patient_data,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "feature_importances": feature_importances,
        "risk_analysis": "",
        "intervention_plan": "",
        "final_report": {},
        "error": None,
    }

    result = graph.invoke(initial_state)
    return result


# ─────────────────────────────────────────────
# Quick test (run this file directly to verify)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os

    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️  Set GROQ_API_KEY environment variable first:")
        print("   export GROQ_API_KEY='gsk_your_key_here'")
        exit(1)

    # Sample patient data matching the Kaggle dataset features
    sample_patient = {
        "Age": 62,
        "Gender": 0,            # 0 = Female, 1 = Male
        "Scholarship": 0,       # Bolsa Família
        "Hypertension": 1,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handicap": 0,
        "SMS_received": 0,
        "WaitDays": 45,
    }

    sample_importances = {
        "WaitDays": 0.4200,
        "Age": 0.1850,
        "SMS_received": 0.1200,
        "Scholarship": 0.0900,
        "Hypertension": 0.0750,
        "Handicap": 0.0500,
        "Diabetes": 0.0300,
        "Alcoholism": 0.0200,
        "Gender": 0.0100,
    }

    print("=" * 60)
    print("  CARE COORDINATION AGENT — TEST RUN")
    print("=" * 60)

    result = generate_care_plan(
        patient_data=sample_patient,
        risk_score=73.5,
        risk_level="High Risk",
        feature_importances=sample_importances,
    )

    if result.get("error"):
        print(f"\n⚠️  Error: {result['error']}")

    report = result.get("final_report", {})
    print(f"\n {report.get('title', 'No Title')}")
    print(f" Generated: {report.get('generated_at', 'N/A')}")
    print(f" Risk: {report.get('risk_score')} ({report.get('risk_level')})")
    print(f"\n--- RISK ANALYSIS ---\n{report.get('risk_analysis', 'N/A')}")
    print(f"\n--- INTERVENTION PLAN ---\n{report.get('intervention_plan', 'N/A')}")
    print(f"\n--- SOURCES ---")
    for src in report.get("sources", []):
        print(f"  • {src}")
    print(f"\n--- DISCLAIMER ---\n{report.get('disclaimer', 'N/A')}")