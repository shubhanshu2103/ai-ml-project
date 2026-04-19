"""
State schema for the Care Coordination Agent.
Defines the TypedDict that flows through the LangGraph state machine.
"""

from typing import TypedDict, List, Optional


class CareCoordinationState(TypedDict):
    """State object that flows through every node in the LangGraph pipeline."""

    # ── Inputs (populated before graph invocation) ──
    patient_data: dict              # Raw patient features (Age, Gender, WaitDays, etc.)
    risk_score: float               # No-show probability from ML model (0-100)
    risk_level: str                 # "High Risk" or "Low Risk"
    feature_importances: dict       # {feature_name: importance_score}

    # ── Intermediate (populated by agent nodes) ──
    risk_analysis: str              # LLM-generated risk factor analysis
    intervention_plan: str          # LLM-generated intervention strategies

    # ── Output ──
    final_report: dict              # Complete structured report with all sections
    error: Optional[str]            # Error message if any node fails