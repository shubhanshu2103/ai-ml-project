"""
Agent node functions for the Care Coordination LangGraph pipeline.
Each function represents one step in the state machine.
"""

import os
import time
from datetime import datetime
from groq import Groq
from agents.state import CareCoordinationState
from agents.prompts import SYSTEM_PROMPT, RISK_ANALYSIS_PROMPT, INTERVENTION_PROMPT


def _call_llm(prompt: str, max_retries: int = 3) -> str:
    """
    Call Groq LLM with retry logic for rate limiting.
    Uses llama-3.1-8b-instant (free tier, fast).
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)          # 2s, 4s, 8s
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM call failed after {max_retries} retries: {e}")


# ─────────────────────────────────────────────
# Node 1: Analyze Risk Factors
# ─────────────────────────────────────────────
def analyze_risk(state: CareCoordinationState) -> dict:
    """
    Takes patient data + ML prediction, asks the LLM to explain
    WHY this patient is at risk using the actual feature values.
    """
    try:
        patient = state["patient_data"]
        importances = state["feature_importances"]

        # Format top contributing factors for the prompt
        sorted_factors = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_factors = "\n".join(
            f"- {name}: importance = {score:.4f}" for name, score in sorted_factors[:5]
        )

        prompt = RISK_ANALYSIS_PROMPT.format(
            age=patient.get("Age", "N/A"),
            gender="Male" if patient.get("Gender") == 1 else "Female",
            scholarship=patient.get("Scholarship", 0),
            hypertension=patient.get("Hypertension", 0),
            diabetes=patient.get("Diabetes", 0),
            alcoholism=patient.get("Alcoholism", 0),
            handicap=patient.get("Handicap", 0),
            sms_received=patient.get("SMS_received", 0),
            wait_days=patient.get("WaitDays", "N/A"),
            risk_score=state["risk_score"],
            risk_level=state["risk_level"],
            top_factors=top_factors,
        )

        analysis = _call_llm(prompt)
        return {"risk_analysis": analysis, "error": None}

    except Exception as e:
        return {
            "risk_analysis": "Risk analysis could not be generated due to a system error.",
            "error": f"Risk analysis failed: {str(e)}",
        }


# ─────────────────────────────────────────────
# Node 2: Generate Intervention Strategies
# ─────────────────────────────────────────────
def generate_intervention(state: CareCoordinationState) -> dict:
    """
    Takes the risk analysis output, generates specific, actionable
    intervention recommendations for care coordinators.
    """
    try:
        patient = state["patient_data"]

        prompt = INTERVENTION_PROMPT.format(
            risk_analysis=state["risk_analysis"],
            risk_score=state["risk_score"],
            wait_days=patient.get("WaitDays", "N/A"),
            sms_received="Yes" if patient.get("SMS_received") == 1 else "No",
        )

        plan = _call_llm(prompt)
        return {"intervention_plan": plan, "error": None}

    except Exception as e:
        return {
            "intervention_plan": "Intervention plan could not be generated due to a system error.",
            "error": f"Intervention generation failed: {str(e)}",
        }


# ─────────────────────────────────────────────
# Node 3: Compile Final Structured Report
# ─────────────────────────────────────────────
def compile_report(state: CareCoordinationState) -> dict:
    """
    Assembles all outputs into the final structured report dict.
    This node is deterministic — no LLM call, just formatting.
    """
    patient = state["patient_data"]

    report = {
        "title": "Care Coordination Report",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_summary": {
            "age": patient.get("Age", "N/A"),
            "gender": "Male" if patient.get("Gender") == 1 else "Female",
            "wait_days": patient.get("WaitDays", "N/A"),
            "sms_received": "Yes" if patient.get("SMS_received") == 1 else "No",
            "conditions": [
                c for c in ["Hypertension", "Diabetes", "Alcoholism"]
                if patient.get(c) == 1
            ],
        },
        "risk_score": f"{state['risk_score']:.1f}%",
        "risk_level": state["risk_level"],
        "risk_analysis": state["risk_analysis"],
        "intervention_plan": state["intervention_plan"],
        "sources": [
            "ML Model: Decision Tree Classifier trained on Kaggle No-Show Appointments dataset",
            "AHRQ Care Coordination Framework (general best practices)",
            "Patient Engagement Strategies for Reducing No-Shows (operational guidelines)",
        ],
        "disclaimer": (
            "⚕️ DISCLAIMER: This report is generated by an AI system for "
            "operational decision support only. It does NOT constitute medical "
            "advice, diagnosis, or treatment. All intervention strategies must "
            "be reviewed and approved by qualified healthcare staff before "
            "implementation. Patient privacy and consent must be maintained "
            "per applicable regulations."
        ),
    }

    return {"final_report": report, "error": None}