"""
Prompt templates for the Care Coordination Agent.
Each prompt is designed to produce structured, actionable output
while avoiding hallucinated medical claims.
"""

SYSTEM_PROMPT = (
    "You are a healthcare operations analyst specializing in appointment "
    "no-show prevention and care coordination. You provide actionable, "
    "evidence-based recommendations to help clinics reduce missed appointments. "
    "You NEVER make clinical diagnoses or prescribe treatments. "
    "Your role is strictly operational — scheduling, outreach, barrier removal."
)

RISK_ANALYSIS_PROMPT = """Analyze this patient's appointment no-show risk.

## Patient Data
- Age: {age}
- Gender: {gender}
- Scholarship (Bolsa Família): {scholarship}
- Hypertension: {hypertension}
- Diabetes: {diabetes}
- Alcoholism: {alcoholism}
- Handicap Level: {handicap}
- SMS Received: {sms_received}
- Wait Days (gap between scheduling and appointment): {wait_days}

## Model Prediction
- No-Show Probability: {risk_score:.1f}%
- Risk Level: {risk_level}

## Top Contributing Factors (from ML model)
{top_factors}

Based on the above, provide a concise risk analysis in this exact format:

**Risk Summary:** (2-3 sentences explaining why this patient is at risk)

**Key Contributing Factors:**
1. (Factor 1 with specific data reference)
2. (Factor 2 with specific data reference)
3. (Factor 3 with specific data reference)

Keep it factual and reference the actual data values. Do not speculate beyond what the data shows."""


INTERVENTION_PROMPT = """Based on the following patient risk analysis, generate a specific care coordination intervention plan.

## Risk Analysis
{risk_analysis}

## Patient Context
- No-Show Probability: {risk_score:.1f}%
- Wait Days until appointment: {wait_days}
- SMS Already Sent: {sms_received}

Generate an actionable intervention plan in this exact format:

**Immediate Actions (24-48 hours before appointment):**
1. (Specific action with timing)
2. (Specific action with timing)

**Communication Strategy:**
1. (Channel + message + timing)
2. (Backup channel if no response)

**Barrier Removal Suggestions:**
1. (Specific suggestion based on patient profile)
2. (Alternative if applicable)

**Follow-Up Protocol:**
1. (What to do if patient confirms)
2. (What to do if patient doesn't respond)
3. (Post-appointment follow-up)

Be specific, practical, and actionable. Reference the patient's actual data when making recommendations.
For example, if WaitDays is high, suggest a reminder closer to the date.
If SMS was not sent, recommend sending one immediately."""