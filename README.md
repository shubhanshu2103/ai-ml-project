# Clinical Appointment No-Show Prediction & Agentic Care Coordination System

## Project Overview
An AI-based healthcare operations system that predicts patient appointment no-shows using machine learning and generates actionable care coordination plans using an agentic AI pipeline.

- **Milestone 1 (Mid-Sem):** ML-based no-show prediction using a Decision Tree classifier trained on the [Kaggle No-Show Appointments dataset](https://www.kaggle.com/datasets/joniarroba/noshowappointments).
- **Milestone 2 (End-Sem):** LangGraph-powered agentic assistant that analyzes patient risk factors and generates structured intervention strategies using Groq (Llama 3.1 8B). Includes PDF report export.

## Tech Stack
| Layer | Tools |
|-------|-------|
| Frontend / UI | Streamlit |
| Machine Learning | Scikit-Learn (Decision Tree), SMOTE (imbalanced-learn) |
| Agentic Workflow | LangGraph, Groq (Llama 3.1 8B) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| PDF Export | FPDF2 |

## Live Demo
[Live Link will come Here]

## Local Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set up your Groq API key (required for Milestone 2):
   ```bash
   export GROQ_API_KEY='gsk_your_key_here'
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Repository Structure
```
├── app.py                      # Main Streamlit app (Milestone 1 + 2 integration)
├── requirements.txt            # Python dependencies
├── noshow_model.pkl            # Trained ML model
├── scaler.pkl                  # StandardScaler for feature scaling
├── agents/                     # Milestone 2 — Agentic pipeline
│   ├── __init__.py
│   ├── graph.py                # LangGraph state machine definition
│   ├── nodes.py                # Agent node functions (analyze, intervene, compile)
│   ├── prompts.py              # LLM prompt templates
│   └── state.py                # TypedDict state schema
├── utils/
│   └── pdf_export.py           # PDF report generation
└── README.md
```

## How It Works

### Milestone 1 — No-Show Prediction
Upload the appointment CSV → the Decision Tree model predicts no-show probability for each patient → results are displayed with risk levels and feature importance charts.

### Milestone 2 — Agentic Care Coordination
Select a high-risk patient → the LangGraph agent runs a 3-step pipeline:
1. **Risk Analysis** — LLM explains why this patient is at risk using actual data values
2. **Intervention Generation** — LLM generates specific actions (SMS reminders, barrier removal, follow-up protocols)
3. **Report Compilation** — Structured report with risk summary, interventions, sources, and medical disclaimer

The final report can be exported as a downloadable PDF.

## Team
| Member | Role |
|--------|------|
| Member 1 | ML Pipeline — SMOTE balancing, model retraining |
| Member 2 | Agentic Engine — LangGraph workflow, Groq integration |
| Member 3 | PDF Report Generation |
| Member 4 | Streamlit UI Integration & Deployment |