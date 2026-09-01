"""
main_deberta.py
---------------
FastAPI application for the fine-tuned DeBERTa-v3-small sales sentiment analysis.
Mirrors main.py structure so the same index.html template renders correctly.
"""

import io
import json
import os
import sys
import uuid
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add project root to sys.path so app package resolves correctly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.transformer_service import TransformerSentimentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sales Sentiment – DeBERTa API",
    description=(
        "Fine-tuned DeBERTa-v3-small model for sales-domain sentiment classification. "
        "Labels: Positive / Neutral / Negative"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
static_path = Path(__file__).parent / "static"
if not static_path.exists():
    static_path.mkdir()
app.mount("/static", StaticFiles(directory=static_path), name="static")

templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# ---------------------------------------------------------------------------
# Global state (mirrors main.py)
# ---------------------------------------------------------------------------

processed_data: pd.DataFrame | None = None
single_analysis_log: list = []
analysis_history: dict = {}

# ---------------------------------------------------------------------------
# Singleton transformer service (loaded once at startup)
# ---------------------------------------------------------------------------

_svc: TransformerSentimentService | None = None


def get_service() -> TransformerSentimentService:
    global _svc
    if _svc is None:
        _svc = TransformerSentimentService()
    return _svc


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SentimentRequest(BaseModel):
    text: str


class BatchSentimentRequest(BaseModel):
    texts: list[str]


class SentimentEditRequest(BaseModel):
    session_id: str
    original_sentiment: str
    new_sentiment: str
    confidence: float
    text: str
    reason: str = ""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

MODEL_NAME = "DeBERTa-v3-small (deberta_sales_v1)"


def _get_confidence_color(confidence: float) -> str:
    if confidence >= 0.75:
        return "success"
    if confidence >= 0.50:
        return "warning"
    return "danger"


def _calculate_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics consumed by the Jinja2 template.
    Identical logic to main.py so the same index.html renders correctly.
    """
    if df is None or df.empty:
        return {}

    stats: dict = {}
    if "Company Name" in df.columns and "Sentiment" in df.columns:
        grouped = df.groupby("Company Name")
        company_stats = []
        for company, grp in grouped:
            total = len(grp)
            sentiment_col = grp["Sentiment"].str.lower() if "Sentiment" in grp else pd.Series(dtype=str)
            positive = int((sentiment_col == "positive").sum())
            negative = int((sentiment_col == "negative").sum())
            neutral = int((sentiment_col == "neutral").sum())
            company_stats.append({
                "company": company,
                "Total": total,
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
            })
        stats["company_stats"] = company_stats

    if "Sentiment" in df.columns:
        sentiment_col = df["Sentiment"].str.lower()
        stats["total_pos"] = int((sentiment_col == "positive").sum())
        stats["total_neg"] = int((sentiment_col == "negative").sum())
        stats["total_neu"] = int((sentiment_col == "neutral").sum())

    return stats


def _calculate_company_sentiment_analysis(df: pd.DataFrame, company_name: str) -> dict:
    """Detailed per-company analysis — mirrors main.py."""
    if df is None or df.empty:
        return {}
    filtered = df[df["Company Name"].str.lower() == company_name.lower()] if "Company Name" in df.columns else pd.DataFrame()
    if filtered.empty:
        return {}

    records = []
    for _, row in filtered.iterrows():
        records.append({
            "Company Name": row.get("Company Name", ""),
            "Sentiment": row.get("Sentiment", ""),
            "Confidence": row.get("Confidence", 0.0),
        })
    records.sort(key=lambda k: k.get("Confidence", 0), reverse=True)
    return {"records": records}


def _predict_for_remark(remark: str) -> dict:
    """
    Thin wrapper: calls the transformer service and converts the label to
    lowercase so it matches the main app's convention ('positive', not 'Positive').
    """
    result = get_service().predict_one(remark)
    return {
        "sentiment": result["sentiment"].lower(),
        "confidence": result["confidence"],
        "method": "deberta",
    }


def _template_defaults() -> dict:
    """Shared template context keys."""
    return {
        "single_result": None,
        "batch_result": None,
        "input_text": "",
        "session_id": str(uuid.uuid4()),
        "data": [],
        "total_records": 0,
        "page": 1,
        "total_pages": 1,
        "companies": [],
        "selected_company": "",
        "search_query": "",
        "selected_sentiment": "",
        "company_stats": [],
        "overall_stats": {},
        "sentiment_pie_data": "[]",
        "company_pie_data": "[]",
        "deberta_mode": True,
        "model_name": MODEL_NAME,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def read_root(request: Request):
    ctx = _template_defaults()
    ctx["request"] = request
    return templates.TemplateResponse("index.html", ctx)


@app.get("/health")
async def health_check():
    model_dir = Path(__file__).parent / "models" / "deberta_sales_v1"
    return {
        "status": "ok",
        "model": "deberta_sales_v1",
        "model_exists": model_dir.exists(),
    }


@app.post("/analyze")
async def analyze_text(text: str = Form(...)):
    """Analyse a single remark and return JSON."""
    try:
        result = get_service().predict_one(text)
        return {
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "method": "deberta",
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Error analysing text: {exc}"})


@app.post("/batch_analyze")
async def batch_analyze(request: Request, file: UploadFile = File(None)):
    global processed_data
    try:
        if file is None:
            return JSONResponse(status_code=400, content={"message": "No file uploaded."})

        contents = await file.read()
        filename = file.filename or ""

        if filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={"message": "Unsupported file type. Please upload .xlsx or .csv"})

        required = {"Company Name", "Opportunity Name", "Remarks"}
        missing = required - set(df.columns)
        if missing:
            return JSONResponse(status_code=400, content={"message": f"Missing required columns: {missing}"})

        remarks = df["Remarks"].fillna("").tolist()
        batch_results = get_service().predict_batch(remarks)

        df["sentiment"] = [r["sentiment"].lower() for r in batch_results]
        df["Sentiment"] = [r["sentiment"] for r in batch_results]
        df["confidence"] = [r["confidence"] for r in batch_results]
        df["Confidence"] = df["confidence"]
        df["Method"] = "deberta"
        df["Confidence_Color"] = df["confidence"].apply(_get_confidence_color)

        processed_data = df

        stats = _calculate_statistics(df)
        records = df.to_dict(orient="records")

        ctx = _template_defaults()
        ctx.update({
            "request": request,
            "batch_result": True,
            "records": records,
            "data": records,
            "total_records": len(records),
            **stats,
        })
        ctx["index"] = True
        return templates.TemplateResponse("index.html", ctx)

    except Exception as exc:
        return JSONResponse(status_code=500, content={"message": f"Error processing file: {exc}"})


@app.post("/add_remark")
async def add_remark(
    request: Request,
    company: str = Form(...),
    opportunity: str = Form(...),
    remark: str = Form(...),
):
    global processed_data
    try:
        if processed_data is None or processed_data.empty:
            return JSONResponse(status_code=400, content={"message": "No data available. Please upload a file first."})

        pred = _predict_for_remark(remark)
        new_row = {
            "Company Name": company,
            "Opportunity Name": opportunity,
            "Remarks": remark,
            "sentiment": pred["sentiment"],
            "Sentiment": pred["sentiment"].title(),
            "confidence": pred["confidence"],
            "Confidence": pred["confidence"],
            "method": pred["method"],
            "Method": pred["method"],
        }
        processed_data = pd.concat([processed_data, pd.DataFrame([new_row])], ignore_index=True)

        stats = _calculate_statistics(processed_data)
        records = processed_data.to_dict(orient="records")

        ctx = _template_defaults()
        ctx.update({
            "request": request,
            "records": records,
            "data": records,
            "total_records": len(records),
            "message": "Remark added successfully!",
            **stats,
        })
        ctx["index"] = True
        return templates.TemplateResponse("index.html", ctx)

    except Exception as exc:
        return JSONResponse(status_code=500, content={"message": f"Error adding remark: {exc}"})


@app.get("/get_current_data")
async def get_current_data(
    page: int = Query(1),
    company: str = Query(""),
    search: str = Query(""),
    sentiment: str = Query(""),
):
    try:
        if processed_data is None or processed_data.empty:
            return JSONResponse(status_code=400, content={"error": "No data available"})

        df = processed_data.copy()
        if company:
            df = df[df.get("Company Name", pd.Series(dtype=str)).str.lower() == company.lower()]
        if search:
            df = df[df.get("Remarks", pd.Series(dtype=str)).str.contains(search, case=False, na=False)]
        if sentiment:
            df = df[df.get("Sentiment", pd.Series(dtype=str)).str.lower() == sentiment.lower()]

        records = df[[c for c in ["Company Name", "Opportunity Name", "Remarks", "Sentiment"] if c in df.columns]].to_dict(orient="records")
        return {"records": records, "total": len(records)}

    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Error retrieving data: {exc}"})


@app.post("/filter_data")
async def filter_data(
    request: Request,
    page: int = Form(1),
    company: str = Form(""),
    search: str = Form(""),
    sentiment: str = Form(""),
):
    try:
        if processed_data is None or processed_data.empty:
            return JSONResponse(status_code=400, content={"message": "No data available. Please upload a file first."})

        df = processed_data.copy()
        if company:
            df = df[df.get("Company Name", pd.Series(dtype=str)).str.lower() == company.lower()]
        if search:
            for col in ["Company Name", "Opportunity Name", "Remarks"]:
                if col in df.columns:
                    df = df[df[col].str.contains(search, case=False, na=False)]
                    break
        if sentiment:
            low = sentiment.lower()
            if low in ("positive", "negative", "neutral"):
                df = df[df.get("Sentiment", pd.Series(dtype=str)).str.lower() == low]

        records = df.to_dict(orient="records")
        stats = _calculate_statistics(df)

        ctx = _template_defaults()
        ctx.update({
            "request": request,
            "records": records,
            "data": records,
            "total_records": len(records),
            "selected_company": company,
            "search_query": search,
            "selected_sentiment": sentiment,
            **stats,
        })
        ctx["index"] = True
        return templates.TemplateResponse("index.html", ctx)

    except Exception as exc:
        return JSONResponse(status_code=500, content={"message": f"Error filtering data: {exc}"})


@app.get("/export_data")
async def export_data(
    company: str = Query(""),
    sentiment: str = Query(""),
):
    try:
        if processed_data is None or processed_data.empty:
            return JSONResponse(status_code=400, content={"message": "No data available"})

        df = processed_data.copy()
        if company and "Company Name" in df.columns:
            df = df[df["Company Name"].str.lower() == company.lower()]
        if sentiment and "Sentiment" in df.columns:
            df = df[df["Sentiment"].str.lower() == sentiment.lower()]

        export_cols = [c for c in ["Company Name", "Sentiment"] if c in df.columns]
        csv_data = df[export_cols].to_csv(index=False)
        filename = f"deberta_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as exc:
        return JSONResponse(status_code=500, content={"message": f"Error exporting data: {exc}"})


@app.get("/company_sentiment_analysis/{company_name}")
async def get_company_sentiment_analysis(company_name: str):
    try:
        if processed_data is None or processed_data.empty:
            return JSONResponse(status_code=400, content={"error": "No data available"})
        result = _calculate_company_sentiment_analysis(processed_data, company_name)
        if not result:
            return JSONResponse(status_code=404, content={"error": "Company not found"})
        return result
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Error: {exc}"})


@app.post("/edit_single_sentiment")
async def edit_single_sentiment(
    session_id: str = Form(...),
    original_sentiment: str = Form(...),
    new_sentiment: str = Form(...),
    confidence: float = Form(...),
    text: str = Form(...),
    reason: str = Form(""),
):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if session_id not in analysis_history:
            analysis_history[session_id] = []
        analysis_history[session_id].append({
            "timestamp": timestamp,
            "type": "user",
            "action": "sentiment_edit",
            "original": original_sentiment,
            "new": new_sentiment,
            "text": text,
            "reason": reason,
        })
        return {"message": "Sentiment updated successfully!", "new_sentiment": new_sentiment}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Error updating sentiment: {exc}"})


@app.post("/save_single_analysis")
async def save_single_analysis(
    session_id: str = Form(...),
    text: str = Form(...),
    sentiment: str = Form(...),
    confidence: float = Form(...),
    method: str = Form("deberta"),
    notes: str = Form(""),
    tags: str = Form(""),
):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "session_id": session_id,
            "timestamp": timestamp,
            "action": "analysis_save",
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "method": method,
            "notes": notes,
            "tags": tags,
        }
        single_analysis_log.append(entry)
        return {"message": "Analysis saved successfully!", "entry": entry}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Error saving analysis: {exc}"})


@app.post("/bulk_analyze_file")
async def bulk_analyze_file(file: UploadFile = File(...)):
    """
    Accepts CSV/XLSX with at least a 'Remarks' column.
    Returns JSON summary + per-row predictions.
    Does NOT require Company Name / Opportunity Name.
    """
    try:
        contents = await file.read()
        filename = file.filename or ""

        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type. Upload CSV or XLSX."})

        if "Remarks" not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"No 'Remarks' column found. Columns: {list(df.columns)}"})

        remarks = df["Remarks"].fillna("").tolist()
        batch_results = get_service().predict_batch(remarks)

        df["sentiment"] = [r["sentiment"].lower() for r in batch_results]
        df["confidence"] = [r["confidence"] for r in batch_results]

        positive = sum(1 for r in batch_results if r["sentiment"].lower() == "positive")
        negative = sum(1 for r in batch_results if r["sentiment"].lower() == "negative")
        neutral = sum(1 for r in batch_results if r["sentiment"].lower() == "neutral")

        return {
            "total": len(batch_results),
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "model": "deberta_sales_v1",
            "records": df.to_dict(orient="records"),
        }

    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Raw JSON API endpoints (for programmatic access)
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    """Single text → raw JSON response."""
    return get_service().predict_one(request.text)


@app.post("/predict-batch")
async def predict_batch(request: BatchSentimentRequest):
    """Batch texts → raw JSON response."""
    return get_service().predict_batch(request.texts)


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    return await bulk_analyze_file(file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main_deberta:app", host="0.0.0.0", port=8042, reload=False)
