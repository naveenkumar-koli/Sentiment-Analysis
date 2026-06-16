from fastapi import FastAPI, Request, Form, File, UploadFile, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import pickle
from typing import List, Optional
from pydantic import BaseModel
import os
from pathlib import Path
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from fastapi.middleware.cors import CORSMiddleware
import math
from datetime import datetime

from fastapi.encoders import jsonable_encoder

import json
from fastapi.responses import FileResponse
import io
import uuid
from .utils import preprocess_text

# Initialize NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Preprocessing handled by preprocess_text() from utils.py

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Add these global variables after your existing global variables
single_analysis_log = []
analysis_history = {}

# Mount static files
static_path = Path(__file__).parent / "static"
if not static_path.exists():
    static_path.mkdir()
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Mount templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    method: str


# Global variable to store processed data
processed_data = None


def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "New_Updated_Models", "sentiment_model_4770.pkl")
    vectorizer_path = os.path.join(base_dir, "models", "New_Updated_Models", "tfidf_vectorizer_4770.pkl")
    encoder_path = os.path.join(base_dir, "models", "New_Updated_Models", "label_encoder_4770.pkl")

    if not all(os.path.exists(p) for p in [model_path, vectorizer_path, encoder_path]):
        missing = [p for p in [model_path, vectorizer_path, encoder_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Model files missing: {missing}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    return model, vectorizer, encoder


try:
    model, vectorizer, encoder = load_models()
except Exception as e:
    print(f"Error loading models: {str(e)}")
    model = None
    vectorizer = None
    encoder = None


def predict_sentiment_enhanced(text: str) -> SentimentResponse:
    """Sentiment prediction using trained model only"""
    if not all([model, vectorizer, encoder]):
        raise ValueError("Models not loaded properly")

    # Preprocess and predict using model only
    cleaned_text = preprocess_text(text)
    X = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    sentiment = encoder.inverse_transform([pred])[0].lower()
    confidence = float(np.max(proba))

    return SentimentResponse(
        sentiment=sentiment,
        confidence=confidence,
        method="model"
    )


def get_sentiment_emoji(sentiment):
    return ""




def get_confidence_color(confidence):
    if confidence >= 0.8:
        return 'success'
    elif confidence >= 0.6:
        return 'warning'
    else:
        return 'danger'


def calculate_statistics(df):
    """Calculate all statistics for the dashboard"""
    companies = sorted(df['Company Name'].unique().tolist())
    company_stats = df.groupby('Company Name')['Sentiment'].value_counts().unstack(fill_value=0)

    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in company_stats.columns:
            company_stats[sentiment] = 0

    company_stats['Total'] = company_stats.sum(axis=1)

    overall_stats = {
        'total_pos': int(df[df['Sentiment'] == 'positive'].shape[0]),
        'total_neg': int(df[df['Sentiment'] == 'negative'].shape[0]),
        'total_neu': int(df[df['Sentiment'] == 'neutral'].shape[0]),
        'total_records': len(df)
    }

    # Prepare pie chart data for overall
    sentiment_pie_data = {
        'labels': ['Positive', 'Negative', 'Neutral'],
        'counts': [overall_stats['total_pos'], overall_stats['total_neg'], overall_stats['total_neu']]
    }

    # Prepare pie chart data per company
    company_pie_data = {}
    for company in companies:
        comp_sent = df[df['Company Name'] == company]['Sentiment'].value_counts()
        company_pie_data[company] = {
            'positive': int(comp_sent.get('positive', 0)),
            'negative': int(comp_sent.get('negative', 0)),
            'neutral': int(comp_sent.get('neutral', 0))
        }

    return companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "single_result": None,
        "batch_result": False,
        "input_text": "",
        "session_id": None,  # Add this line
        "data": [],
        "total_records": 0,
        "page": 1,
        "total_pages": 0,
        "companies": [],
        "selected_company": "",
        "search_query": "",
        "selected_sentiment": "",
        "company_stats": {},
        "overall_stats": {},
        "sentiment_pie_data": {},
        "company_pie_data": {}
    })



@app.post("/analyze")
async def analyze_text(text: str = Form(...)):
    """
    Analyze the sentiment of a given text and return a JSON response.
    - **text**: The input text to analyze
    """
    try:
        result = predict_sentiment_enhanced(text)
        session_id = str(uuid.uuid4())
        return JSONResponse(content={
            "session_id": session_id,
            "input_text": text,
            "sentiment": result.sentiment,
            "confidence": round(result.confidence, 4),
            "method": result.method
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error analyzing text: {str(e)}"}
        )

from fastapi import File, UploadFile, Request
from fastapi.responses import JSONResponse
import pandas as pd
import math

@app.post("/batch_analyze")
async def batch_analyze(request: Request, file: UploadFile = File(...)):
    global processed_data
    try:
        if not all([model, vectorizer, encoder]):
            raise ValueError("Models not loaded properly")

        filename = (file.filename or "").lower()

        # Read file
        if filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        elif filename.endswith(".csv"):
            # if you face encoding issues, switch to encoding="utf-8-sig"
            df = pd.read_csv(file.file)
        else:
            return JSONResponse(
                status_code=400,
                content={"message": "Unsupported file type. Please upload .xlsx or .csv"}
            )

        required_cols = ["Company Name", "Opportunity Name", "Remarks"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return JSONResponse(
                status_code=400,
                content={"message": f"Missing required columns: {', '.join(missing)}"}
            )

        # ✅ Sanitize text columns to avoid float/NaN issues
        for col in required_cols:
            df[col] = df[col].fillna("").astype(str)

        # Enhanced sentiment analysis for each row
        sentiments = []
        confidences = []
        methods = []

        for remark in df["Remarks"].tolist():
            # remark is now guaranteed to be a string
            result = predict_sentiment_enhanced(remark)
            sentiments.append(result.sentiment)
            confidences.append(result.confidence)
            methods.append(result.method)

        df["Sentiment"] = sentiments
        df["Confidence"] = confidences
        df["Method"] = methods
        df["Confidence_Color"] = df["Confidence"].apply(get_confidence_color)

        # Store processed data globally
        processed_data = df

        # Calculate all statistics
        companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data = calculate_statistics(df)

        # Pagination for first page
        page = 1
        per_page = 10
        total_pages = max(1, math.ceil(len(df) / per_page))
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_data = df.iloc[start_idx:end_idx].to_dict("records")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "batch_result": True,
            "data": paginated_data,
            "total_records": len(df),
            "page": page,
            "total_pages": total_pages,
            "companies": companies,
            "selected_company": "",
            "search_query": "",
            "selected_sentiment": "",
            "company_stats": company_stats.to_dict("index"),
            "overall_stats": overall_stats,
            "single_result": None,
            "input_text": "",
            "sentiment_pie_data": sentiment_pie_data,
            "company_pie_data": company_pie_data
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing file: {str(e)}"}
        )

@app.post("/add_remark")
async def add_remark(company: str = Form(...), opportunity: str = Form(...), remark: str = Form(...)):
    global processed_data

    if processed_data is None:
        return JSONResponse(status_code=400,
                            content={"success": False, "message": "No data available. Please upload a file first."})

    try:
        # Create new row as dataframe
        new_row = pd.DataFrame({
            'Company Name': [company],
            'Opportunity Name': [opportunity],
            'Remarks': [remark]
        })

        # Enhanced sentiment analysis
        result = predict_sentiment_enhanced(remark)
        new_row['Sentiment'] = result.sentiment
        new_row['Confidence'] = result.confidence
        new_row['Method'] = result.method
        new_row['Confidence_Color'] = get_confidence_color(result.confidence)

        # Append new row to processed data
        processed_data = pd.concat([processed_data, new_row], ignore_index=True)

        # Calculate updated statistics
        companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data = calculate_statistics(
            processed_data)

        # Get current page data (first page by default after adding)
        page = 1
        per_page = 10
        total_pages = math.ceil(len(processed_data) / per_page)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        current_page_data = processed_data.iloc[start_idx:end_idx].to_dict('records')

        # Return comprehensive response with ALL updated data
        return JSONResponse(status_code=200, content={
            "success": True,
            "message": "Remark added successfully!",
            "new_remark": {
                'Company Name': company,
                'Opportunity Name': opportunity,
                'Remarks': remark,
                'Sentiment': result.sentiment,
                'Confidence': result.confidence,
                'Method': result.method,
                'Confidence_Color': get_confidence_color(result.confidence)
            },
            "updated_stats": {
                "companies": companies,
                "company_stats": company_stats.to_dict('index'),
                "overall_stats": overall_stats,
                "sentiment_pie_data": sentiment_pie_data,
                "company_pie_data": company_pie_data,
                "total_records": len(processed_data)
            },
            "updated_table": {
                "data": current_page_data,
                "total_records": len(processed_data),
                "page": page,
                "total_pages": total_pages
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Error adding remark: {str(e)}"})


@app.get("/get_current_data")
async def get_current_data(
        page: int = Query(1),
        company: Optional[str] = Query(""),
        search: Optional[str] = Query(""),
        sentiment: Optional[str] = Query("")
):
    global processed_data

    if processed_data is None:
        return JSONResponse(status_code=400, content={"success": False, "message": "No data available"})

    try:
        df = processed_data.copy()
        original_df = processed_data.copy()

        # Apply filters
        if company and company != "all":
            df = df[df['Company Name'] == company]

        if search:
            search_mask = (
                    df['Company Name'].str.contains(search, case=False, na=False) |
                    df['Opportunity Name'].str.contains(search, case=False, na=False) |
                    df['Remarks'].str.contains(search, case=False, na=False)
            )
            df = df[search_mask]

        if sentiment and sentiment != "all":
            df = df[df['Sentiment'] == sentiment]

        # Pagination
        per_page = 10
        total_pages = math.ceil(len(df) / per_page) if len(df) > 0 else 1
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_data = df.iloc[start_idx:end_idx].to_dict('records')

        # Calculate statistics for all data
        companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data = calculate_statistics(
            original_df)

        return JSONResponse(status_code=200, content={
            "success": True,
            "data": paginated_data,
            "total_records": len(df),
            "page": page,
            "total_pages": total_pages,
            "companies": companies,
            "company_stats": company_stats.to_dict('index'),
            "overall_stats": overall_stats,
            "sentiment_pie_data": sentiment_pie_data,
            "company_pie_data": company_pie_data
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Error retrieving data: {str(e)}"})


@app.get("/filter_data")
async def filter_data(
        request: Request,
        page: int = Query(1),
        company: Optional[str] = Query(""),
        search: Optional[str] = Query(""),
        sentiment: Optional[str] = Query("")
):
    global processed_data

    if processed_data is None:
        return JSONResponse(
            status_code=400,
            content={"message": "No data available. Please upload a file first."}
        )

    try:
        df = processed_data.copy()
        original_df = processed_data.copy()

        # Apply filters
        if company and company != "all":
            df = df[df['Company Name'] == company]

        if search:
            search_mask = (
                    df['Company Name'].str.contains(search, case=False, na=False) |
                    df['Opportunity Name'].str.contains(search, case=False, na=False) |
                    df['Remarks'].str.contains(search, case=False, na=False)
            )
            df = df[search_mask]

        if sentiment and sentiment != "all":
            df = df[df['Sentiment'] == sentiment]

        # Pagination
        per_page = 10
        total_pages = math.ceil(len(df) / per_page) if len(df) > 0 else 1
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_data = df.iloc[start_idx:end_idx].to_dict('records')

        # Calculate stats for ALL data (not filtered)
        companies, company_stats, overall_stats, overall_sentiment_pie_data, overall_company_pie_data = calculate_statistics(
            original_df)

        # Calculate filtered pie chart data
        filtered_sentiment_pie_data = {
            'labels': ['Positive', 'Negative', 'Neutral'],
            'counts': [
                int(df[df['Sentiment'] == 'positive'].shape[0]),
                int(df[df['Sentiment'] == 'negative'].shape[0]),
                int(df[df['Sentiment'] == 'neutral'].shape[0])
            ]
        }

        return templates.TemplateResponse("index.html", {
            "request": request,
            "batch_result": True,
            "data": paginated_data,
            "total_records": len(df),
            "page": page,
            "total_pages": total_pages,
            "companies": companies,
            "selected_company": company,
            "search_query": search,
            "selected_sentiment": sentiment,
            "company_stats": company_stats.to_dict('index'),
            "overall_stats": overall_stats,
            "single_result": None,
            "input_text": "",
            "sentiment_pie_data": filtered_sentiment_pie_data,
            "company_pie_data": overall_company_pie_data
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error filtering data: {str(e)}"}
        )


@app.get("/export_data")
async def export_data(company: Optional[str] = Query(""), sentiment: Optional[str] = Query("")):
    global processed_data

    if processed_data is None:
        return JSONResponse(
            status_code=400,
            content={"message": "No data available"}
        )

    try:
        df = processed_data.copy()

        # Apply filters
        if company and company != "all":
            df = df[df['Company Name'] == company]

        if sentiment and sentiment != "all":
            df = df[df['Sentiment'] == sentiment]

        # Export to CSV with method information
        export_df = df[['Company Name', 'Opportunity Name', 'Remarks', 'Sentiment', 'Confidence', 'Method']].copy()
        csv_data = export_df.to_csv(index=False)

        return JSONResponse({
            "data": csv_data,
            "filename": f"sentiment_analysis_{company}_{sentiment}.csv"
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error exporting data: {str(e)}"}
        )


def calculate_company_sentiment_analysis(df, company_name):
    """Calculate detailed sentiment analysis for a specific company"""
    company_data = df[df['Company Name'] == company_name].copy()

    if company_data.empty:
        return None

    # Calculate confidence-weighted sentiment scores
    sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
    total_weighted_confidence = 0

    for _, row in company_data.iterrows():
        sentiment = row['Sentiment']
        confidence = row['Confidence']
        sentiment_scores[sentiment] += confidence
        total_weighted_confidence += confidence

    # Normalize scores to percentages
    if total_weighted_confidence > 0:
        for sentiment in sentiment_scores:
            sentiment_scores[sentiment] = (sentiment_scores[sentiment] / total_weighted_confidence) * 100

    # Determine overall sentiment based on highest weighted score
    overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    overall_confidence = sentiment_scores[overall_sentiment]

    # Prepare data for line chart (sentiment distribution over confidence ranges)
    confidence_ranges = ['0-50%', '51-70%', '71-90%', '91-100%']
    sentiment_distribution = {
        'positive': [0, 0, 0, 0],
        'negative': [0, 0, 0, 0],
        'neutral': [0, 0, 0, 0]
    }

    for _, row in company_data.iterrows():
        confidence = row['Confidence'] * 100
        sentiment = row['Sentiment']

        if confidence <= 50:
            sentiment_distribution[sentiment][0] += 1
        elif confidence <= 70:
            sentiment_distribution[sentiment][1] += 1
        elif confidence <= 90:
            sentiment_distribution[sentiment][2] += 1
        else:
            sentiment_distribution[sentiment][3] += 1

    return {
        'company_name': company_name,
        'overall_sentiment': overall_sentiment,
        'overall_confidence': round(overall_confidence, 1),
        'sentiment_scores': sentiment_scores,
        'total_remarks': len(company_data),
        'confidence_ranges': confidence_ranges,
        'sentiment_distribution': sentiment_distribution,
        'raw_data': company_data[['Opportunity Name', 'Remarks', 'Sentiment', 'Confidence']].to_dict('records')
    }


@app.get("/company_sentiment_analysis/{company_name}")
async def get_company_sentiment_analysis(company_name: str):
    global processed_data

    if processed_data is None:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "No data available"}
        )

    try:
        analysis = calculate_company_sentiment_analysis(processed_data, company_name)

        if analysis is None:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "Company not found"}
            )

        return JSONResponse(
            status_code=200,
            content={"success": True, "data": analysis}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error analyzing company sentiment: {str(e)}"}
        )




# Add this new model class after your existing SentimentResponse class
class SentimentEditRequest(BaseModel):
    original_sentiment: str
    new_sentiment: str
    confidence: float
    text: str
    reason: str = ""


# Add these new endpoints AFTER your existing endpoints

@app.post("/edit_single_sentiment")
async def edit_single_sentiment(
        session_id: str = Form(...),
        original_sentiment: str = Form(...),
        new_sentiment: str = Form(...),
        confidence: float = Form(...),
        text: str = Form(...),
        reason: str = Form("")
):
    """Edit sentiment for single analysis and log the change"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "text": text,
            "original_sentiment": original_sentiment,
            "new_sentiment": new_sentiment,
            "confidence": confidence,
            "reason": reason,
            "edited_by": "user",
            "action": "sentiment_edit"
        }

        single_analysis_log.append(log_entry)

        if session_id not in analysis_history:
            analysis_history[session_id] = []
        analysis_history[session_id].append(log_entry)

        return JSONResponse({
            "success": True,
            "message": "Sentiment updated successfully!",
            "updated_sentiment": new_sentiment,
            "confidence": confidence,
            "timestamp": timestamp
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error updating sentiment: {str(e)}"}
        )


@app.post("/save_single_analysis")
async def save_single_analysis(
        session_id: str = Form(...),
        text: str = Form(...),
        sentiment: str = Form(...),
        confidence: float = Form(...),
        method: str = Form(...),
        notes: str = Form(""),
        tags: str = Form("")
):
    """Save single analysis result with notes and tags"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        save_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "method": method,
            "notes": notes,
            "tags": tags.split(",") if tags else [],
            "action": "analysis_save"
        }

        single_analysis_log.append(save_entry)

        if session_id not in analysis_history:
            analysis_history[session_id] = []
        analysis_history[session_id].append(save_entry)

        return JSONResponse({
            "success": True,
            "message": "Analysis saved successfully!",
            "saved_data": save_entry
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error saving analysis: {str(e)}"}
        )


@app.get("/download_single_analysis_log")
async def download_single_analysis_log(
        format: str = Query("excel", regex="^(excel|csv)$"),
        session_id: Optional[str] = Query(None)
):
    """Download analysis log as Excel or CSV"""
    try:
        if session_id:
            data = analysis_history.get(session_id, [])
            filename_suffix = f"_session_{session_id}"
        else:
            data = single_analysis_log
            filename_suffix = "_all_sessions"

        if not data:
            return JSONResponse(
                status_code=404,
                content={"message": "No analysis data found"}
            )

        df = pd.DataFrame(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "excel":
            filename = f"single_analysis_log{filename_suffix}_{timestamp}.xlsx"
            filepath = f"temp_{filename}"

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Analysis_Log', index=False)

                if len(df) > 0:
                    summary_data = {
                        'Total Analyses': [len(df)],
                        'Positive Sentiments': [int((df['sentiment'] == 'positive').sum()) if 'sentiment' in df.columns else 0],
                        'Negative Sentiments': [int((df['sentiment'] == 'negative').sum()) if 'sentiment' in df.columns else 0],
                        'Neutral Sentiments': [int((df['sentiment'] == 'neutral').sum()) if 'sentiment' in df.columns else 0],
                        'Total Edits': [int((df['action'] == 'sentiment_edit').sum()) if 'action' in df.columns else 0],
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

            return FileResponse(
                filepath,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=filename
            )

        else:  # CSV format
            filename = f"single_analysis_log{filename_suffix}_{timestamp}.csv"
            csv_data = df.to_csv(index=False)

            return JSONResponse({
                "data": csv_data,
                "filename": filename,
                "content_type": "text/csv"
            })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error generating download: {str(e)}"}
        )


@app.get("/get_analysis_history/{session_id}")
async def get_analysis_history(session_id: str):
    """Get analysis history for a specific session"""
    try:
        history = analysis_history.get(session_id, [])

        return JSONResponse({
            "success": True,
            "history": history,
            "total_entries": len(history)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error retrieving history: {str(e)}"}
        )


@app.delete("/clear_analysis_log")
async def clear_analysis_log(session_id: Optional[str] = Query(None)):
    """Clear analysis log (all or specific session)"""
    try:
        global single_analysis_log, analysis_history

        if session_id:
            if session_id in analysis_history:
                del analysis_history[session_id]
                single_analysis_log = [entry for entry in single_analysis_log
                                       if entry.get("session_id") != session_id]
                message = f"Session {session_id} cleared successfully"
            else:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "message": "Session not found"}
                )
        else:
            single_analysis_log = []
            analysis_history = {}
            message = "All analysis logs cleared successfully"

        return JSONResponse({
            "success": True,
            "message": message
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error clearing logs: {str(e)}"}
        )


@app.post("/bulk_analyze_file")
async def bulk_analyze_file(file: UploadFile = File(...)):
    """Bulk sentiment analysis from CSV or XLSX file.
    Auto-detects Remarks/remark/Remark/remarks column."""
    try:
        if not all([model, vectorizer, encoder]):
            return JSONResponse(status_code=500, content={"success": False, "message": "Models not loaded"})

        contents = await file.read()
        filename = (file.filename or "").lower()

        # Read file
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": "Unsupported file type. Please upload a CSV or XLSX file."
            })

        # Find remarks column — case-insensitive match
        remarks_col = None
        for col in df.columns:
            if col.strip().lower() in ["remarks", "remark"]:
                remarks_col = col
                break

        if not remarks_col:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": f"No 'Remarks' column found. Columns in file: {list(df.columns)}"
            })

        # Run predictions
        results = []
        pos_count = neg_count = neu_count = skipped = 0

        for idx, row in df.iterrows():
            raw_text = row[remarks_col]
            text = str(raw_text).strip() if pd.notna(raw_text) else ""

            if not text or text.lower() == "nan":
                skipped += 1
                continue

            cleaned = preprocess_text(text)
            X = vectorizer.transform([cleaned])
            proba = model.predict_proba(X)[0]
            pred = model.predict(X)[0]
            sentiment = encoder.inverse_transform([pred])[0].lower()
            confidence = float(np.max(proba))

            results.append({
                "row": int(idx) + 1,
                "remarks": text[:250],
                "sentiment": sentiment,
                "confidence": round(confidence, 4)
            })

            if sentiment == "positive":
                pos_count += 1
            elif sentiment == "negative":
                neg_count += 1
            else:
                neu_count += 1

        return JSONResponse({
            "success": True,
            "total": len(results),
            "positive": pos_count,
            "negative": neg_count,
            "neutral": neu_count,
            "skipped": skipped,
            "results": results
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8024)

