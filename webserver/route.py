from fastapi import HTTPException, UploadFile, File
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from IDProcessing import DataObject, IDProcessing
import pandas as pd
from pathlib import Path
from ydata_profiling import ProfileReport

Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("temp_reports").mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

current_df = {}
report_html = None

@app.get("/DataFrame/", tags=['Обработка данных'])
async def processing_data(id: int):
    global current_df
    data = DataObject(method_id=id, params=[current_df])
    if type(current_df) is pd.DataFrame:
        tmp = IDProcessing(data)
        current_df = tmp.get()
        print(type(current_df))
        return JSONResponse(content={"message": "DataFrame processing successfully", "shape": current_df.shape})
    raise HTTPException


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "notebook.html",
        {"request": request, "notebook_name": "DPro Data Analysis", "report_html": report_html}
    )


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    global current_df
    try:
        if file.filename.endswith('.csv'):
            current_df = pd.read_csv(file.file, na_values=['', ' ', 'NA', 'N/A', 'null'])
        elif file.filename.endswith('.xlsx'):
            current_df = pd.read_excel(file.file, na_values=['', ' ', 'NA', 'N/A', 'null'])
        else:
            return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

        return JSONResponse(content={"message": "File uploaded successfully", "shape": current_df.shape})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/generate_report/")
async def generate_report():
    global current_df, report_html
    try:
        profile = ProfileReport(current_df, title=f"Your New Magic Data {chr(129392)} \u2728... or not, if you just loaded it {chr(128577)}")
        report_html = profile.to_html()
        return HTMLResponse(content=report_html, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
