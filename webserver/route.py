from fastapi import HTTPException, UploadFile, File
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from IDProcessing import DataObject, IDProcessing
#from Detector import IDet
import pandas as pd
from pathlib import Path

Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("temp_reports").mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

current_df = {}

@app.get("/DataFrame/", tags=['Обработка данных'])
async def processing_data(id: int):
    global current_df
    data = DataObject(method_id=id, params=[current_df])
    if type(current_df) is pd.DataFrame:
        tmp = IDProcessing(data)
        current_df = tmp.get()
        return JSONResponse(content={"message": "DataFrame processing successfully", "shape": current_df.shape})
    raise HTTPException


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "notebook.html",
        {"request": request, "notebook_name": "DPro Data Analysis"}
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

'''@app.get("/generate_report/")
async def generate_report():
    global current_df, current_report_html
    try:
        res = IDet(True, True, True, True).logging_results(current_df)
        print(res)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)'''


