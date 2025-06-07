from fastapi import FastAPI, HTTPException
from IDProcessing import DataObject, IDProcessing
from pandas import DataFrame

app = FastAPI()
df = {}


@app.post("/DataFrame", tags=["Принимает датасет"])
def create_data(new_df: dict):
    global df
    df = DataFrame(new_df)
    return {"success": True, 'message': 'Датасет успешно загружен!'}


@app.get("/DataFrame/{id}", tags=['Обработка данных'])
def processing_data(id: int):
    data = DataObject(method_id=id, params=[df])
    print(data)
    if type(df) is DataFrame:
        tmp = IDProcessing(data)
        return tmp.get()
    raise HTTPException
