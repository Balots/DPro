from DataProcessing import CleanData, HandleMissingValues, NormalizeData, StandardizeData, DetectAndRemoveOutliers
from pydantic import BaseModel
from Logger import *
import json

class DataObject(BaseModel):
    method_id: int
    params: list

class IDProcessing:
    __METHODS__ = [CleanData, HandleMissingValues, NormalizeData, StandardizeData, DetectAndRemoveOutliers]
    def __init__(self, request: DataObject):
        self.method_id = request.method_id
        self.params = request.params

    @decorator
    def get(self):
        process = self.__METHODS__[self.method_id](*self.params)
        process.run()
        result = process.get_answ().to_dict()
        return json.dumps(result)


