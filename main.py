# uvicorn main:app --reload
# $Env:UVICORN_PORT=5000
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}