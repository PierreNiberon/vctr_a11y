# uvicorn main:app --reload
# $Env:UVICORN_PORT=5000
from fastapi import FastAPI
from lib import redis_ping, load_redis, wiki_page_content, get_answer

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health_check_redis")
async def health_check():
    redis_check = redis_ping()
    health_check_redis_bool = f"Redis is accessible : {redis_check}"
    return {"message": health_check_redis_bool}

@app.post("/load_redis_wiki/{page_name}")
async def load_redis_wiki(page_name: str):
    content = wiki_page_content(page_name)
    redis_response = load_redis(content)
    return {"redis_response": redis_response}

@app.post("/ask_question/{question}")
async def ask_question(question: str):
    response = get_answer('idx', question)
    return {"Question": question,
            "Answer": response}