from fastapi import FastAPI
from pydantic import BaseModel, Field
app = FastAPI()
class Task(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    priority: int = Field(ge=1, le=5)
    tags: list[str] = []
@app.get("/sync")
def sync_endpoint():
    return {"status": "ok", "worker": "sync"}
@app.get("/async")
async def async_endpoint():
    return {"status": "ok", "worker": "async"}
@app.post("/tasks")
async def create_task(task: Task):
    return {"id": 1, **task.model_dump()}
