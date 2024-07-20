from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import video_app
import uvicorn
import asyncio
import sys

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify allowed origins here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/progress")
async def progress(request: Request):
    async def event_generator():
        while True:
            # Fetch progress updates
            progress = get_progress()
            print(progress["percentage"])
            sys.stdout.flush()  # Force output to be written to the terminal
            yield f"data: {json.dumps(progress)}\n\n"
            if progress["percentage"] >= 100:
                yield f"data: {json.dumps({'status': 'complete'})}\n\n"
                break
            await asyncio.sleep(1)
            if await request.is_disconnected():
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def get_progress():
    progress = video_app.progress
    return progress

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
