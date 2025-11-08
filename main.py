from fastapi import FastAPI
from api import routes_upload, routes_monitoring
from core.config import settings
from core.logger import logger

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

# Include routers
app.include_router(routes_upload.router)
app.include_router(routes_monitoring.router)


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 FastAPI App started successfully!")


@app.get("/")
def root():
    return {"message": f"Welcome to {settings.APP_NAME}"}
