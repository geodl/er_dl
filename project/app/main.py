from fastapi import FastAPI, UploadFile
from fastapi.responses import Response

from project.app.routers.sync_endpoints import router_sync

description = """
## ERT WebKits

### Система способна:
1. Визуализировать модель сопротивлений
2. ...
"""

app = FastAPI(description=description)
app.include_router(router_sync)

