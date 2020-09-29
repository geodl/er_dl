from typing import Optional
from pathlib import Path
import uuid
import io

from fastapi import UploadFile, APIRouter, File, Body, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import Field
from PIL import Image

from project.app.routers.models import ResistivityModelParams
from project.utils.common_utils import show_zondres2d_model
from project.config import common_config


router_sync = APIRouter()


@router_sync.post("/show_model",
                  description='Эндпоинт для визуализации табличной модели сопротивлений, полученной в ZondRes2D',
                  tags=['Визулизации'],
                  summary='Визуализация модели сопротивлений')
async def send(
        zondres2d_dat: UploadFile = File(..., description='Файл формата .dat с моделью'),
        min_rho: float = Body(0, ge=0, example=0, description='Минимальное сопротивление среды', embed=True),
        max_rho: float = Body(0, ge=0, example=0, description='Максимальное сопротивление среды', embed=True)):
    try:
        unique_id = str(uuid.uuid4())
        unique_dir = common_config.app_dir / 'static' / unique_id
        unique_dir.mkdir(parents=True, exist_ok=True)

        file = await zondres2d_dat.read()

        with open(unique_dir / zondres2d_dat.filename, "wb") as fout:
            fout.write(file)

        min_value = min_rho if min_rho > 0.0 else None
        max_value = max_rho if max_rho > 0.0 else None
        img = show_zondres2d_model(unique_dir / zondres2d_dat.filename, plot=False, return_as_pil=True, min_value=min_value, max_value=max_value)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
    # unique_dir.unlink()
        return StreamingResponse(buffer, media_type="image/png")
    except Exception:
        return Response('something go wrongs')


