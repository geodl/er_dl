from typing import Optional

from pydantic import Field, BaseModel


class ResistivityModelParams(BaseModel):
    min_rho: Optional[int] = Field(None, ge=0)
    max_rho: Optional[int] = Field(None, ge=0)
