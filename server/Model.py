from redis_om import (EmbeddedJsonModel, Field, JsonModel)
from typing import Optional

class Model(JsonModel):
  uniqueName: str = Field(index = True)
  name: str = Field(index = True)
  description: str = Field(index = False, default="No description")
  modelData: bytes = Field(index = False, default=b"0x1")