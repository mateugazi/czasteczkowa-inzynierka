from redis_om import (Field, JsonModel)
from typing import List

class ModelType(JsonModel):
  name: str = Field(index = True)
  parameters: List[str] = Field(index = False, default = [])