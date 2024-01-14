from redis_om import (Field, EmbeddedJsonModel)

class Parameter(EmbeddedJsonModel):
  name: str = Field(index = True)
  example: str = Field(index = True)