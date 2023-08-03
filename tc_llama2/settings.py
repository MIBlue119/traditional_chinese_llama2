
from typing import List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Base(BaseSettings):
    FINETUNED_DATA_PATH_LIST:  List[str] = Field([], env="FINETUNED_DATA_PATH_LIST")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


app_settings = Base()