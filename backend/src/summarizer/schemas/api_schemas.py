from pydantic import BaseModel

class SummarizeResponse(BaseModel):
    row: int = 1
    news: str
    link: str

class SummarizeReport(BaseModel):
    summarize: list[SummarizeResponse]