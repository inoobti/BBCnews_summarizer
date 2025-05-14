from fastapi import APIRouter
from summarizer.parser.parser import BBCNewsParser
from summarizer.script.summ import SummarizerProcessor
from summarizer.schemas.api_schemas import SummarizeResponse, SummarizeReport

csv_router = APIRouter()

news = BBCNewsParser()
summ = SummarizerProcessor()

## Парсинг и вывод всех новостей разом
# @csv_router.post("/summarize/", response_model=SummarizeReport)
# async def get_news_list():
#     data = news.return_dataset()
#     amount_of_news = len(data)
#     results = []
#
#     for index in range(amount_of_news):
#
#             text = data.loc[index]['text']
#             link = data.loc[index]['link']
#             summ_text = summ.inference(text)
#             result = SummarizeResponse(
#                                     row=index + 1,
#                                     news=summ_text,
#                                     link=link
#                                         )
#             results.append(result)
#
#     return SummarizeReport(summarize=results)


@csv_router.post("/summarize/", response_model=SummarizeResponse)
async def get_news_list():
    data, index = news.get_one_news()
    text = data['text'].values[0]
    link = data['link'].values[0]
    summ_text = summ.inference(text)
    result = SummarizeResponse(
        row=index + 1,
        news=summ_text,
        link=link
    )
    return result


@csv_router.post("/get_users_news/")
def get_news_from_user(item: dict):
    return summ.inference(item['text'])

