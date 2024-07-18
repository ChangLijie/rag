import io
from typing import Optional

import torch

# prompt
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from haystack.components.rankers import LostInTheMiddleRanker
from haystack.dataclasses import ChatMessage
from PIL import Image

from core.context_relevance.main import Service as ContextRelevance
from core.gen_text.main import Service as LLM_Service
from core.img_text2text.main import Service as Emb_Img_Service
from core.memory.long_term import Instruction
from core.memory.short_term import ChatHistory
from core.prompt.main import Service as Prompt_Engineering
from core.rag.main import Service as rag_service
from core.vec_db.faiss.data import Process as Faiss_Data
from core.vec_db.faiss.main import Operator as Faiss_Operater
from core.vec_db.pgvector.data import Process as Pgvec_Data
from core.vec_db.pgvector.main import Operator as Pgvec_Operater
from tools.download_model import download_model
from tools.setting import HUGGING_FACE_TOKEN

app = FastAPI()
# download model from hugging face
model_list = download_model()

# Init datasets
faiss = Faiss_Operater(dataset_path="./embeddings/data")
faiss.load(faiss_path="./embeddings/img_embedding.faiss")
pgvec = Pgvec_Operater()

# Init model
clip = Emb_Img_Service(model_path=model_list["img_text2text"]["path"])
rag = rag_service(model_path=model_list["rag"]["path"], type="text")
llm = LLM_Service(model_path=model_list["gen_text"]["path"])

# retriever
ranker = LostInTheMiddleRanker(top_k=1)

# Init prompt
prompt_engineer = Prompt_Engineering()


# Init memory
s_memory = ChatHistory()
l_memory = Instruction()

# context checker

context_relevance = ContextRelevance(model=rag)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.post("/chat/")
async def chat(
    file: Optional[UploadFile] = File(None), prompt: Optional[str] = Form(None)
):
    response = {}
    if file and prompt:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_embedding = clip.get_image_vector(image=image)
        scores, answer = faiss.search(q_vector=img_embedding)
        s_memory.remember(user_prompt="What is it?", bot_answer=answer["describe"][0])

        question_vector = rag.run(data=answer["describe"][0])
        retriever_result = pgvec.search(query_embedding=question_vector)

        if retriever_result["documents"]:
            rank_documents = ranker.run(documents=retriever_result["documents"])
            rank_documents = rank_documents["documents"][0].content

        summary_his_prompt = prompt_engineer._summary_history(
            chat_history=s_memory.get()
        )
        summary_history = llm.run(prompt=summary_his_prompt)
        summary_history_context_relevance = context_relevance.compare(
            context1=prompt, context2=summary_history
        )
        rank_documents_relevance = context_relevance.compare(
            context1=prompt, context2=rank_documents
        )

        if summary_history_context_relevance:
            history = summary_history

        else:
            history = summary_history_context_relevance

        if rank_documents_relevance:
            retrieval = rank_documents

        else:
            retrieval = rank_documents_relevance

        new_prompt = prompt_engineer.generate(
            history=history,
            retrieval=retrieval,
            prompt=prompt,
            instruction=l_memory.get(),
        )

        llm_answer = llm.run(prompt=new_prompt)
        s_memory.remember(user_prompt=prompt, bot_answer=llm_answer)

        response["message"] = llm_answer
        return JSONResponse(content=response)

    elif file:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_embedding = clip.get_image_vector(image=image)
        scores, answer = faiss.search(q_vector=img_embedding)
        s_memory.remember(user_prompt="What is it?", bot_answer=answer["describe"][0])
        response["message"] = answer["describe"][0]
        return JSONResponse(content=response)

    else:
        question_vector = rag.run(data=prompt)
        retriever_result = pgvec.search(query_embedding=question_vector)

        if retriever_result["documents"]:
            rank_documents = ranker.run(documents=retriever_result["documents"])
            rank_documents = rank_documents["documents"][0].content

        summary_his_prompt = prompt_engineer._summary_history(
            chat_history=s_memory.get()
        )

        summary_history = llm.run(prompt=summary_his_prompt)

        summary_history_context_relevance = context_relevance.compare(
            context1=prompt, context2=summary_history
        )
        rank_documents_relevance = context_relevance.compare(
            context1=prompt, context2=rank_documents
        )

        if summary_history_context_relevance:
            history = summary_history

        else:
            history = summary_history_context_relevance

        if rank_documents_relevance:
            retrieval = rank_documents

        else:
            retrieval = rank_documents_relevance

        new_prompt = prompt_engineer.generate(
            history=history,
            retrieval=retrieval,
            prompt=prompt,
            instruction=l_memory.get(),
        )

        llm_answer = llm.run(prompt=new_prompt)

        s_memory.remember(user_prompt=prompt, bot_answer=llm_answer)
        response["message"] = llm_answer
        return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
