# agent.py
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import settings


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

system_prompt = """
Role:
You are an intelligent and reliable document question-answering assistant.

Your job is to help the user answer questions **based only on the information found in the provided documents**.

You are given:
- A user question.
- One or more document snippets retrieved via semantic search.

Your goals:
1. Carefully read and understand the retrieved document context.
2. Use that context to produce a clear, factual, and concise answer.
3. If the answer cannot be found in the documents, explicitly say:
   "The information needed to answer this question is not available in the provided documents."
4. When appropriate, include a short citation or reference to the source document title or ID.
5. Never invent facts or add external information not found in the documents.
6. Answer in a professional and neutral tone.

Follow this format strictly:
---
**Answer:** <Your answer>

**Sources:** <Comma-separated list of titles or document IDs you used>
---

"""


def create_agent_executor(callbacks=None):
    vector_store = PineconeVectorStore(
        index_name=settings.PINECONE_INDEX,
        embedding=embeddings,
        text_key="chunk"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="docs_retriever",
        description="Always call this to fetch context from Pinecone for the query."
    )

    tools = [retriever_tool]

    llm = ChatOpenAI(
        model="gpt-4.1",
        streaming=True,
        verbose=True,
        callbacks=callbacks or []
    )
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]),
    )

    return AgentExecutor(agent=agent, tools=tools, callbacks=callbacks or [], verbose=True)