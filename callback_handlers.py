# callbacks.py
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult, BaseMessage
from typing import Optional, Any, Dict, List
from uuid import UUID
from fastapi import WebSocket


class FastAPIStreamingCallbackHandler(AsyncCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        text = ""
        if isinstance(token, dict) and "text" in token:
            text = token["text"]
        elif isinstance(token, list):
            # Sometimes token is a list of dicts
            for t in token:
                if isinstance(t, dict) and "text" in t:
                    text += t["text"]
        elif isinstance(token, str):
            text = token
        else:
            text = str(token)
        if text.strip():
            await self.websocket.send_json({"type": "text", "text": text})  

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        # await self.websocket.send_json({"type": "end"})
        pass
    
    def on_agent_action(self, action, **kwargs):
        print(f"\n🤖 Agent decided: {action.tool} with input: {action.tool_input}\n")
        
    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> Any:
        
        pass