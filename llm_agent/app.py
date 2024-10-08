"""
This module defines the FastAPI application and the endpoints for the chat API.

It imports necessary modules and functions, sets up logging, 
loads global settings, and initializes the chat agent and the Webex bot manager.

It defines a Pydantic model for the message data and two POST endpoints:
one for sending messages to the chat agent and another for processing alerts.
"""
import uvicorn
import threading
from fastapi import FastAPI

from logging_config.main import setup_logging
from load_global_settings import (
    HOST_URL,
    LLM_HTTP_PORT,
)
from webex.bot import WebexBotManager
from llm_agent import LLMChatAgent

from fastapi_models import Message


app = FastAPI()
logger = setup_logging()
chat_agent = LLMChatAgent()
webex_bot_manager = WebexBotManager()


@app.post("/chat")
def chat_to_llm(message: Message) -> str:
    logger.info(f"MESSAGE_RECEIVED: {message.message}")
    return chat_agent.chat(message.message)

def notify(notification: str) -> None:
    """
    Sends a notification message.
    """
    logger.info(f"SENDING_NOTIFICATION: {notification}")
    webex_bot_manager.send_notification(notification)


if __name__ == "__main__":
    threading.Thread(
        target=uvicorn.run,
        args=("app:app",),
        kwargs={"host": HOST_URL, "port": LLM_HTTP_PORT},
    ).start()
    webex_bot_manager.run()