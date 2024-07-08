"""
This module contains Pydantic models for handling webhook messages in a FastAPI application.
"""

from pydantic import BaseModel


class Message(BaseModel):
    """
    This class represents a message model.
    """

    message: str