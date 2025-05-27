"""
Schema models for the application
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class Document(BaseModel):
    """Schema for a document in the system"""
    id: str
    content: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
