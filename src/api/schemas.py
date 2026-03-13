from typing import List, Optional
from pydantic import BaseModel, Field

class ParseRequest(BaseModel):
    """
    Schema for the incoming client request.
    """
    text: str = Field(
        ..., 
        description="The natural language command in Vietnamese.",
        json_schema_extra={"example": "bật đèn và quạt nhanh lên sau 10 giây nữa"}
    )

class ExecutionStep(BaseModel):
    """
    Schema for a single execution step within the plan.
    Strictly follows the Stateless Interpreter output.
    """
    action: str = Field(..., description="The standardized system action (e.g., LED_ON)")
    delay_seconds: float = Field(default=0.0, description="Seconds to wait before executing")
    duration_seconds: Optional[float] = Field(default=None, description="How long to keep the state active")
    interval_seconds: Optional[float] = Field(default=None, description="Time between repeats")
    hold_seconds: Optional[float] = Field(default=None, description="Hold duration inside a repeat loop")

class ParseResponse(BaseModel):
    """
    Schema for the final API response sent back to the client.
    """
    raw_text: str = Field(..., description="The original parsed text")
    intents: List[str] = Field(..., description="List of recognized core intents")
    execution_plan: List[ExecutionStep] = Field(..., description="Chronological array of hardware commands")