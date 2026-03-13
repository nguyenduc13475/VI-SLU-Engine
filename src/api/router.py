import logging
from fastapi import APIRouter, HTTPException

from .schemas import ParseRequest, ParseResponse
from ..engine.interpreter import ExecutionPlanInterpreter

from ..engine.bigru import IntentBiGRU

logger = logging.getLogger(__name__)

# Initialize the Router (Group the APIs related to NLU/SLU)
router = APIRouter(prefix="/api/v1", tags=["Spoken Language Understanding"])

# Initialize the AI ​​model (Singleton Pattern - Loads weights once when the server is turned on).
try:
    slu_model = IntentBiGRU()
    # The load_weights() function in the main.py file will be called when the app starts up.
except Exception as e:
    logger.error(f"Failed to initialize AI Model: {e}")
    slu_model = None


@router.post("/parse", response_model=ParseResponse)
async def parse_command(request: ParseRequest):
    """
    **Parse Vietnamese Natural Language Command into an IoT Execution Plan.**
    
    - **text**: The spoken/typed text.
    - **Returns**: A JSON array containing exact seconds to wait and actions to perform.
    """
    if not slu_model:
        raise HTTPException(status_code=503, detail="AI Model is not currently available.")

    try:
        # 1. AI Layer: Extract Semantic Tuples (e.g., [('BatDen', 10)])
        predicted_tuples = slu_model.predict(request.text)
        
        # 2. Logic Layer: Build the robust JSON plan
        plan_dict = ExecutionPlanInterpreter.generate_plan(
            raw_text=request.text, 
            predicted_tuples=predicted_tuples
        )
        
        # 3. FastAPI & Pydantic automatically validate 'plan_dict' against 'ParseResponse'
        return plan_dict

    except ValueError as ve:
        # Error due to invalid input text format.
        logger.warning(f"Validation Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Server error (Code bug, Tensor mismatch, etc.)
        logger.error(f"Internal Server Error during parsing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the command.")