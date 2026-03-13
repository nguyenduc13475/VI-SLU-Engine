from typing import List, Dict, Any, Tuple, Union

class ExecutionPlanInterpreter:
    """
    Stateless Interpreter that translates raw semantic tuples from the NLP models 
    into a standardized JSON Execution Plan. 
    """
    
    # Mapping Vietnamese Intent tags to Standardized System Actions
    INTENT_ACTION_MAP: Dict[str, str] = {
        'BatDen': 'LED_ON',
        'TatDen': 'LED_OFF',
        'BatQuat': 'FAN_ON',
        'TatQuat': 'FAN_OFF',
        'QuatNhanh': 'FAN_SPEED_UP',
        'QuatCham': 'FAN_SPEED_DOWN',
        'MoCua': 'DOOR_OPEN',
        'DongCua': 'DOOR_CLOSE',
        'NhietDo': 'CHECK_TEMPERATURE',
        'DoAm': 'CHECK_HUMIDITY'
    }

    @classmethod
    def _parse_time_info(cls, time_info: Union[int, float, Tuple]) -> Dict[str, Union[float, None]]:
        """
        Parses the complex nested time tuples into a flat dictionary of temporal parameters.
        
        Formats handled:
        1. T -> wait T
        2. (Start, End) -> wait Start, last for (End - Start)
        3. (Start, End, Interval) -> wait Start, repeat every Interval until End
        4. (Start, End, Interval, Execution_Time) -> like 3, but hold action for Execution_Time
        
        Args:
            time_info: The time structure extracted by the converter.
            
        Returns:
            Dict containing delay, duration, interval, and execution seconds.
        """
        # Default temporal state
        parsed_time: Dict[str, Union[float, None]] = {
            "delay_seconds": 0.0,
            "duration_seconds": None,
            "interval_seconds": None,
            "hold_seconds": None  # Corresponds to element 'E' in format 4
        }

        if isinstance(time_info, (int, float)):
            parsed_time["delay_seconds"] = float(time_info)

        elif isinstance(time_info, tuple):
            length = len(time_info)
            if length >= 2:
                start, end = time_info[0], time_info[1]
                parsed_time["delay_seconds"] = float(start)
                parsed_time["duration_seconds"] = float(end - start) if end > start else None

            if length >= 3:
                parsed_time["interval_seconds"] = float(time_info[2])

            if length == 4:
                parsed_time["hold_seconds"] = float(time_info[3])

        return parsed_time

    @classmethod
    def generate_plan(cls, raw_text: str, predicted_tuples: List[Tuple]) -> Dict[str, Any]:
        """
        Constructs the final Execution Plan payload to be returned by the API.
        
        Args:
            raw_text (str): The original spoken/typed command.
            predicted_tuples (List[Tuple]): The semantic tuples from BiGRU/Rule-based.
            
        Returns:
            Dict[str, Any]: A JSON-serializable dictionary representing the system instructions.
        """
        intents_list: List[str] = []
        execution_plan: List[Dict[str, Any]] = []

        for item in predicted_tuples:
            if not item or len(item) != 2:
                continue

            intent_tag, time_info = item
            intents_list.append(intent_tag)

            # Map the NLP intent to a robust system action. 
            # Fallback to uppercase intent if not explicitly mapped.
            action = cls.INTENT_ACTION_MAP.get(intent_tag, intent_tag.upper())

            # Resolve timing rules
            timing_details = cls._parse_time_info(time_info)

            # Construct the execution step
            step = {
                "action": action,
                **timing_details
            }
            execution_plan.append(step)

        # Deduplicate intents while preserving order (for summary purposes)
        unique_intents = list(dict.fromkeys(intents_list))

        return {
            "raw_text": raw_text,
            "intents": unique_intents,
            "execution_plan": execution_plan
        }