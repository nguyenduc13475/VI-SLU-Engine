import re
from typing import List, Tuple, Dict

class TimeParser:
    """
    Utility class for handling temporal expressions in Vietnamese.
    Extracts time concepts and groups them into execution tuples.
    """

    @staticmethod
    def is_time_word(word: str) -> bool:
        """
        Checks if a given token belongs to the time-indicating category.
        
        Args:
            word (str): The token to check.
            
        Returns:
            bool: True if it's a number or time unit, False otherwise.
        """
        if re.match(r'^\d+$', str(word)): 
            return True
        if str(word).lower() in ['ngày', 'giây', 'phút', 'giờ', 'tiếng', 'rưỡi', 'nửa']: 
            return True
        return False

    @staticmethod
    def parse_time_phrases(words: List[str]) -> float:
        """
        Converts a sequence of time tokens into total seconds.
        Example: ["1", "phút", "20", "giây"] -> 80.0
        
        Args:
            words (List[str]): List of time-related words.
            
        Returns:
            float: Total time in seconds.
        """
        text = " ".join(words).lower().replace("và", "")
        tokens = text.split()
        
        total_sec: float = 0.0
        current_num: float = 0.0
        last_unit: float = 0.0

        for w in tokens:
            if re.match(r'^\d+$', w):
                current_num = float(w)
            elif w == 'nửa':
                current_num = 0.5
            elif w == 'ngày': 
                total_sec += current_num * 86400
                last_unit = 86400
                current_num = 0
            elif w in ['giờ', 'tiếng']:
                total_sec += current_num * 3600
                last_unit = 3600
                current_num = 0
            elif w == 'phút':
                total_sec += current_num * 60
                last_unit = 60
                current_num = 0
            elif w == 'giây':
                total_sec += current_num * 1
                last_unit = 1
                current_num = 0
            elif w == 'rưỡi':
                total_sec += 0.5 * last_unit
                
        # Handle trailing numbers without explicit units (default to seconds)
        if current_num > 0 and last_unit == 0:
            total_sec += current_num
            
        return float(total_sec)

    @classmethod
    def extract_temporal_tuples(cls, words: List[str], tags: List[str]) -> List[Tuple]:
        """
        Converts parallel sequences of words and NER tags into actionable tuples.
        
        Args:
            words (List[str]): Original sentence tokens.
            tags (List[str]): Corresponding model predicted tags.
            
        Returns:
            List[Tuple]: Execution tuples e.g., ('BatDen', 10.0)
        """
        blocks: List[Dict[str, List[str]]] = []
        current_block = {'intents': [], 'TimeAfter': [], 'TimeWithin': [], 'TimeRepeat': [], 'TimeRange': []}

        for w, t in zip(words, tags):
            if t == 'O': 
                continue
            if t == 'Sep':
                blocks.append(current_block)
                current_block = {'intents': [], 'TimeAfter': [], 'TimeWithin': [], 'TimeRepeat': [], 'TimeRange': []}
            elif t in ['BatDen', 'TatDen', 'MoCua', 'DongCua', 'BatQuat', 'TatQuat', 'QuatNhanh', 'QuatCham', 'NhietDo', 'DoAm']:
                if t not in current_block['intents']:
                    current_block['intents'].append(t)
            elif t in ['TimeAfter', 'TimeWithin', 'TimeRepeat', 'TimeRange']:
                current_block[t].append(w)
                
        blocks.append(current_block)

        results: List[Tuple] = []
        ref_time: float = 0.0

        for block in blocks:
            if not block['intents']:
                continue

            t_after = cls.parse_time_phrases(block['TimeAfter']) if block['TimeAfter'] else 0.0
            t_within = cls.parse_time_phrases(block['TimeWithin']) if block['TimeWithin'] else 0.0
            t_repeat = cls.parse_time_phrases(block['TimeRepeat']) if block['TimeRepeat'] else 0.0
            t_range = cls.parse_time_phrases(block['TimeRange']) if block['TimeRange'] else 0.0

            current_start = ref_time + t_after

            for intent in block['intents']:
                if t_repeat > 0:
                    rng = t_range if t_range > 0 else 999999.0 # Infinite loop fallback
                    results.append((intent, (current_start, current_start + rng, t_repeat) + ((t_within,) if t_within > 0 else tuple())))
                elif t_within > 0:
                    results.append((intent, (current_start, current_start + t_within)))
                else:
                    results.append((intent, current_start))

            # Update reference timeline for subsequent blocks (separated by 'Sep')
            ref_time = current_start
            if t_within > 0:
                ref_time += t_within

        return results