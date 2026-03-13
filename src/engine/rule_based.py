from typing import List, Dict, Union, Tuple, Any
from .time_parser import TimeParser

class IntentRuleBased:
    """
    Rule-based model for extracting Intents and Time Slots.
    Utilizes N-gram Matching for Intents and a Sliding Context Window for Temporal tags.
    Acts as a lightweight, lightning-fast fallback mechanism.
    """
    
    def __init__(self) -> None:
        # Rule vocabulary for Intents (Supports up to 4-grams)
        # Includes negative cases (e.g., 'không bật' -> 'Tat')
        self.intent_rules: Dict[str, List[List[str]]] = {
            'TatDen': [['không', 'bật', 'đèn'], ['cấm', 'bật', 'đèn'], ['tắt', 'đèn'], ['đóng', 'đèn']],
            'BatDen': [['bật', 'đèn'], ['mở', 'đèn']],
            
            'TatQuat': [['không', 'bật', 'quạt'], ['cấm', 'bật', 'quạt'], ['tắt', 'quạt'], ['đóng', 'quạt']],
            'BatQuat': [['bật', 'quạt'], ['mở', 'quạt']],
            
            'MoCua': [['mở', 'cửa']],
            'DongCua': [['đóng', 'cửa']],
            
            'QuatNhanh': [['tăng', 'tốc', 'độ', 'quạt'], ['tăng', 'tốc'], ['quạt', 'nhanh'], ['mạnh', 'lên']],
            'QuatCham': [['giảm', 'tốc', 'độ', 'quạt'], ['giảm', 'tốc'], ['quạt', 'chậm'], ['yếu', 'đi']],
            
            'NhietDo': [['nhiệt', 'độ']],
            'DoAm': [['độ', 'ẩm']],
            
            'Sep': [['rồi', 'sau', 'đó'], ['rồi'], ['sau', 'đó'], ['kế', 'tiếp']]
        }

    def predict(self, texts: Union[List[str], List[List[str]]]) -> Union[List[Tuple], List[List[Tuple]]]:
        """
        Predicts intent and temporal tags for a single sentence or a batch of sentences.
        
        Args:
            texts: A list of words (1 sentence) or a list of lists of words (batch).
            
        Returns:
            Extracted execution tuples mapped from tags.
        """
        # Determine if input is a single sentence or a batch
        is_single = len(texts) > 0 and isinstance(texts[0], str)
        batch_words = [texts] if is_single else texts
            
        batch_results: List[Any] = []
        
        for words in batch_words:
            tags = ['O'] * len(words)
            words_lower = [str(w).lower() for w in words]

            # 1. N-gram Dictionary Matching for Intents and Separators
            for i in range(len(words_lower)):
                if tags[i] != 'O': 
                    continue
                matched = False
                for length in [4, 3, 2, 1]:
                    if i + length <= len(words_lower):
                        phrase = words_lower[i:i+length]
                        for intent, patterns in self.intent_rules.items():
                            if phrase in patterns:
                                for j in range(length): 
                                    tags[i+j] = intent
                                matched = True
                                break
                    if matched: 
                        break

            # 2. Heuristics & Contextual Modifiers
            for i, w in enumerate(words_lower):
                if tags[i] == 'O':
                    # Sử dụng TimeParser thay cho hàm cũ
                    if w == 'và' and i + 1 < len(words_lower) and TimeParser.is_time_word(words_lower[i+1]):
                        tags[i] = 'Sep'
                    elif w == 'hơn' and i > 0 and tags[i-1] == 'QuatNhanh': 
                        tags[i] = 'QuatNhanh'
                    elif w == 'lại' and i > 0 and tags[i-1] in ['QuatCham', 'BatDen']: 
                        tags[i] = tags[i-1]

            # 3. Sliding Window for Temporal Tags Extraction
            i = 0
            while i < len(words_lower):
                if TimeParser.is_time_word(words_lower[i]):
                    start = i
                    while i < len(words_lower) and TimeParser.is_time_word(words_lower[i]): 
                        i += 1
                    end = i

                    context_full = words_lower[max(0, start-2):start] + words_lower[end:min(len(words_lower), end+2)]
                    time_tag = None
                    
                    if any(w in context_full for w in ['cứ', 'mỗi', 'lần']):
                        time_tag = 'TimeRepeat'
                    elif any(w in context_full for w in ['duy', 'trì', 'kéo', 'dài']):
                        time_tag = 'TimeRange'
                    elif any(w in context_full for w in ['trong']):
                        time_tag = 'TimeWithin'
                    elif any(w in context_full for w in ['sau']):
                        time_tag = 'TimeAfter'
                    else:
                        time_tag = 'TimeAfter'  # Default fallback
                    
                    for j in range(start, end):
                        tags[j] = time_tag
                else:
                    i += 1
                    
            # 4. Convert tags directly into tuples using the TimeParser module (KHÔNG CÒN TODO)
            pred_tuples = TimeParser.extract_temporal_tuples(words, tags)
            batch_results.append(pred_tuples)
            
        return batch_results[0] if is_single else batch_results