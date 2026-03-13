from torch.utils.data import DataLoader
from typing import List, Dict, Any

from src.core.config import config
from src.engine.dataset import SmartHomeDataset
from src.engine.bigru import IntentBiGRU
from src.engine.rule_based import IntentRuleBased
from src.utils.metrics import evaluate_model

def print_pretty_errors(model_name: str, accuracy: float, errors: List[Dict[str, Any]], max_display: int = 5) -> None:
    """
    Prints the evaluation report and a subset of errors to the terminal with ANSI colors.
    
    Args:
        model_name (str): The name of the evaluated model.
        accuracy (float): The accuracy ratio (0.0 to 1.0).
        errors (List[Dict]): List of error dictionaries containing 'sentence', 'predicted', and 'true'.
        max_display (int): Maximum number of errors to display.
    """
    # ANSI Color Codes
    CYAN = '\033[96m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}{YELLOW} 🚀 BÁO CÁO ĐÁNH GIÁ: {model_name.upper()}{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f" 🎯 Độ chính xác (Accuracy): {BOLD}{GREEN}{accuracy*100:.2f}%{RESET}")
    print(f" 🐞 Tổng số câu sai      : {BOLD}{RED}{len(errors)}{RESET}")
    
    if len(errors) > 0:
        display_count = min(max_display, len(errors))
        print(f"\n{'-'*65}")
        print(f" 🔍 CHI TIẾT {display_count} LỖI TIÊU BIỂU:")
        print(f"{'-'*65}")
        for i, err in enumerate(errors[:display_count]):
            print(f" 🔴 {BOLD}Lỗi #{i+1}:{RESET}")
            print(f"    📝 Input     : {CYAN}{err['sentence']}{RESET}")
            print(f"    ❌ Dự đoán   : {RED}{err['predicted']}{RESET}")
            print(f"    ✅ Thực tế   : {GREEN}{err['true']}{RESET}")
            print(f"{'-'*65}")
    print("\n")

def main() -> None:
    """Main execution function for the evaluation script."""
    print("⏳ Đang chuẩn bị dữ liệu đánh giá...")
    
    try:
        # Initialize BiGRU first to load the vocabulary needed for the Dataset
        bigru_model = IntentBiGRU()
        vocab = bigru_model.vocab
    except Exception as e:
        print(f"❌ Không thể khởi tạo BiGRU Model: {e}")
        return

    # Load validation data
    val_dataset = SmartHomeDataset(config.VAL_DATA_PATH, vocab)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=IntentBiGRU.custom_collate_fn
    )

    # 1. Evaluate BiGRU Model
    print("\n🧠 Đang đánh giá mô hình BiGRU...")
    bigru_acc, bigru_errors = evaluate_model(bigru_model, val_loader)
    print_pretty_errors("BiGRU Model", bigru_acc, bigru_errors, max_display=10)
    
    # 2. Evaluate Rule-Based Model
    print("⚙️ Đang đánh giá mô hình Rule-Based...")
    rule_based_model = IntentRuleBased()
    rule_based_acc, rule_based_errors = evaluate_model(rule_based_model, val_loader)
    print_pretty_errors("Rule-Based Model", rule_based_acc, rule_based_errors, max_display=10)

if __name__ == "__main__":
    main()