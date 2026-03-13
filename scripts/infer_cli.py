import json
from src.engine.bigru import IntentBiGRU
from src.engine.rule_based import IntentRuleBased
from src.engine.interpreter import ExecutionPlanInterpreter

def main():
    print("🤖 Vi-SLU Engine CLI Interative Mode")
    print("Type 'exit' or 'quit' to stop.\n")
    
    try:
        model = IntentBiGRU()
        print("✅ BiGRU Model loaded successfully.")
    except Exception as e:
        print(f"⚠️ BiGRU Model load failed: {e}. Falling back to Rule-Based.")
        model = IntentRuleBased()

    while True:
        try:
            user_input = input("🗣️ Enter command: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                continue
                
            # 1. Prediction
            tuples = model.predict(user_input)
            
            # 2. Convert to JSON Plan
            plan = ExecutionPlanInterpreter.generate_plan(user_input, tuples)
            
            print("\n⚙️ Execution Plan:")
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            print("-" * 50)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()