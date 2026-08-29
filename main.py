import os
from src.credit_engine import CreditRiskEngine
from src.vector_store import RegulatoryVectorDB
from src.compliance_agent import ComplianceSLMAgent

def main():
    print("Step 1: Training Credit Scorecard & Backtesting...")
    engine = CreditRiskEngine()
    X_train, X_test, y_train, y_test = engine.load_and_preprocess()
    engine.train(X_train, y_train)
    metrics_payload = engine.evaluate(X_train, X_test, y_train, y_test)

    print("Step 2: Initializing Regulatory FAISS Vector Store...")
    vector_db = RegulatoryVectorDB(docs_dir="data/regulatory_docs/")
    vector_db.build()

    print("Step 3: Running Compliance SLM Agent...")
    agent = ComplianceSLMAgent(vector_db=vector_db)
    validation_memo = agent.run_compliance_audit(metrics_payload)

    os.makedirs("output", exist_ok=True)
    with open("output/validation_memo.md", "w", encoding="utf-8") as f:
        f.write(validation_memo)

    print("\nValidation Memo generated at output/validation_memo.md\n")
    print(validation_memo)

if __name__ == "__main__":
    main()
