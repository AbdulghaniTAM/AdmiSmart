import time
from typing import List, Dict, Any
from execution_logging import run_observed_agent

# 1. Establish the Definitive Golden Evaluation Benchmark Dataset
# Anchored to target guidelines: May 30, 2026 deadline, transfer credits, visa escalation rules
ADMISSMART_GOLDEN_SET = [
    {
        "case_id": "EVAL_001",
        "category": "GREEN_FACTUAL",
        "user_query": "What is the final deadline to apply for the bootcamp?",
        "target_keywords": ["May 30, 2026", "deadline"],
        "forbidden_keywords": ["escalated", "Senior Admission", "Staff Review Queue"]
    },
    {
        "case_id": "EVAL_002",
        "category": "GREEN_STATUS",
        "user_query": "Check current admission status for applicant registration ID: 404",
        "target_keywords": ["No applicant record found", "404"],
        "forbidden_keywords": ["May 30", "bootcamp"]
    },
    {
        "case_id": "EVAL_003",
        "category": "YELLOW_COMPLIANCE",
        "user_query": "I have some transfer credit requests from my previous college. Can you apply them?",
        "target_keywords": ["Staff Review Queue", "manual", "transfer credit"],
        "forbidden_keywords": ["Senior Admission Officers", "May 30, 2026"]
    },
    {
        "case_id": "EVAL_004",
        "category": "RED_ESCALATION",
        "user_query": "I need help with my visa sponsorship for the upcoming course.",
        "target_keywords": ["Senior Admission Officers", "escalat"],
        "forbidden_keywords": ["Auto-Resolved", "Staff Review Queue"]
    },
    {
        "case_id": "EVAL_005",
        "category": "RED_COMPLAINT",
        "user_query": "This system is terrible. I want to file a formal complaint about my application rejection.",
        "target_keywords": ["Senior Admission Officers", "escalat"],
        "forbidden_keywords": ["Auto-Resolved", "Staff Review Queue"]
    },
    {
        "case_id": "EVAL_006",
        "category": "ABSENCE_FALLBACK",
        "user_query": "What is the lunch menu for the campus cafeteria?",
        "target_keywords": ["human admissions officer", "lack", "verified data"],
        "forbidden_keywords": ["May 30, 2026", "Staff Review Queue", "Senior Admission"]
    }
]

# 2. Automated Evaluation Engine Scorer
def evaluate_agent_response(response_text: str, target_words: List[str], blocked_words: List[str]) -> Dict[str, Any]:
    """
    Analyzes agent responses against expected keywords, forbidden terms, 
    and policy compliance rules to compute a final quality score.
    """
    text_lower = response_text.lower()
    
    # Calculate target keyword matches
    matched_targets = [word for word in target_words if word.lower() in text_lower]
    target_score = (len(matched_targets) / len(target_words)) * 100 if target_words else 100.0
    
    # Calculate forbidden keyword penalties
    triggered_violations = [word for word in blocked_words if word.lower() in text_lower]
    penalty = len(triggered_violations) * 25.0
    
    # Final normalized calculation
    final_score = max(0.0, target_score - penalty)
    
    # Pass threshold set at a minimum of 70% accuracy with 0 policy violations
    passed = (final_score >= 70.0) and (len(triggered_violations) == 0)
    
    return {
        "score": round(final_score, 2),
        "passed": passed,
        "matched_count": len(matched_targets),
        "total_targets": len(target_words),
        "violations": triggered_violations
    }

# 3. Batch Evaluation Suite Runner
def run_automated_evaluation_suite():
    print("=" * 70)
    print("STARTING ADMISSMART SYSTEM-WIDE AUTOMATED EVALUATION RUN")
    print("=" * 70)
    
    suite_start_time = time.perf_counter()
    passed_cases = 0
    cumulative_score = 0.0
    detailed_metrics = []
    
    for case in ADMISSMART_GOLDEN_SET:
        print(f"Processing {case['case_id']} ({case['category']})...")
        
        # Run through instrumented logging wrapper
        trace_log = run_observed_agent(case["user_query"])
        agent_answer = trace_log["final_output"]
        
        # Grade output automatically
        score_card = evaluate_agent_response(
            response_text=agent_answer,
            target_words=case["target_keywords"],
            blocked_words=case["forbidden_keywords"]
        )
        
        cumulative_score += score_card["score"]
        if score_card["passed"]:
            passed_cases += 1
            
        detailed_metrics.append({
            "id": case["case_id"],
            "category": case["category"],
            "score": score_card["score"],
            "passed": score_card["passed"],
            "execution_ms": trace_log["execution_time_ms"],
            "violations": score_card["violations"]
        })
        
    total_cases = len(ADMISSMART_GOLDEN_SET)
    suite_duration = time.perf_counter() - suite_start_time
    
    # Calculate summary statistics
    final_success_rate = (passed_cases / total_cases) * 100
    final_avg_score = cumulative_score / total_cases
    
    print("\n" + "=" * 70)
    print("AUTOMATED PERFORMANCE DASHBOARD EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Evaluated Test Cases : {total_cases}")
    print(f"Overall Success Rate       : {final_success_rate:.2f}%")
    print(f"Average Evaluation Score   : {final_avg_score:.2f}%")
    print(f"Total Suite Runtime        : {suite_duration:.2f} seconds")
    print("-" * 70)
    print(f"{'Case ID':<10} | {'Category':<22} | {'Score':<6} | {'Passed':<6} | {'Latency (ms)':<10}")
    print("-" * 70)
    for metric in detailed_metrics:
        print(f"{metric['id']:<10} | {metric['category']:<22} | {metric['score']:<6}% | {str(metric['passed']):<6} | {metric['execution_ms']:<10.2f}")
    print("=" * 70)

if __name__ == "__main__":
    run_automated_evaluation_suite()
