from automated_evaulation import ADMISSMART_GOLDEN_SET, evaluate_agent_response
from execution_logging import run_observed_agent
import time

for case in [ADMISSMART_GOLDEN_SET[0], ADMISSMART_GOLDEN_SET[1], ADMISSMART_GOLDEN_SET[4]]:
    print(f"Testing {case['case_id']}")
    trace = run_observed_agent(case["user_query"])
    ans = trace["final_output"]
    print(f"OUTPUT: {ans}")
    score = evaluate_agent_response(ans, case["target_keywords"], case["forbidden_keywords"])
    print(f"SCORE: {score}")
    print("-" * 50)
