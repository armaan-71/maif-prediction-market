llm_decision = __import__('llm-decision')
compare_consistent = llm_decision.compare_consistent
from test_data import get_test_data

data = get_test_data()

# test pairs and their expected outputs. hard-coded from test_data.py just for debugging purposes.
# format: (idx1, idx2, expected-relation)

total_tests = 0
correct_tests = 0

pairs = [
    # IDENTICAL
    (0, 1, "IDENTICAL"),
    (6, 7, "IDENTICAL"),

    # COMPLEMENTS
    (0, 2, "COMPLEMENT"),
    (9, 10, "COMPLEMENT"),

    # SUBSET / SUPERSET (threshold logic)
    (4, 0, "SUBSET"),     # 120k ⊆ 110k
    (0, 3, "SUBSET"),     # 110k ⊆ 100k
    (3, 0, "SUPERSET"),

    # TIME WINDOW SUBSET
    (5, 0, "SUBSET"),     # shorter window

    # OPERATOR EDGE
    (8, 0, "SUBSET"),     # >110k ⊆ >=110k

    # CPI MACRO
    (11, 12, "SUBSET"),   # >=4 ⊆ >=3
    (12, 11, "SUPERSET"),

    # MUTUALLY EXCLUSIVE
    (13, 14, "MUTUALLY_EXCLUSIVE"),

    # UNRELATED
    (0, 6, "UNRELATED"),
    (0, 11, "UNRELATED"),
    (6, 13, "UNRELATED"),
    (15, 0, "UNRELATED"),
]

for pair in pairs:
    idx1 = pair[0]
    idx2 = pair[1]
    expected = pair[2]
    print(f"Comparing {data[idx1]['payout_condition']} to {data[idx2]['payout_condition']} | EXPECTED: {expected}")
    result = compare_consistent(data[idx1], data[idx2], model="gpt-5.4-mini")
    print(f"Result: {result.label} | EXPECTED: {expected} | CONFIDENCE: {result.confidence}")
    print(f'{"✅" if result.label==expected else "❌"}')
    if result.label == expected:
        correct_tests += 1
    total_tests += 1
    print("\n"+ "="*10 + "\n")
print(f"Total tests: {total_tests}")
print(f"Correct tests: {correct_tests}")
print(f"Score: {correct_tests / total_tests * 100}%")