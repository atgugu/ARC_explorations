"""Quick test of conditional solver on a few tasks"""

import json
import numpy as np
from pathlib import Path
from arc_curiosity_solver.solver_conditional import ConditionalARCCuriositySolver

def test_quick():
    solver = ConditionalARCCuriositySolver()

    # Test on previously solved tasks + a few others
    test_tasks = [
        '25ff71a9',  # Previously solved
        '3c9b0459',  # Previously solved
        '0b17323b',  # 99.11% (very close!)
        '025d127b',  # 98.00%
        '27a77e38',  # 98.77%
    ]

    training_dir = Path("ARC-AGI/data/training")

    print("Testing Conditional Solver on 5 tasks\n")
    print("="*60)

    for task_id in test_tasks:
        task_file = training_dir / f"{task_id}.json"

        if not task_file.exists():
            print(f"✗ {task_id}: Not found")
            continue

        with open(task_file, 'r') as f:
            task_data = json.load(f)

        try:
            # Test
            train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                          for ex in task_data['train']]
            test_input = np.array(task_data['test'][0]['input'])
            expected = np.array(task_data['test'][0]['output'])

            # Generate hypotheses
            hypotheses = solver._generate_hypotheses(train_pairs, test_input)

            # Evaluate top hypotheses
            predictions = []
            for h in hypotheses[:2]:
                try:
                    pred = h.program.function(test_input.copy())
                    predictions.append(pred)
                except:
                    predictions.append(test_input.copy())

            # Check accuracy
            if len(predictions) >= 1:
                match1 = np.array_equal(predictions[0], expected)
                match2 = np.array_equal(predictions[1], expected) if len(predictions) > 1 else False

                if match1 or match2:
                    accuracy = 100.0
                    symbol = "✓"
                else:
                    accuracy = (predictions[0] == expected).mean() * 100 if predictions[0].shape == expected.shape else 0
                    symbol = " "

                print(f"{symbol} {task_id}: {accuracy:.1f}% ({len(hypotheses)} hypotheses)")
            else:
                print(f"✗ {task_id}: No predictions")

        except Exception as e:
            print(f"✗ {task_id}: ERROR - {str(e)[:50]}")

    print("="*60)

if __name__ == '__main__':
    test_quick()
