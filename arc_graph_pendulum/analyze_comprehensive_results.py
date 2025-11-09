"""
Analyze comprehensive test results to identify failure patterns and priorities.
"""

import json
import numpy as np
from collections import defaultdict
from solver_v3_plus import ARCGraphPendulumSolverV3Plus
from utils.arc_loader import ARCLoader

# Load results
with open('comprehensive_test_results.json', 'r') as f:
    data = json.load(f)

results = data['results']

# Categorize by performance
perfect = [r for r in results if r['avg_score'] >= 0.99]
high_quality = [r for r in results if 0.8 <= r['avg_score'] < 0.99]
medium_quality = [r for r in results if 0.5 <= r['avg_score'] < 0.8]
failures = [r for r in results if r['avg_score'] < 0.2]

print("="*80)
print("COMPREHENSIVE ANALYSIS - 46 DIVERSE ARC TASKS")
print("="*80)

print("\n" + "="*80)
print("PERFORMANCE BREAKDOWN")
print("="*80)
print(f"Perfect solves (≥0.99):      {len(perfect):3d} ({len(perfect)/len(results)*100:5.1f}%)")
print(f"High quality (0.80-0.98):   {len(high_quality):3d} ({len(high_quality)/len(results)*100:5.1f}%)")
print(f"Medium quality (0.50-0.79):  {len(medium_quality):3d} ({len(medium_quality)/len(results)*100:5.1f}%)")
print(f"Failures (<0.20):            {len(failures):3d} ({len(failures)/len(results)*100:5.1f}%)")

print("\n" + "="*80)
print("PERFECT SOLVES (≥0.99 IoU)")
print("="*80)
for r in perfect:
    print(f"  {r['task_id']}: {r['avg_score']:.3f}")

print("\n" + "="*80)
print("COMPLETE FAILURES (<0.20 IoU)")
print("="*80)
for r in failures:
    print(f"  {r['task_id']}: {r['avg_score']:.3f}")

# Now analyze what transformations the failed tasks need
print("\n" + "="*80)
print("ANALYZING FAILED TASKS...")
print("="*80)

loader = ARCLoader(cache_dir='./arc_data')
all_tasks = loader.load_all_tasks('training')

failure_analysis = []

for r in failures:
    task = all_tasks[r['task_id']]

    # Get basic stats
    analysis = {
        'task_id': r['task_id'],
        'iou': r['avg_score'],
        'num_train': task.num_train,
        'num_test': task.num_test,
    }

    # Analyze first training example
    if task.num_train > 0:
        inp, out = task.train[0]
        analysis['input_shape'] = inp.shape
        analysis['output_shape'] = out.shape
        analysis['shape_change'] = inp.shape != out.shape
        analysis['size_ratio'] = (out.shape[0] * out.shape[1]) / (inp.shape[0] * inp.shape[1])
        analysis['input_colors'] = len(np.unique(inp))
        analysis['output_colors'] = len(np.unique(out))
        analysis['color_change'] = len(np.unique(inp)) != len(np.unique(out))

    failure_analysis.append(analysis)

# Group failures by characteristics
shape_changing = [f for f in failure_analysis if f.get('shape_change', False)]
same_shape = [f for f in failure_analysis if not f.get('shape_change', False)]
color_changing = [f for f in failure_analysis if f.get('color_change', False)]

print(f"\nShape-changing tasks: {len(shape_changing)}/{len(failures)}")
print(f"Same-shape tasks: {len(same_shape)}/{len(failures)}")
print(f"Color-changing tasks: {len(color_changing)}/{len(failures)}")

# Analyze size ratios for shape-changing tasks
if shape_changing:
    print(f"\nShape-changing task size ratios:")
    for f in shape_changing:
        print(f"  {f['task_id']}: {f['input_shape']} → {f['output_shape']} (ratio: {f['size_ratio']:.2f})")

# Save detailed analysis
with open('failure_analysis.json', 'w') as f:
    json.dump({
        'total_failures': len(failures),
        'shape_changing': len(shape_changing),
        'same_shape': len(same_shape),
        'color_changing': len(color_changing),
        'details': failure_analysis
    }, f, indent=2)

print("\n✓ Detailed analysis saved to failure_analysis.json")

# Now let's look at what worked well
print("\n" + "="*80)
print("WHAT WORKED WELL (High quality ≥0.80)")
print("="*80)

high_quality_all = perfect + high_quality
print(f"Total high-quality: {len(high_quality_all)}/{len(results)} ({len(high_quality_all)/len(results)*100:.1f}%)")

# Analyze characteristics of successful tasks
success_analysis = []
for r in high_quality_all:
    task = all_tasks[r['task_id']]
    if task.num_train > 0:
        inp, out = task.train[0]
        success_analysis.append({
            'task_id': r['task_id'],
            'iou': r['avg_score'],
            'shape_change': inp.shape != out.shape,
            'size_ratio': (out.shape[0] * out.shape[1]) / (inp.shape[0] * inp.shape[1]),
        })

success_shape_changing = [s for s in success_analysis if s['shape_change']]
success_same_shape = [s for s in success_analysis if not s['shape_change']]

print(f"\nSuccessful shape-changing: {len(success_shape_changing)}/{len(success_analysis)}")
print(f"Successful same-shape: {len(success_same_shape)}/{len(success_analysis)}")

if success_shape_changing:
    print(f"\nSuccessful shape-changing size ratios:")
    for s in success_shape_changing[:10]:
        print(f"  {s['task_id']}: {s['iou']:.3f} (ratio: {s['size_ratio']:.2f})")
