# LLM Integration Design: Technical Specification

## Executive Summary

**Goal**: Add semantic understanding to program synthesis by using an LLM to analyze tasks and guide program generation.

**Approach**: **Hybrid LLM-Guided Synthesis** - LLM analyzes task semantics, program synthesis generates verified implementations.

**Expected Impact**: 1% → 15-30% success rate

**Integration Point**: Between Step 1 (Perception) and Step 2 (Program Synthesis)

---

## Current System Flow (9 Steps)

```python
# arc_program_solver.py, solve() method

Step 1: Perception           → Extract basic features (colors, objects)
Step 2: Program Synthesis    → Generate 100+ programs compositionally  ← INSERT LLM HERE
Step 3: Active Inference     → Initialize beliefs (uniform prior)
Step 4: Active Inference     → Update beliefs from training data
Step 5: Stability            → Assess program robustness
Step 6: Workspace            → Select top-20 programs (attention)
Step 7: Ranking              → Score by posterior × stability
Step 8: Selection            → Choose diverse top-2 programs
Step 9: Execution            → Apply to test input, return predictions
```

**Problem**: Step 2 generates programs blindly without understanding what the task is asking for.

**Solution**: Add LLM analysis before Step 2 to guide program generation.

---

## Proposed System Flow (Enhanced)

```python
Step 1: Perception           → Extract basic features
Step 1.5: LLM Analysis       → Understand task semantics  ← NEW
Step 2: Guided Synthesis     → Generate task-specific programs  ← ENHANCED
Step 3-9: Same as before     → Active Inference, selection, execution
```

---

## What the LLM Does (Detailed)

### Input to LLM: Task Visualization + Question

```
You are an expert at analyzing ARC-AGI tasks. Analyze this task and describe the transformation pattern.

Training Example 1:
Input (3×3):
0 0 0
0 8 0
0 0 0

Output (6×6):
0 0 0 0 0 0
0 0 8 8 0 0
0 0 8 8 0 0
0 0 0 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0

Training Example 2:
Input (2×2):
3 0
0 0

Output (4×4):
3 3 0 0
3 3 0 0
0 0 0 0
0 0 0 0

Question: What is the transformation pattern? Be specific about:
1. What operation is applied (scale, rotate, recolor, etc.)?
2. What are the parameters (scale factor, colors, directions)?
3. Are there conditionals or special cases?
4. What should happen to different parts of the input?

Provide a structured answer with specific details.
```

### Output from LLM: Structured Analysis

```
Analysis:
1. Operation: Scale/Zoom
   - Each pixel is zoomed by a factor of 2×2
   - A single pixel becomes a 2×2 block

2. Parameters:
   - Scale factor: 2 (both horizontal and vertical)
   - All colors preserved (no color transformation)
   - Background (0) and foreground pixels both scaled

3. Conditionals: None
   - Simple uniform scaling applied to entire grid

4. Spatial behavior:
   - Top-left alignment
   - Output size = input size × 2 in each dimension
   - No rotation, flip, or translation

Confidence: High (both examples show consistent 2× scaling)

Keywords: zoom, scale, 2x, pixel_to_block
```

### Translation to Program Constraints

The LLM output is parsed and translated into **synthesis constraints**:

```python
{
    'primary_operation': 'zoom',
    'scale_factor': 2,
    'color_transform': None,
    'conditionals_needed': False,
    'spatial_ops': ['scale'],
    'keywords': ['zoom', 'scale', '2x', 'pixel_to_block'],
    'confidence': 'high'
}
```

---

## Where It Fits: Modified Architecture

### Current Code (arc_program_solver.py)

```python
def solve(self, task: ARCTask, verbose: bool = None):
    # Step 1: Perception
    features = {}

    # Step 2: Program Synthesis (CURRENT - BLIND GENERATION)
    hypotheses = self.generator.generate_hypotheses(task, features, verbose)

    # Steps 3-9: Active Inference, ranking, selection...
```

### Modified Code (with LLM)

```python
def solve(self, task: ARCTask, verbose: bool = None):
    # Step 1: Perception
    features = {}

    # Step 1.5: LLM Analysis (NEW)
    if self.llm_analyzer is not None:
        semantic_constraints = self.llm_analyzer.analyze_task(task, verbose)
        features['llm_constraints'] = semantic_constraints

    # Step 2: Guided Synthesis (ENHANCED)
    hypotheses = self.generator.generate_hypotheses(task, features, verbose)

    # Steps 3-9: Same as before
```

**Key Change**: `features` now contains LLM guidance that synthesis can use.

---

## New Component: LLMAnalyzer

### Class Structure

```python
class LLMAnalyzer:
    """
    Uses LLM (Claude/GPT-4) to analyze ARC tasks and extract semantic understanding

    Provides high-level task understanding to guide program synthesis
    """

    def __init__(self, model: str = "claude-sonnet-4", cache_enabled: bool = True):
        self.model = model
        self.cache = {} if cache_enabled else None
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    def analyze_task(self, task: ARCTask, verbose: bool = False) -> Dict:
        """
        Analyze task and return semantic constraints

        Returns:
            {
                'primary_operation': str,      # 'zoom', 'rotate', 'recolor', etc.
                'scale_factor': int,           # 2, 3, etc.
                'color_transform': Dict,       # {0: 1, 1: 2} or None
                'conditionals_needed': bool,   # True if task has if-then logic
                'spatial_ops': List[str],      # ['scale', 'translate', etc.]
                'keywords': List[str],         # Relevant operation keywords
                'confidence': str,             # 'high', 'medium', 'low'
                'description': str,            # Natural language description
            }
        """
        # Check cache
        task_signature = self._compute_task_signature(task)
        if self.cache and task_signature in self.cache:
            return self.cache[task_signature]

        # Create prompt
        prompt = self._create_analysis_prompt(task)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        constraints = self._parse_llm_response(response)

        # Cache result
        if self.cache:
            self.cache[task_signature] = constraints

        return constraints

    def _create_analysis_prompt(self, task: ARCTask) -> str:
        """Convert task to LLM prompt with grid visualizations"""
        prompt = "Analyze this ARC-AGI task:\n\n"

        for i, (input_grid, output_grid) in enumerate(task.train_pairs, 1):
            prompt += f"Training Example {i}:\n"
            prompt += f"Input ({input_grid.height}×{input_grid.width}):\n"
            prompt += self._grid_to_text(input_grid)
            prompt += f"\nOutput ({output_grid.height}×{output_grid.width}):\n"
            prompt += self._grid_to_text(output_grid)
            prompt += "\n"

        prompt += """
What is the transformation pattern?

Provide analysis in this format:
1. Primary Operation: [zoom/rotate/recolor/detect_objects/pattern_match/etc.]
2. Parameters: [specific values like scale=2, rotation=90, colors={0:1, 1:2}]
3. Conditionals: [if-then rules or 'none']
4. Spatial Operations: [list operations like scale, translate, flip, etc.]
5. Confidence: [high/medium/low]
6. Keywords: [relevant operation names for synthesis]

Be specific and concrete.
"""
        return prompt

    def _grid_to_text(self, grid: Grid) -> str:
        """Convert grid to readable text format"""
        lines = []
        for row in grid.data:
            lines.append(' '.join(str(cell) for cell in row))
        return '\n'.join(lines)

    def _call_llm(self, prompt: str) -> str:
        """Call Claude API"""
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into structured constraints"""
        constraints = {
            'primary_operation': None,
            'scale_factor': None,
            'color_transform': None,
            'conditionals_needed': False,
            'spatial_ops': [],
            'keywords': [],
            'confidence': 'medium',
            'description': response,
        }

        # Parse primary operation
        if 'zoom' in response.lower() or 'scale' in response.lower():
            constraints['primary_operation'] = 'zoom'
            constraints['keywords'].append('zoom')

            # Extract scale factor
            import re
            scale_match = re.search(r'(\d+)×', response)
            if scale_match:
                constraints['scale_factor'] = int(scale_match.group(1))

        elif 'rotate' in response.lower():
            constraints['primary_operation'] = 'rotate'
            constraints['keywords'].append('rotate')

            # Extract rotation angle
            if '90' in response:
                constraints['keywords'].append('rotate_90')
            elif '180' in response:
                constraints['keywords'].append('rotate_180')

        elif 'color' in response.lower() or 'recolor' in response.lower():
            constraints['primary_operation'] = 'recolor'
            constraints['keywords'].append('recolor')

        elif 'object' in response.lower():
            constraints['primary_operation'] = 'object_detection'
            constraints['keywords'].extend(['detect_objects', 'object'])

        # Parse conditionals
        if 'if' in response.lower() or 'conditional' in response.lower():
            constraints['conditionals_needed'] = True

        # Parse confidence
        if 'high' in response.lower() and 'confidence' in response.lower():
            constraints['confidence'] = 'high'
        elif 'low' in response.lower() and 'confidence' in response.lower():
            constraints['confidence'] = 'low'

        return constraints

    def _compute_task_signature(self, task: ARCTask) -> str:
        """Compute unique signature for task (for caching)"""
        import hashlib

        # Hash all training examples
        content = ""
        for inp, out in task.train_pairs:
            content += inp.data.tobytes().hex()
            content += out.data.tobytes().hex()

        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

---

## Modified Component: ProgramSynthesizer

### Current Code (arc_program_synthesis.py)

```python
def synthesize(self, task: ARCTask, verbose: bool = False):
    # Phase 3: Infer parameters from training examples
    param_inferrer = ParameterInference()
    inferred_params = param_inferrer.infer_all_parameters(task.train_pairs, verbose)

    all_programs = []

    # Level 0: Primitives (with inferred parameters)
    level_0 = self._generate_primitives_with_params(inferred_params)
    # ... rest of synthesis
```

### Modified Code (with LLM constraints)

```python
def synthesize(self, task: ARCTask, verbose: bool = False, llm_constraints: Dict = None):
    """
    Synthesize programs, optionally guided by LLM semantic understanding

    Args:
        task: ARC task
        verbose: Print debug info
        llm_constraints: Optional semantic constraints from LLM analysis
    """
    # Phase 3: Infer parameters
    param_inferrer = ParameterInference()
    inferred_params = param_inferrer.infer_all_parameters(task.train_pairs, verbose)

    all_programs = []

    # NEW: Generate LLM-guided programs first (if constraints provided)
    if llm_constraints:
        llm_programs = self._generate_llm_guided_programs(llm_constraints, inferred_params)
        all_programs.extend(llm_programs)
        if verbose:
            print(f"Level 0: Generated {len(llm_programs)} LLM-guided programs")

    # Level 0: Primitives (with inferred parameters)
    level_0 = self._generate_primitives_with_params(inferred_params)
    level_0_pruned = self._prune_by_training(level_0, task.train_pairs, keep_top=22)
    all_programs.extend(level_0_pruned)

    # ... rest of synthesis (levels 1-3)
```

### New Method: _generate_llm_guided_programs()

```python
def _generate_llm_guided_programs(self, constraints: Dict, inferred_params: Dict) -> List[Program]:
    """
    Generate programs based on LLM semantic analysis

    This is the KEY innovation: instead of blindly generating all programs,
    we generate targeted programs that match the LLM's understanding.

    Args:
        constraints: Semantic constraints from LLM
        inferred_params: Parameters from parameter inference

    Returns:
        List of targeted programs
    """
    programs = []

    primary_op = constraints.get('primary_operation')
    confidence = constraints.get('confidence', 'medium')

    # High confidence: prioritize LLM-suggested operations
    if confidence == 'high':

        # ZOOM/SCALE operations
        if primary_op == 'zoom':
            scale = constraints.get('scale_factor') or inferred_params.get('scale_factor')
            if scale:
                # Generate the specific zoom operation suggested by LLM
                def zoom_op(g, s=scale):
                    new_data = np.repeat(np.repeat(g.data, s, axis=0), s, axis=1)
                    return Grid(new_data)

                programs.append(Program(
                    f"zoom_{scale}x_llm_guided",
                    zoom_op,
                    {"scale": scale, "source": "llm"}
                ))

                # Also try nearby scale factors
                for nearby_scale in [scale-1, scale+1]:
                    if nearby_scale >= 2:
                        def zoom_nearby(g, s=nearby_scale):
                            new_data = np.repeat(np.repeat(g.data, s, axis=0), s, axis=1)
                            return Grid(new_data)

                        programs.append(Program(
                            f"zoom_{nearby_scale}x_llm_nearby",
                            zoom_nearby,
                            {"scale": nearby_scale, "source": "llm_nearby"}
                        ))

        # ROTATION operations
        elif primary_op == 'rotate':
            for keyword in constraints.get('keywords', []):
                if 'rotate_90' in keyword:
                    programs.append(Program(
                        "rotate_90_llm_guided",
                        lambda g: Grid(np.rot90(g.data, k=1)),
                        {"rotation": 90, "source": "llm"}
                    ))
                elif 'rotate_180' in keyword:
                    programs.append(Program(
                        "rotate_180_llm_guided",
                        lambda g: Grid(np.rot90(g.data, k=2)),
                        {"rotation": 180, "source": "llm"}
                    ))
                elif 'rotate_270' in keyword:
                    programs.append(Program(
                        "rotate_270_llm_guided",
                        lambda g: Grid(np.rot90(g.data, k=3)),
                        {"rotation": 270, "source": "llm"}
                    ))

        # COLOR operations
        elif primary_op == 'recolor':
            color_map = inferred_params.get('color_map')
            if color_map:
                programs.append(Program(
                    "recolor_llm_guided",
                    lambda g, cm=color_map: apply_color_mapping(g, cm),
                    {"color_map": str(color_map), "source": "llm"}
                ))

            # Also try common recoloring patterns
            for target_color in [1, 2, 3, 4]:
                def recolor_all(g, c=target_color):
                    result = g.copy()
                    result.data[result.data > 0] = c
                    return result

                programs.append(Program(
                    f"recolor_nonzero_{target_color}_llm",
                    recolor_all,
                    {"target_color": target_color, "source": "llm"}
                ))

        # OBJECT DETECTION operations
        elif primary_op == 'object_detection':
            # Prioritize object-based operations
            programs.append(Program("keep_largest", keep_largest_op, {"source": "llm"}))
            programs.append(Program("keep_smallest", keep_smallest_op, {"source": "llm"}))

            # Recolor objects by size
            for color in [1, 2, 3, 4]:
                programs.append(Program(
                    f"recolor_largest_{color}_llm",
                    recolor_largest_op,
                    {"color": color, "source": "llm"}
                ))

    # Medium/Low confidence: generate broader set
    else:
        # Fall back to keyword-based generation
        keywords = constraints.get('keywords', [])

        if 'zoom' in keywords or 'scale' in keywords:
            for scale in [2, 3, 4]:
                def zoom_op(g, s=scale):
                    new_data = np.repeat(np.repeat(g.data, s, axis=0), s, axis=1)
                    return Grid(new_data)

                programs.append(Program(
                    f"zoom_{scale}x_llm_keyword",
                    zoom_op,
                    {"scale": scale, "source": "llm_keyword"}
                ))

        if 'rotate' in keywords:
            programs.extend(get_geometric_primitives())

        if 'object' in keywords:
            programs.extend(self._generate_object_programs([])[:10])

    return programs
```

---

## What It Replaces vs Enhances

### REPLACES: Nothing (Hybrid Approach)

The LLM **does not replace** any existing components:
- ✓ Program synthesis still runs (generates all programs as before)
- ✓ Active Inference still evaluates (Bayesian belief updating)
- ✓ Stability filter still assesses robustness
- ✓ Workspace still selects top programs

### ENHANCES: Program Generation (Step 2)

The LLM **enhances** program generation by:

1. **Prioritizing relevant operations**
   - Current: Generate all 100+ programs blindly
   - Enhanced: Generate LLM-suggested programs first (higher priority)

2. **Setting initial beliefs**
   - Current: Uniform prior over all programs
   - Enhanced: LLM-guided programs get higher initial belief

3. **Reducing search space**
   - Current: Try all operations (many irrelevant)
   - Enhanced: Focus on operations matching task semantics

4. **Providing semantic guidance**
   - Current: Syntactic composition only
   - Enhanced: Semantic understanding + syntactic composition

### ADDS: Semantic Understanding (New Step 1.5)

**New capability**: Understanding WHAT the task is asking for, not just trying operations.

---

## Concrete Example: How It Helps

### Task: Zoom 2x (Task 60c09cac)

**Current System (Phase 3)**:
```
1. Generate 119 programs (identity, flip, rotate, zoom_2x, zoom_3x, ...)
2. Evaluate all 119 on training examples
3. Identity scores best (complexity=1, auto-resize handles size)
4. zoom_2x_learned scores worse (complexity=2)
5. Select identity ✓ (happens to work due to auto-resize)
```

**LLM-Enhanced System**:
```
1. LLM analyzes: "This is 2× scaling, high confidence"
2. Generate LLM-guided programs:
   - zoom_2x_llm_guided (priority: HIGH)
   - zoom_3x_llm_nearby (priority: MEDIUM)
   - identity (priority: LOW)
3. Initialize beliefs:
   - zoom_2x_llm_guided: p=0.7 (high confidence)
   - zoom_3x_llm_nearby: p=0.2
   - identity: p=0.1
4. Evaluate on training:
   - zoom_2x_llm_guided: perfect match → p=0.95
   - identity: also matches (resize) → p=0.05
5. Select zoom_2x_llm_guided ✓ (better explanation)
```

**Why it's better**: Even though both work, LLM-guided program selected because it has **higher prior** and **better explanation** of task.

### Task: Recolor Objects by Size (Hypothetical)

**Current System (Phase 3)**:
```
1. Generate 119 programs
2. None include "recolor by object size" (not in DSL)
3. Try generic recoloring, object detection separately
4. Fail to compose "detect size then recolor"
5. Return identity (wrong) ✗
```

**LLM-Enhanced System**:
```
1. LLM analyzes: "Detect objects, recolor based on their size:
   - 3 pixels → blue
   - 5 pixels → red"
2. Generate LLM-guided programs:
   - detect_objects + count_pixels + recolor_conditional
   - (NEW program type based on LLM understanding)
3. Synthesize specific program:
   def recolor_by_size(grid):
       objects = detect_objects(grid)
       for obj in objects:
           size = count_pixels(obj)
           if size == 3:
               obj = recolor(obj, color=1)
           elif size == 5:
               obj = recolor(obj, color=2)
       return compose_objects(objects)
4. Evaluate: perfect match ✓
5. Select recolor_by_size ✓ (solves task)
```

**Why it's better**: LLM identified the specific composition needed ("detect + count + conditional recolor"), which generic synthesis wouldn't discover.

---

## Expected Impact Analysis

### Why 15-30% Success Rate?

**Conservative Estimate (15%)**:
- LLM correctly identifies pattern: 50% of tasks
- System has operations to implement pattern: 60% of identified
- 0.5 × 0.6 = 30 tasks out of 200 = 15%

**Realistic Estimate (25%)**:
- LLM correctly identifies pattern: 70% of tasks
- System can implement: 70% of identified
- 0.7 × 0.7 ≈ 50 tasks = 25%

**Optimistic Estimate (30%)**:
- LLM correctly identifies: 80%
- System implements: 75%
- 0.8 × 0.75 = 60 tasks = 30%

### What Types of Tasks Would Benefit?

**High Benefit** (20-30 tasks):
1. **Scaling/Zooming** (10 tasks)
   - LLM identifies scale factor
   - Current: 2/200 solved
   - With LLM: 10/200 (+8 tasks)

2. **Object-based transformations** (10 tasks)
   - LLM identifies "detect objects + transform each"
   - Current: 0/200 solved
   - With LLM: 10/200 (+10 tasks)

3. **Conditional recoloring** (5-10 tasks)
   - LLM identifies "if color X then Y"
   - Current: 0/200 solved
   - With LLM: 5-10/200 (+5-10 tasks)

**Medium Benefit** (10-20 tasks):
4. **Pattern detection and extension** (5-10 tasks)
5. **Spatial transformations** (5-10 tasks)

**Low Benefit** (0-5 tasks):
6. **Complex multi-step reasoning** (still too hard)
7. **Novel operations** (not in DSL)

---

## Implementation Complexity

### Simple Version (1 day)

**Minimal LLM integration**:
```python
# 1. Create llm_analyzer.py (~200 lines)
# 2. Modify arc_program_solver.py (+10 lines)
# 3. Modify arc_program_synthesis.py (+50 lines)
# Total: ~260 lines
```

### Full Version (2-3 days)

**Complete implementation**:
```python
# 1. llm_analyzer.py (~400 lines)
#    - Task serialization
#    - Prompt engineering
#    - Response parsing
#    - Caching
#    - Error handling
#
# 2. arc_program_synthesis.py (+200 lines)
#    - LLM-guided program generation
#    - Priority-based synthesis
#    - Confidence-weighted priors
#
# 3. arc_program_solver.py (+50 lines)
#    - LLM integration
#    - Prior initialization from LLM
#
# 4. test_llm_integration.py (~300 lines)
#    - Evaluation script
#    - Comparison with Phase 3
#
# Total: ~950 lines
```

---

## Comparison: What Each Component Does

| Component | Current Role | With LLM Enhancement |
|-----------|-------------|---------------------|
| **Perception** | Extract basic features | Same (minimal change) |
| **LLM Analyzer** | N/A | **NEW**: Analyze task semantics |
| **Program Synthesis** | Generate all programs blindly | Generate LLM-guided programs first |
| **Active Inference** | Uniform prior over programs | **LLM-weighted prior** (high for LLM-suggested) |
| **Stability Filter** | Assess robustness | Same |
| **Workspace** | Select top-20 programs | Same |
| **Ranking** | posterior × stability | Same |
| **Selection** | Diverse top-2 | Same |
| **Execution** | Apply to test input | Same |

**Key Change**: Steps 1.5, 2, and 3 are enhanced; Steps 4-9 unchanged.

---

## Risks and Mitigations

### Risk 1: LLM Hallucination

**Problem**: LLM might confidently suggest wrong operation

**Mitigation**:
- Active Inference still evaluates all programs on training data
- Wrong suggestions get filtered out by low posterior probability
- LLM provides **guidance, not final decision**

### Risk 2: API Costs

**Problem**: Claude API costs ~$0.01 per task analysis

**Mitigation**:
- Cache results (same task analyzed once)
- Use cheaper model (Haiku) for simple tasks
- Cost for 200 tasks: ~$2 (acceptable)

### Risk 3: Latency

**Problem**: LLM call adds 1-2 seconds per task

**Mitigation**:
- Parallel processing (analyze multiple tasks concurrently)
- Caching (repeated evaluations fast)
- Overall: 200 tasks in ~5 minutes (vs 30 seconds now)

### Risk 4: LLM Fails to Parse Task

**Problem**: LLM can't understand visual patterns

**Mitigation**:
- Fallback to baseline synthesis if LLM returns low confidence
- System works without LLM (graceful degradation)
- **Hybrid approach**: LLM + baseline always available

---

## Success Criteria

### Quantitative Metrics

1. **Success Rate**: 1% → 15-30%
   - Baseline: 2/200 tasks
   - Target: 30-60/200 tasks

2. **LLM Accuracy**: 70%+ correct pattern identification
   - Measure: LLM suggestion matches ground truth
   - Method: Manual inspection of 50 tasks

3. **Priority Effectiveness**: LLM-guided programs in top-10 for 60%+ of tasks
   - Measure: Fraction where LLM program ranks high
   - Method: Check final rankings

### Qualitative Indicators

1. **Solved tasks are semantically correct**
   - Not just lucky matches
   - Programs reflect actual task intent

2. **Failures are informative**
   - LLM correctly identifies pattern
   - System lacks operation to implement
   - → Clear direction for DSL expansion

---

## Next Steps

### Phase 4a: Minimal LLM Integration (1 day)

1. Create `llm_analyzer.py` with basic analysis
2. Modify synthesis to use LLM keywords
3. Test on 10 hand-picked tasks
4. Validate LLM suggestions are reasonable

### Phase 4b: Full Integration (2 days)

1. Implement LLM-guided program generation
2. Add priority-based belief initialization
3. Full evaluation on 200 tasks
4. Analysis and iteration

### Phase 4c: Refinement (ongoing)

1. Improve prompts based on failures
2. Add few-shot examples to LLM
3. Expand DSL based on LLM suggestions
4. Iterate toward 30%+ success rate

---

## Conclusion

### What LLM Does

**Core Function**: Provides **semantic understanding** of what the task is asking for.

**Specific Roles**:
1. Analyze training examples
2. Identify transformation pattern
3. Extract parameters (scale, colors, etc.)
4. Suggest relevant operations
5. Estimate confidence

### What LLM Does NOT Do

**Does NOT**:
- Replace program synthesis (still needed for verified execution)
- Replace active inference (still needed for evaluation)
- Replace any existing components (pure enhancement)
- Generate code directly (synthesis does this)

### Why This Works

**Complementary Strengths**:
- **LLM**: Semantic understanding, pattern recognition
- **Synthesis**: Verified execution, compositional reasoning
- **Active Inference**: Bayesian evaluation, robustness

**Together**: Understanding (LLM) + Implementation (Synthesis) + Verification (Active Inference) = Robust solver

### Bottom Line

**Integration Type**: **Hybrid LLM-Guided Synthesis**

**Architecture Change**: Add Step 1.5 (LLM Analysis), enhance Step 2 (Synthesis)

**Expected Impact**: 1% → 15-30% success rate

**Implementation Effort**: 2-3 days

**Key Innovation**: Semantic understanding guides syntactic composition

---

*Ready to implement? Let me know and I'll start with Phase 4a (minimal integration)!*
