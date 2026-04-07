"""
Synthetic data generation for 7 reasoning hop generalization tasks.
All tasks follow the paper's formulation with controlled hop counts.
"""
import json
import random
import re
from typing import List, Dict, Any, Tuple


# --- Name pools for resynthesis (paper Section D.1) ---
FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nina", "Oscar", "Paula",
    "Quinn", "Rachel", "Samuel", "Tina", "Ulysses", "Vera", "Wesley", "Xena",
    "Yuri", "Zoe", "Jack", "Emma", "Liam", "Olivia", "Noah", "Ava",
    "William", "Isabella", "James", "Sophia", "Benjamin", "Mia", "Lucas",
    "Charlotte", "Henry", "Amelia", "Alexander", "Harper", "Mason", "Evelyn",
]

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


# ============================================================
# TASK 1: Parity-NL (Coin Flip Parity)
# ============================================================
def generate_parity_nl(hop_count: int, seed: int = None) -> Dict[str, Any]:
    """Generate a Parity-NL (coin flip parity) instance.

    Args:
        hop_count: Number of flip actions (reasoning hops)
        seed: Random seed for reproducibility

    Returns:
        Dict with 'input', 'answer', 'hop_count', 'task'
    """
    rng = random.Random(seed)

    # Initial state: always heads up per the paper's template
    initial_state = "heads up"

    # Generate random flips (0 = doesn't flip, 1 = flips)
    flips = [rng.randint(0, 1) for _ in range(hop_count)]

    # Count how many flips result in tails
    tail_count = sum(flips)

    # Final answer: heads up if even number of tails, tails up if odd
    final_state = "heads up" if tail_count % 2 == 0 else "tails up"

    # Generate person names for each action
    names = rng.sample(FIRST_NAMES, hop_count)

    # Build the problem text
    actions = []
    for i, (name, flip) in enumerate(zip(names, flips)):
        action = f"flips" if flip == 1 else "doesn't flip"
        actions.append(f"Then {name} {action}.")

    problem = f"Initially the coin is {initial_state}. " + " ".join(actions) + " Finally, is the coin heads up or tails up?"

    return {
        "task": "parity_nl",
        "hop_count": hop_count,
        "input": problem,
        "answer": final_state,
        "flip_sequence": flips,
        "seed": seed,
    }


# ============================================================
# TASK 2: LLC (Last Letter Concatenation)
# ============================================================
def generate_llc(word_count: int, seed: int = None) -> Dict[str, Any]:
    """Generate an LLC (Last Letter Concatenation) instance.

    Args:
        word_count: Number of words in the list
        seed: Random seed

    Returns:
        Dict with 'input', 'answer', 'hop_count', 'task'
    """
    rng = random.Random(seed)

    # Generate random words (simple common words)
    word_pool = [
        "apple", "banana", "cherry", "dragon", "eagle", "forest", "garden",
        "harbor", "island", "jungle", "kitchen", "lemon", "mountain", "nature",
        "ocean", "planet", "quartz", "river", "shadow", "temple", "umbrella",
        "valley", "window", "yellow", "zebra", "bright", "castle", "desert",
        "energy", "flower", "glacier", "health", "internet", "journey", "kernel",
    ]
    words = rng.sample(word_pool, word_count)

    # Extract last letters and concatenate
    last_letters = [w[-1] for w in words]
    answer = "".join(last_letters)

    # Build the problem
    word_list_str = ", ".join(words)
    problem = f"Take the last letter of each of the following words and concatenate them: {word_list_str}."

    return {
        "task": "llc",
        "hop_count": word_count,
        "input": problem,
        "answer": answer,
        "words": words,
        "seed": seed,
    }


# ============================================================
# TASK 3: MDM (Multi-Digit Multiplication)
# ============================================================
def generate_mdm(digits_a: int, digits_b: int, seed: int = None) -> Dict[str, Any]:
    """Generate an MDM (Multi-Digit Multiplication) instance.

    Args:
        digits_a: Number of digits in first operand
        digits_b: Number of digits in second operand
        seed: Random seed

    Returns:
        Dict with 'input', 'answer', 'hop_count', 'task'
    """
    rng = random.Random(seed)

    # Generate random numbers with specified digits
    min_a = 10 ** (digits_a - 1)
    max_a = 10 ** digits_a - 1
    min_b = 10 ** (digits_b - 1)
    max_b = 10 ** digits_b - 1

    a = rng.randint(min_a, max_a)
    b = rng.randint(min_b, max_b)
    answer = a * b

    problem = f"{a} * {b} = ? Please think step by step."

    # Approximate hop count as digit count
    hop_count = digits_b

    return {
        "task": "mdm",
        "hop_count": hop_count,
        "digits_a": digits_a,
        "digits_b": digits_b,
        "input": problem,
        "answer": str(answer),
        "operand_a": a,
        "operand_b": b,
        "seed": seed,
    }


# ============================================================
# TASK 4: MOAS (Multi-Operand Addition and Subtraction)
# ============================================================
def generate_moas(operand_count: int, max_digit: int = 2, seed: int = None) -> Dict[str, Any]:
    """Generate an MOAS (Multi-Operand Addition and Subtraction) instance.

    Args:
        operand_count: Number of operands (reasoning hops)
        max_digit: Maximum digit length of each operand
        seed: Random seed

    Returns:
        Dict with 'input', 'answer', 'hop_count', 'task'
    """
    rng = random.Random(seed)

    # Generate alternating + and - operations
    operands = []
    operations = []
    running = rng.randint(1, 9)  # Start with a small positive number

    for i in range(operand_count):
        op_val = rng.randint(1, 99)
        is_add = rng.choice([True, False])

        operands.append(op_val)
        operations.append("+" if is_add else "-")

        if is_add:
            running += op_val
        else:
            running -= op_val

    answer = running

    # Build the problem string
    expr_parts = []
    for i, (op, num) in enumerate(zip(operations, operands)):
        expr_parts.append(f"{num}")
        if i < len(operations) - 1:
            expr_parts.append(op)

    problem = f"Calculate: " + " ".join(expr_parts) + f" = ?"

    return {
        "task": "moas",
        "hop_count": operand_count,
        "input": problem,
        "answer": str(answer),
        "operands": operands,
        "operations": operations,
        "seed": seed,
    }


# ============================================================
# TASK 5: CLF (Crawler-Log-Folder)
# ============================================================
def generate_clf(seq_length: int, seed: int = None) -> Dict[str, Any]:
    """Generate a CLF (Crawler-Log-Folder) instance.

    Each log entry: "./" = stay, "../" = go up, "x/" = go down into folder x
    Goal: find final depth relative to root.
    """
    rng = random.Random(seed)

    # Generate operations
    operations = []
    current_depth = 0

    for _ in range(seq_length):
        op_type = rng.randint(0, 2)
        if op_type == 0:  # Stay
            operations.append("./")
        elif op_type == 1:  # Go up
            if current_depth > 0:
                operations.append("../")
                current_depth -= 1
            else:
                operations.append("./")
        else:  # Go down
            folder = rng.choice(["a", "b", "c", "d", "x", "y", "z"])
            operations.append(f"{folder}/")
            current_depth += 1

    answer = current_depth
    ops_string = " ".join(operations)
    problem = f"Given a sequence of folder operations: {ops_string}. Starting from the root folder, what is the final depth relative to the root?"

    return {
        "task": "clf",
        "hop_count": seq_length,
        "input": problem,
        "answer": str(answer),
        "operations": operations,
        "seed": seed,
    }


# ============================================================
# TASK 6: ObjC (Object Counting)
# ============================================================
def generate_objc(object_count: int, seed: int = None) -> Dict[str, Any]:
    """Generate an ObjC (Object Counting) instance from BBH.

    Count objects satisfying a condition.
    """
    rng = random.Random(seed)

    # Generate objects with attributes
    colors = ["red", "blue", "green", "yellow", "purple"]
    shapes = ["circle", "square", "triangle"]
    sizes = ["small", "large"]

    objects = []
    for i in range(object_count):
        obj = {
            "name": rng.choice(FIRST_NAMES),
            "color": rng.choice(colors),
            "shape": rng.choice(shapes),
            "size": rng.choice(sizes),
            "position": i + 1,
        }
        objects.append(obj)

    # Pick a condition: count objects with a specific property
    # Keep it simple: count objects of a specific color that are also large
    target_color = rng.choice(colors)
    count = sum(1 for o in objects if o["color"] == target_color and o["size"] == "large")

    # Build description
    obj_descriptions = [f"{o['name']} has a {o['size']} {o['color']} {o['shape']} (position {o['position']})"
                        for o in objects]
    desc_str = " ".join(obj_descriptions)

    problem = (f"Consider the following objects: {desc_str}. "
               f"Count how many objects are {target_color} and large. "
               f"Return just the number.")

    return {
        "task": "objc",
        "hop_count": object_count,
        "input": problem,
        "answer": str(count),
        "objects": objects,
        "target_color": target_color,
        "seed": seed,
    }


# ============================================================
# TASK 7: NumS (Number of Students doing homework)
# ============================================================
def generate_nums(student_count: int, seed: int = None) -> Dict[str, Any]:
    """Generate a NumS (Number of Students doing homework at a given time) instance.

    Given start and end times of multiple students' homework sessions and a query time,
    count how many students are doing homework at the query time.
    """
    rng = random.Random(seed)

    # Generate start times (hour of day, 0-23)
    start_times = sorted([rng.randint(8, 20) for _ in range(student_count)])
    # Generate end times (must be after start)
    end_times = [rng.randint(s + 1, 23) for s in start_times]

    # Query time
    query_hour = rng.randint(10, 22)

    # Count students doing homework at query time
    count = sum(1 for s, e in zip(start_times, end_times) if s <= query_hour <= e)

    # Build problem
    sessions = [f"Student {i+1}: {s}:00 to {e}:00" for i, (s, e) in enumerate(zip(start_times, end_times))]
    problem = (f"Students are doing homework at these times: {', '.join(sessions)}. "
               f"At {query_hour}:00, how many students are doing homework? "
               f"Return just the number.")

    return {
        "task": "nums",
        "hop_count": student_count,
        "input": problem,
        "answer": str(count),
        "start_times": start_times,
        "end_times": end_times,
        "query_hour": query_hour,
        "seed": seed,
    }


# ============================================================
# Dataset Generation
# ============================================================
def generate_dataset(task: str, count: int, **kwargs) -> List[Dict[str, Any]]:
    """Generate a dataset for a specific task.

    Args:
        task: Task name ('parity_nl', 'llc', 'mdm', 'moas', 'clf', 'objc', 'nums')
        count: Number of instances to generate
        **kwargs: Task-specific parameters (e.g., hop_count for parity_nl)

    Returns:
        List of instances
    """
    generators = {
        "parity_nl": generate_parity_nl,
        "llc": generate_llc,
        "mdm": generate_mdm,
        "moas": generate_moas,
        "clf": generate_clf,
        "objc": generate_objc,
        "nums": generate_nums,
    }

    gen = generators[task]
    instances = []
    for i in range(count):
        _seed = kwargs.get("seed", None)
        if _seed is not None:
            _seed = _seed + i
        gen_kwargs = {k: v for k, v in kwargs.items() if k != "seed"}
        inst = gen(seed=_seed, **gen_kwargs)
        instances.append(inst)

    return instances


def save_dataset(instances: List[Dict], path: str):
    """Save dataset to JSONL file."""
    with open(path, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")


def load_dataset(path: str) -> List[Dict]:
    """Load dataset from JSONL file."""
    instances = []
    with open(path) as f:
        for line in f:
            instances.append(json.loads(line))
    return instances


# ============================================================
# Solution Templates (for in-context demonstrations)
# ============================================================
PARITY_NL_TEMPLATE = """Example:

Problem: Initially the coin is heads up. Then Alice flips. Then Bob doesn't flip. Finally, is the coin heads up or tails up?
Answer: Let's trace through step by step:
- Initial: coin is heads up (even = 0 flips so far)
- Step 1: Alice flips → coin becomes tails up (odd = 1 flip so far)
- Step 2: Bob doesn't flip → coin stays tails up (odd = 1 flip so far)
The coin is tails up.

Problem: {input}
Answer:"""

LLC_TEMPLATE = """Example:

Problem: Take the last letter of each of the following words and concatenate them: apple, banana, cherry.
Answer: Let's extract the last letter of each word:
- apple → 'e'
- banana → 'a'
- cherry → 'y'
Concatenating: e + a + y = "eay"

Problem: {input}
Answer:"""

MDM_TEMPLATE = """Example:

Problem: 23 * 45 = ? Please think step by step.
Answer: Let's multiply 23 by 45 step by step:
- Multiply 23 by 5 (units): 23 * 5 = 115
- Multiply 23 by 4 (tens): 23 * 40 = 920
- Add: 115 + 920 = 1035
So 23 * 45 = 1035.

Problem: {input}
Answer:"""

MOAS_TEMPLATE = """Example:

Problem: Calculate: 10 + 5 - 3 + 8 - 2 = ?
Answer: Let's calculate step by step:
- Start with 10
- Add 5: 10 + 5 = 15
- Subtract 3: 15 - 3 = 12
- Add 8: 12 + 8 = 20
- Subtract 2: 20 - 2 = 18
So the result is 18.

Problem: {input}
Answer:"""

CLF_TEMPLATE = """Example:

Problem: Given a sequence of folder operations: ./ a/ b/ ../ c/ ../. Starting from the root folder, what is the final depth relative to the root?
Answer: Let's trace through step by step:
- Start: at root (depth 0)
- ./ : stay at root (depth 0)
- a/ : go into folder a (depth 1)
- b/ : go into folder b (depth 2)
- ../ : go up one level (depth 1)
- c/ : go into folder c (depth 2)
- ../ : go up one level (depth 1)
Final depth: 1

Problem: {input}
Answer:"""

NUM_TEMPLATE = """Example:

Problem: Consider 3 objects: Alex has a small red circle (position 1), Bella has a large blue square (position 2), Chris has a small red triangle (position 3). Count how many objects are red. Return just the number.
Answer: Let's identify red objects:
- Position 1: small red circle → red ✓
- Position 2: large blue square → not red ✗
- Position 3: small red triangle → red ✓
Count: 2

Problem: {input}
Answer:"""


def get_prompt_with_template(instance: Dict[str, Any]) -> str:
    """Add the appropriate CoT template to an instance for generation."""
    task = instance["task"]
    input_text = instance["input"]

    templates = {
        "parity_nl": PARITY_NL_TEMPLATE,
        "llc": LLC_TEMPLATE,
        "mdm": MDM_TEMPLATE,
        "moas": MOAS_TEMPLATE,
        "clf": CLF_TEMPLATE,
        "objc": NUM_TEMPLATE,  # ObjC uses similar counting format
        "nums": NUM_TEMPLATE,
    }

    template = templates.get(task, "")
    if template:
        return template.format(input=input_text)
    return input_text


if __name__ == "__main__":
    # Generate sample data for each task
    import os
    os.makedirs("/home/user/data", exist_ok=True)

    # Generate 10 samples per task for quick testing
    tasks_config = [
        ("parity_nl", {"hop_count": 10}),
        ("parity_nl", {"hop_count": 30}),
        ("parity_nl", {"hop_count": 50}),
        ("llc", {"word_count": 4}),
        ("llc", {"word_count": 6}),
        ("mdm", {"digits_a": 3, "digits_b": 4}),
        ("mdm", {"digits_a": 3, "digits_b": 6}),
        ("moas", {"operand_count": 20}),
        ("moas", {"operand_count": 50}),
        ("clf", {"seq_length": 20}),
        ("clf", {"seq_length": 30}),
        ("objc", {"object_count": 20}),
        ("objc", {"object_count": 30}),
        ("nums", {"student_count": 10}),
        ("nums", {"student_count": 20}),
    ]

    for task_name, params in tasks_config:
        count = 5  # Small test set
        instances = generate_dataset(task_name, count, **params)
        filename = f"{task_name}_{params}"
        filename = filename.replace(" ", "").replace("'", "").replace("{", "_").replace("}", "").replace(":", "_")
        path = f"/home/user/data/test_{filename}.jsonl"
        save_dataset(instances, path)
        print(f"Generated {len(instances)} instances for {task_name} with params {params} -> {path}")
