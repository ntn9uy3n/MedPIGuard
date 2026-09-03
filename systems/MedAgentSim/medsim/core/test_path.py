import os
from pathlib import Path


def resolve_dataset_path(input_path: str, dataset: str) -> Path:
    input_path = Path(input_path).resolve()
    parts = input_path.parts

    if "MedAgentSim" not in parts:
        raise ValueError("Path does not contain 'MedAgentSim'")

    # Get everything up to and including MedAgentSim
    medagent_index = parts.index("MedAgentSim")
    project_root = Path(*parts[:medagent_index + 1])

    # Append datasets/_medqa.jsonl
    return project_root / "datasets" / dataset

# Detect runtime cwd (caller location), and assume it's project root

# project_root = os.path.abspath(os.path.join(os.getcwd()))
# data_path = os.path.join(project_root, "datasets", "_medqa.jsonl")
# print(data_path)

# current_file_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
# data_path = os.path.join(project_root, "datasets", "_medqa.jsonl")
# print(data_path)

def run_me():
    project_root = os.path.abspath(os.path.join(os.getcwd()))
    data_path = os.path.join(project_root, "datasets", "_medqa.jsonl")
    print(data_path)

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    data_path = os.path.join(project_root, "datasets", "_medqa.jsonl")
    print(data_path)

    project_root = Path(os.getcwd())

    # Optional sanity check
    # expected_file = project_root / "datasets" / "_medqa.jsonl"
    print(project_root)
    project_root = Path(os.getcwd())
    data_path = resolve_dataset_path(project_root, "_medqa.jsonl")
    print(data_path)

run_me()