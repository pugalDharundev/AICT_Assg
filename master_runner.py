import os
import subprocess
import sys
from typing import Dict, Tuple


TASKS: Dict[str, Tuple[str, str]] = {
    "1": ("routing.py", "Run search algorithms (BFS/DFS/GBFS/A*) and export results"),
    "2": ("inference.py", "Validate advisory logic with resolution prover"),
    "3": ("run_bn_scenarios.py", "Execute Bayesian crowding scenarios"),
    "4": ("crowding_bn.py", "Query standalone Bayesian Network example"),
    "5": ("bonus_optimization.py", "Run disruption optimization (hill climbing + simulated annealing)"),
}


def print_menu() -> None:
    print("\n=== ChangiLink AI Master Runner ===")
    for key, (script, description) in sorted(TASKS.items()):
        print(f"  {key}. {description} ({script})")
    print("  q. Quit")


def run_task(script: str) -> None:
    script_path = os.path.join(os.path.dirname(__file__), script)
    if not os.path.isfile(script_path):
        print(f"[ERR] Cannot find {script}")
        return

    print(f"\n[RUN] python {script}\n")
    try:
        result = subprocess.run([sys.executable, script_path], check=False)
        print(f"\n[EXIT] {script} finished with code {result.returncode}\n")
    except KeyboardInterrupt:
        print("\n[ABORT] Run canceled by user\n")
    except Exception as exc:
        print(f"\n[ERR] Failed to run {script}: {exc}\n")


def main() -> None:
    while True:
        print_menu()
        choice = input("Select an option: ").strip().lower()
        if choice == "q":
            print("Exiting master runner. Goodbye!")
            return
        if choice in TASKS:
            script, _ = TASKS[choice]
            run_task(script)
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
