#!/usr/bin/env python3
"""
Composable CLI for evolutionary code optimization.

Two ways to use:

1. `cevolve run` - Primary CLI for humans
   Full workflow with built-in LLM, TUI, automatic rethink.

2. Composable commands - For extensions (like pi-evolve)
   init / next / eval / record / revert / rethink / status / stop / sessions
   LLM-agnostic building blocks.

Examples:
    # Human workflow
    cevolve run --target train.py --metric val_bpb --llm claude
    cevolve run --scope "src/**/*.py" --metric time_ms --llm pi --no-tui

    # Extension workflow
    cevolve init --name my-opt --ideas ideas.json --bench ./bench.sh
    cevolve next --json
    # (extension implements genes)
    cevolve eval --id ind-abc123
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="cevolve",
        description="Evolutionary code optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true",
                        help="JSON output (for extensions)")
    parser.add_argument("--work-dir", default=".",
                        help="Working directory")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # =========================================================================
    # run - Primary CLI for humans
    # =========================================================================
    p_run = subparsers.add_parser("run",
        help="Run evolution with built-in LLM (primary CLI)")
    p_run.add_argument("--name", help="Session name")
    p_run.add_argument("--target", action="append", dest="targets",
                       help="Target file(s) to optimize (can specify multiple)")
    # Supports multiple files: --target train.py --target prompts/system.md
    p_run.add_argument("--bench", help="Benchmark command")
    p_run.add_argument("--metric", default="val_bpb",
                       help="Metric to optimize")
    p_run.add_argument("--direction", choices=["lower", "higher"], default="lower")
    p_run.add_argument("--llm", choices=["claude", "pi"], default="claude",
                       help="LLM CLI to use")
    p_run.add_argument("--pop-size", type=int, default=6)
    p_run.add_argument("--max-evals", type=int, default=20)
    p_run.add_argument("--rethink", type=int, default=5,
                       help="Rethink every N evals (0 to disable)")
    p_run.add_argument("--no-tui", action="store_true",
                       help="Disable TUI")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Mock LLM and training")
    
    # =========================================================================
    # init - Create session
    # =========================================================================
    p_init = subparsers.add_parser("init", help="Create session with ideas")
    p_init.add_argument("--name", required=True, help="Session name")
    p_init.add_argument("--ideas",
                        help="Ideas as JSON file or inline JSON array")
    p_init.add_argument("--idea", action="append", dest="inline_ideas",
                        help="Inline idea: 'name: desc' or 'name[v1,v2]: desc'")
    p_init.add_argument("--bench", required=True,
                        help="Benchmark command")
    p_init.add_argument("--metric", default="time_ms")
    p_init.add_argument("--direction", choices=["lower", "higher"], default="lower")
    p_init.add_argument("--scope", action="append",
                        help="File patterns to include")
    p_init.add_argument("--exclude", action="append",
                        help="File patterns to exclude")
    p_init.add_argument("--target", help="Single target file")
    p_init.add_argument("--pop-size", type=int, default=6)
    p_init.add_argument("--elitism", type=int, default=2)
    p_init.add_argument("--mutation-rate", type=float, default=0.2)
    p_init.add_argument("--crossover-rate", type=float, default=0.7)
    p_init.add_argument("--max-evals", type=int)
    p_init.add_argument("--convergence-evals", type=int)
    p_init.add_argument("--rethink-interval", type=int, default=5)
    p_init.add_argument("--timeout", type=int, default=600)
    p_init.add_argument("--revert", choices=["git", "stash", "cache", "single"],
                        default="git")
    p_init.add_argument("--secondary-metric", action="append",
                        dest="secondary_metrics")
    
    # =========================================================================
    # next - Get next individual
    # =========================================================================
    p_next = subparsers.add_parser("next",
        help="Get next individual to evaluate")
    p_next.add_argument("--session", help="Session name")
    
    # =========================================================================
    # eval - Run benchmark, record, revert
    # =========================================================================
    p_eval = subparsers.add_parser("eval",
        help="Run benchmark, record result, revert")
    p_eval.add_argument("--session", help="Session name")
    p_eval.add_argument("--id", required=True, help="Individual ID")
    p_eval.add_argument("--timeout", type=int)
    
    # =========================================================================
    # record - Just record result
    # =========================================================================
    p_record = subparsers.add_parser("record",
        help="Record result (agent ran benchmark)")
    p_record.add_argument("--session", help="Session name")
    p_record.add_argument("--id", required=True, help="Individual ID")
    p_record.add_argument("--fitness", type=float,
                          help="Fitness value")
    p_record.add_argument("--metrics",
                          help="Additional metrics as JSON")
    p_record.add_argument("--failed", action="store_true",
                          help="Mark as failed")
    p_record.add_argument("--error", help="Error message")
    
    # =========================================================================
    # revert - Just revert files
    # =========================================================================
    p_revert = subparsers.add_parser("revert",
        help="Revert file changes")
    p_revert.add_argument("--session", help="Session name")
    
    # =========================================================================
    # rethink - Analyze and modify
    # =========================================================================
    p_rethink = subparsers.add_parser("rethink",
        help="Analyze results, modify ideas")
    p_rethink.add_argument("--session", help="Session name")
    p_rethink.add_argument("--add-idea", action="append", dest="add_ideas",
                           help="Add new idea")
    p_rethink.add_argument("--remove-idea", action="append", dest="remove_ideas",
                           help="Remove idea by name")
    p_rethink.add_argument("--commit-best", action="store_true",
                           help="Commit best as new baseline")
    
    # =========================================================================
    # status - Get state
    # =========================================================================
    p_status = subparsers.add_parser("status", help="Get session state")
    p_status.add_argument("--session", help="Session name")
    p_status.add_argument("-v", "--verbose", action="store_true")
    
    # =========================================================================
    # stop - Finalize
    # =========================================================================
    p_stop = subparsers.add_parser("stop", help="Finalize session")
    p_stop.add_argument("--session", help="Session name")
    p_stop.add_argument("--cleanup", action="store_true",
                        help="Remove session data")
    
    # =========================================================================
    # sessions - List/switch
    # =========================================================================
    p_sessions = subparsers.add_parser("sessions",
        help="List or switch sessions")
    p_sessions.add_argument("--switch", help="Switch to session")
    
    # =========================================================================
    # guide - Workflow guide
    # =========================================================================
    p_guide = subparsers.add_parser("guide",
        help="Print workflow guide for coding agents")
    
    args = parser.parse_args()
    
    # Import handlers
    from evolve.commands import (
        init, next, eval, record, revert, rethink, status, stop, sessions, run, guide
    )
    
    handlers = {
        "run": run.handle,
        "init": init.handle,
        "next": next.handle,
        "eval": eval.handle,
        "record": record.handle,
        "revert": revert.handle,
        "rethink": rethink.handle,
        "status": status.handle,
        "stop": stop.handle,
        "sessions": sessions.handle,
        "guide": guide.handle,
    }
    
    try:
        result = handlers[args.command](args)
        _output(args, result)
    except KeyboardInterrupt:
        if args.json:
            print(json.dumps({"error": "interrupted"}))
        else:
            print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _output(args, result: dict):
    """Output result."""
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return
    
    cmd = args.command
    
    if cmd == "run":
        # run handles its own output
        pass
    
    elif cmd == "guide":
        # guide handles its own output (prints directly)
        pass
    
    elif cmd == "init":
        print(f"✓ Session '{result['session']}' created")
        print(f"  Ideas: {result['ideas']}")
        print(f"  Search space: {result['search_space']:,} combinations")
        print(f"\nNext: cevolve next")
    
    elif cmd == "next":
        if result.get("status") in ("converged", "max_evals"):
            print(f"🎯 {result['status'].upper()}")
            if result.get("best"):
                print(f"   Best: {result['best']['fitness']} ({result['best']['improvement']})")
                print(f"   Config: {_fmt_genes(result['best']['genes'])}")
            print(f"\n{result.get('message', '')}")
        elif result.get("status") == "rethink_required":
            print(f"⏸️  RETHINK REQUIRED")
            print(f"\n{result.get('message', '')}")
            if result.get("best"):
                print(f"\nBest config to commit:")
                for g in result['best'].get('genes_to_implement', []):
                    val = f"= {g['value']}" if g['value'] != 'on' else ""
                    print(f"  • {g['name']} {val}")
                    print(f"    {g['description']}")
            print(f"\n{result.get('instructions', '')}")
        else:
            print(f"Individual: {result['individual_id']}")
            print(f"Generation: {result['generation']}")
            if result.get("is_baseline"):
                print("\n  (baseline - no changes needed)")
            else:
                print("\n  Active genes:")
                for g in result.get("active", []):
                    val = f"= {g['value']}" if g['value'] != 'on' else ""
                    print(f"    • {g['name']} {val}")
                    print(f"      {g['description']}")
            print(f"\nAfter implementing: cevolve eval --id {result['individual_id']}")
            print(f"Then revert: git checkout .")
    
    elif cmd == "eval":
        if result.get("fitness") is not None:
            icon = "🏆" if result.get("is_best") else "✓"
            print(f"{icon} Fitness: {result['fitness']}")
            if result.get("is_best"):
                print(f"  NEW BEST! {result.get('improvement', '')}")
        else:
            print(f"✗ Failed: {result.get('error', 'unknown')}")
        print(f"\nEvaluations: {result['evaluations']}")
        print(f"Status: {result['status']}")
        if result['status'] == 'continue':
            print(f"\nNext: cevolve next")
    
    elif cmd == "record":
        if result.get("fitness") is not None:
            print(f"✓ Recorded fitness: {result['fitness']}")
        else:
            print(f"✓ Recorded failure")
        print(f"\n{result.get('note', '')}")
    
    elif cmd == "revert":
        print(f"✓ Reverted")
    
    elif cmd == "rethink":
        print(f"=== Rethink Analysis ===")
        print(f"Evaluations: {result['evaluations']}")
        print(f"Best: {result['best_fitness']} ({result['improvement']})")
        print(f"\nIdea effectiveness:")
        for name, stats in result.get("ideas", {}).items():
            rate = f"{stats['success_rate']:.0%}" if stats['eval_count'] > 0 else "N/A"
            print(f"  • {name}: {stats['eval_count']} evals, {rate} success")
        if result.get("added"):
            print(f"\nAdded: {', '.join(result['added'])}")
        if result.get("removed"):
            print(f"Removed: {', '.join(result['removed'])}")
    
    elif cmd == "status":
        print(f"Session: {result['session']}")
        print(f"Evaluations: {result['evaluations']}")
        print(f"Generation: {result['generation']} | Era: {result['era']}")
        baseline = result.get("baseline_fitness")
        if baseline is not None:
            print(f"Baseline: {baseline}")
        if result.get("best"):
            improvement = result.get('improvement', '')
            if improvement:
                print(f"\nBest: {result['best']['fitness']} ({improvement})")
            else:
                print(f"\nBest: {result['best']['fitness']}")
            print(f"  {_fmt_genes(result['best']['genes'])}")
        print(f"\nConverged: {result['converged']}")
        print(f"Evals since improvement: {result['evals_since_improvement']}")
    
    elif cmd == "stop":
        print(f"✓ Session '{result['session']}' finalized")
        if result.get("best"):
            baseline = result.get('baseline_fitness')
            improvement = result['best'].get('improvement', '')
            if baseline is not None:
                print(f"\nBaseline: {baseline}")
                print(f"Best: {result['best']['fitness']} ({improvement})")
            else:
                print(f"\nBest: {result['best']['fitness']}")
            print(f"Config: {_fmt_genes(result['best']['genes'])}")
        print(f"\nResults: {result['results_dir']}")
    
    elif cmd == "sessions":
        if result.get("switched"):
            print(f"✓ Switched to: {result['switched']}")
        else:
            print("Sessions:")
            for s in result.get("sessions", []):
                marker = "* " if s.get("current") else "  "
                print(f"{marker}{s['name']} ({s['evaluations']} evals)")


def _fmt_genes(genes: dict) -> str:
    """Format genes dict."""
    if not genes:
        return "baseline"
    return ", ".join(f"{k}={v}" if v != "on" else k for k, v in sorted(genes.items()))


if __name__ == "__main__":
    main()
