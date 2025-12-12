#!/usr/bin/env python3
"""
Script to automatically count and update problem statistics in README.md
Run this script to update the problem counts in the README.
"""

import os
from pathlib import Path


def count_research_problems(research_dir: Path) -> int:
    """Count research problems by counting evaluator.py files, with special handling for poc_generation."""
    count = 0
    
    # Special case: poc_generation counts as 4 problems (4 subcategories)
    poc_dir = research_dir / 'poc_generation'
    if poc_dir.exists():
        count += 4
    
    # Count all evaluator.py files, excluding those in poc_generation
    for evaluator_file in research_dir.rglob('evaluator.py'):
        # Skip if it's under poc_generation directory
        if 'poc_generation' not in str(evaluator_file):
            count += 1
    
    return count


def count_algorithmic_problems(algorithmic_dir: Path) -> int:
    """Count algorithmic problems by counting numbered directories."""
    problems_dir = algorithmic_dir / 'problems'
    
    if not problems_dir.exists():
        return 0
    
    count = 0
    for item in problems_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            count += 1
    
    return count


def update_readme_badge(readme_path: Path, research_count: int, algo_count: int):
    """Update the README with current problem counts using badges."""
    
    if not readme_path.exists():
        print(f"README not found at {readme_path}")
        return
    
    content = readme_path.read_text()
    
    # Create badge line
    badge_line = f"![Research Problems](https://img.shields.io/badge/Research_Problems-{research_count}-blue) ![Algorithmic Problems](https://img.shields.io/badge/Algorithmic_Problems-{algo_count}-green)\n\n"
    
    # Find where to insert badges (after the title line)
    lines = content.split('\n')
    
    # Remove old badge line if exists
    lines = [line for line in lines if not line.startswith('![Research Problems]')]
    
    # Insert new badge line after the header links line
    for i, line in enumerate(lines):
        if '[Website]' in line and '[GitHub]' in line:
            lines.insert(i + 1, '')
            lines.insert(i + 2, badge_line.strip())
            break
    
    # Write back
    readme_path.write_text('\n'.join(lines))
    print(f"✅ Updated README with counts: Research={research_count}, Algorithmic={algo_count}")


def main():
    # Get repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Count problems
    research_dir = repo_root / 'research'
    algorithmic_dir = repo_root / 'algorithmic'
    readme_path = repo_root / 'README.md'
    
    research_count = count_research_problems(research_dir)
    algo_count = count_algorithmic_problems(algorithmic_dir)
    
    print(f"📊 Problem Statistics:")
    print(f"   Research Problems: {research_count}")
    print(f"   Algorithmic Problems: {algo_count}")
    print(f"   Total: {research_count + algo_count}")
    
    # Update README
    update_readme_badge(readme_path, research_count, algo_count)


if __name__ == '__main__':
    main()
