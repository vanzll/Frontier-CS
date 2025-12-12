.PHONY: update-count install-hooks help

help:
	@echo "Frontier-CS Utilities"
	@echo ""
	@echo "Available commands:"
	@echo "  make update-count    - Update problem count in README"
	@echo "  make install-hooks   - Install pre-commit hooks"
	@echo "  make help           - Show this help message"

update-count:
	@echo "📊 Updating problem count..."
	@python3 scripts/update_problem_count.py

install-hooks:
	@echo "🔧 Installing pre-commit hooks..."
	@pip install pre-commit
	@pre-commit install
	@echo "✅ Pre-commit hooks installed!"
