# Business Intelligence Pattern Loading
.PHONY: load-patterns check-qdrant patterns-help

# Load all business patterns into Qdrant vector database
load-patterns:
	@echo "🚀 Loading business patterns into Qdrant..."
	python testing/scripts/load_patterns_to_qdrant.py

# Check Qdrant connection and status
check-qdrant:
	@echo "🔍 Checking Qdrant connection..."
	python testing/scripts/load_patterns_to_qdrant.py --check

# Load patterns with verbose output
load-patterns-verbose:
	@echo "🚀 Loading business patterns (verbose)..."
	python testing/scripts/load_patterns_to_qdrant.py --verbose

# Test pattern search functionality
search-patterns:
	@echo "🔍 Running pattern search examples..."
	python testing/scripts/example_pattern_search.py --examples

# Search patterns with custom query
search-query:
	@read -p "Enter search query: " query; \
	python testing/scripts/example_pattern_search.py "$$query"

# Show pattern loading help
patterns-help:
	@echo "📋 Business Intelligence Pattern Loading Commands:"
	@echo ""
	@echo "  make load-patterns         - Load all patterns into Qdrant"
	@echo "  make check-qdrant          - Check Qdrant connection"
	@echo "  make load-patterns-verbose - Load patterns with verbose output"
	@echo "  make search-patterns       - Run example pattern searches"
	@echo "  make search-query          - Search patterns with custom query"
	@echo ""
	@echo "Environment variables needed:"
	@echo "  QDRANT_URL     - Qdrant server URL"
	@echo "  QDRANT_API_KEY - Qdrant API key"
	@echo ""
	@echo "Pattern files location: app/data/patterns/*.json"





