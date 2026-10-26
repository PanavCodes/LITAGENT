#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party
SOURCES_FILE="third_party/SOURCES.txt"
echo "# Cloned reference repositories and inspected SHAs" > "$SOURCES_FILE"
echo "# Recorded: $(date -u)" >> "$SOURCES_FILE"

REPOS=(
  "https://github.com/PMRA-DSA/dsa-public.git dsa-public"
  "https://github.com/neo4j/neo4j-graphrag-python.git neo4j-graphrag-python"
  "https://github.com/langchain-ai/langchain-neo4j.git langchain-neo4j"
  "https://github.com/neo4j-labs/text2cypher.git text2cypher"
  "https://github.com/piyushagni5/langgraph-ai.git langgraph-ai"
  "https://github.com/langchain-ai/langgraph.git langgraph"
  "https://github.com/explodinggradients/ragas.git ragas"
)

for entry in "${REPOS[@]}"; do
  url=$(echo "$entry" | awk '{print $1}')
  name=$(echo "$entry" | awk '{print $2}')
  target="third_party/$name"
  if [ ! -d "$target" ]; then
    echo "Shallow cloning $name..."
    git clone --depth 1 "$url" "$target" || echo "Warning: Failed to clone $name, continuing..."
  fi
  if [ -d "$target" ]; then
    sha=$(git -C "$target" rev-parse HEAD 2>/dev/null || echo "unknown")
    echo "$name: $url @ $sha" >> "$SOURCES_FILE"
  fi
done

echo "Repository sources recorded in $SOURCES_FILE"
