#!/usr/bin/env python3
"""
Fix SQL syntax in intelligence schema
"""
import re


def fix_sqlite_schema(schema_text):
    """Fix SQLite schema by moving INDEX definitions outside CREATE TABLE"""
    # Extract table name
    table_match = re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", schema_text)
    if not table_match:
        return schema_text

    table_name = table_match.group(1)

    # Find and extract INDEX lines
    index_pattern = r"^\s*INDEX\s+(\w+)\s*\(([^)]+)\),?\s*$"
    indexes = []

    lines = schema_text.split("\n")
    cleaned_lines = []

    for line in lines:
        index_match = re.match(index_pattern, line.strip())
        if index_match:
            index_name = index_match.group(1)
            index_columns = index_match.group(2)
            indexes.append(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({index_columns});"
            )
        else:
            cleaned_lines.append(line)

    # Remove trailing comma before closing parenthesis
    for i, line in enumerate(cleaned_lines):
        if line.strip() == ")" and i > 0 and cleaned_lines[i - 1].strip().endswith(","):
            cleaned_lines[i - 1] = cleaned_lines[i - 1].rstrip().rstrip(",")

    # Combine table creation with indexes
    result = "\n".join(cleaned_lines)
    if indexes:
        result += "\n                \n" + "\n                ".join(indexes)

    return result


# Test on a simple example
test_schema = """CREATE TABLE IF NOT EXISTS test (
    id INTEGER PRIMARY KEY,
    name TEXT,
    INDEX idx_name (name)
)"""

print("Original:")
print(test_schema)
print("\nFixed:")
print(fix_sqlite_schema(test_schema))
