"""Code Parser MCP Server - Provides Python code analysis tools."""

import ast
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

# Initialize FastMCP server
app = FastMCP("code-parser-server")


class PythonCodeAnalyzer:
    """Helper class for analyzing Python code."""

    def __init__(self, source_code: str, file_path: str = ""):
        self.source_code = source_code
        self.file_path = file_path
        self.tree = None
        try:
            self.tree = ast.parse(source_code)
        except SyntaxError:
            pass  # We'll handle this gracefully

    def extract_functions(self) -> list[dict[str, Any]]:
        """Extract function definitions from the code."""
        if not self.tree:
            return []

        functions = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "end_line_number": getattr(node, "end_lineno", node.lineno),
                    "parameters": [arg.arg for arg in node.args.args],
                    "has_docstring": False,
                    "complexity_score": self._calculate_complexity(node),
                    "code_length": len(
                        self.source_code.split("\n")[node.lineno - 1 : getattr(node, "end_lineno", node.lineno)]
                    ),
                }

                # Check for docstring
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                    func_info["has_docstring"] = True

                functions.append(func_info)

        return functions

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.And):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.Or):
                complexity += len(child.values) - 1

        return complexity


@app.tool()
def parse_functions(file_path: str) -> list[dict[str, Any]]:
    """
    Parse a Python file and extract function definitions.

    Args:
        file_path: Path to the Python file to analyze

    Returns:
        List of dictionaries containing function information
    """
    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            return [{"error": f"File {file_path} does not exist"}]

        if not path.suffix == ".py":
            return [{"error": f"File {file_path} is not a Python file"}]

        with open(path, encoding="utf-8") as f:
            source_code = f.read()

        analyzer = PythonCodeAnalyzer(source_code, str(path))
        functions = analyzer.extract_functions()

        return functions

    except UnicodeDecodeError:
        return [{"error": f"Cannot read file {file_path} - encoding issue"}]
    except Exception as e:
        return [{"error": f"Error parsing file: {str(e)}"}]


@app.tool()
def get_function_details(file_path: str, function_name: str) -> dict[str, Any]:
    """
    Get detailed information about a specific function.

    Args:
        file_path: Path to the Python file
        function_name: Name of the function to analyze

    Returns:
        Detailed function information
    """
    try:
        functions = parse_functions(file_path)

        for func in functions:
            if func.get("name") == function_name:
                # Get the actual source code for the function
                path = Path(file_path)
                if not path.is_absolute():
                    path = Path.cwd() / path

                with open(path, encoding="utf-8") as f:
                    lines = f.readlines()

                start_line = func["line_number"] - 1  # Convert to 0-based indexing
                end_line = func.get("end_line_number", start_line + 10)

                # Extract function code (with some context)
                context_start = max(0, start_line - 2)
                context_end = min(len(lines), end_line + 2)
                code_lines = lines[context_start:context_end]

                func["code"] = "".join(code_lines)
                func["context_lines"] = context_start + 1  # 1-based line numbers

                return func

        return {"error": f"Function '{function_name}' not found in {file_path}"}

    except Exception as e:
        return {"error": str(e)}


@app.tool()
def analyze_complexity(functions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze complexity of multiple functions and provide recommendations.

    Args:
        functions: List of function information dictionaries

    Returns:
        Analysis report with complexity issues and recommendations
    """
    try:
        analysis = {
            "total_functions": len(functions),
            "complexity_issues": [],
            "recommendations": [],
            "summary": {"high_complexity": 0, "too_many_params": 0, "too_long": 0, "missing_docstrings": 0},
        }

        for func in functions:
            if isinstance(func, dict) and "error" not in func:
                issues = []

                # Check complexity
                complexity = func.get("complexity_score", 0)
                if complexity > 10:
                    issues.append(f"High complexity ({complexity}) - consider refactoring")
                    analysis["summary"]["high_complexity"] += 1

                # Check parameter count
                params = func.get("parameters", [])
                if len(params) > 5:
                    issues.append(f"Too many parameters ({len(params)}) - violates single responsibility")
                    analysis["summary"]["too_many_params"] += 1

                # Check function length
                code_length = func.get("code_length", 0)
                if code_length > 50:
                    issues.append(f"Function too long ({code_length} lines) - consider splitting")
                    analysis["summary"]["too_long"] += 1

                # Check docstring
                if not func.get("has_docstring", False):
                    issues.append("Missing docstring")
                    analysis["summary"]["missing_docstrings"] += 1

                if issues:
                    analysis["complexity_issues"].append(
                        {
                            "function": func.get("name", "unknown"),
                            "line_number": func.get("line_number", 0),
                            "issues": issues,
                        }
                    )

        # Generate recommendations
        if analysis["summary"]["high_complexity"] > 0:
            analysis["recommendations"].append(
                "Consider breaking down high-complexity functions into smaller, focused functions"
            )

        if analysis["summary"]["too_many_params"] > 0:
            analysis["recommendations"].append(
                "Functions with many parameters should be refactored to use data objects or split into multi functions"
            )

        if analysis["summary"]["too_long"] > 0:
            analysis["recommendations"].append("Long functions should be split into smaller, single-purpose functions")

        if analysis["summary"]["missing_docstrings"] > 0:
            analysis["recommendations"].append("Add docstrings to all public functions for better code documentation")

        return analysis

    except Exception as e:
        return {"error": str(e)}


def main():
    """Run the MCP server."""
    import asyncio
    
    asyncio.run(app.run_stdio_async())


if __name__ == "__main__":
    main()
