"""Filesystem MCP Server - Provides file system operations."""

from pathlib import Path

from fastmcp import FastMCP

# Initialize FastMCP server
app = FastMCP("filesystem-server")


@app.tool()
def list_files(directory: str = ".", pattern: str = "*") -> list[str]:
    """
    List files in a directory with optional pattern matching.

    Args:
        directory: Directory path to list (relative to current working directory)
        pattern: Glob pattern to filter files (e.g., "*.py", "**/*.txt")

    Returns:
        List of file names matching the pattern
    """
    try:
        path = Path(directory)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            return [f"Error: Directory {directory} does not exist"]

        if pattern == "*" or pattern == "**/*":
            files = list(path.rglob("*")) if "**" in pattern else list(path.glob("*"))
        else:
            files = list(path.glob(pattern))

        # Filter to only files, not directories
        file_names = [f.name for f in files if f.is_file()]
        return file_names

    except Exception as e:
        return [f"Error listing files: {str(e)}"]


@app.tool()
def read_file(file_path: str) -> str:
    """
    Read the contents of a file.

    Args:
        file_path: Path to the file to read (relative to current working directory)

    Returns:
        File contents as a string, or error message
    """
    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            return f"Error: File {file_path} does not exist"

        if not path.is_file():
            return f"Error: {file_path} is not a file"

        with open(path, encoding="utf-8") as f:
            content = f.read()

        return content

    except UnicodeDecodeError:
        return f"Error: Cannot read file {file_path} - may be binary or have encoding issues"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@app.tool()
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file. Creates directories as needed.

    Args:
        file_path: Path where to write the file (relative to current working directory)
        content: Content to write to the file

    Returns:
        Success message or error message
    """
    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path.cwd() / path

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote {len(content)} characters to {file_path}"

    except Exception as e:
        return f"Error writing file: {str(e)}"


@app.tool()
def get_file_info(file_path: str) -> dict:
    """
    Get information about a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            return {"error": f"File {file_path} does not exist"}

        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path),
            "size": stat.st_size,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "modified": stat.st_mtime,
            "exists": True,
        }

    except Exception as e:
        return {"error": str(e)}


def main():
    """Run the MCP server."""
    import asyncio
    
    asyncio.run(app.run_stdio_async())


if __name__ == "__main__":
    main()
