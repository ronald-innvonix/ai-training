import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from data import DIRECTORIES

# Name shown to clients. `instructions` helps the LLM understand the server.
mcp = FastMCP(
    name="Employee Directory",
    instructions=(
        "Company employee directory. Use the search tools to find people by "
        "name, title, department, location or skill. Employee IDs look like E001."
    ),
)

# Shortcut to the employee list defined in data.py
EMPLOYEES = DIRECTORIES["employees"]


# --------------------------------------------------------------------------
# Small helpers (not exposed to MCP)
# --------------------------------------------------------------------------
def _find_by_id(emp_id: str) -> Optional[dict]:
    """Return one employee dict by ID (case-insensitive) or None."""
    for emp in EMPLOYEES:
        if emp["id"].lower() == emp_id.lower():
            return emp
    return None


# ==========================================================================
# RESOURCES  (read-only data identified by URI)
# ==========================================================================
@mcp.resource("directory://employees", mime_type="application/json")
def all_employees() -> str:
    """Static resource: the full employee list as JSON."""
    return json.dumps(EMPLOYEES, indent=2)


@mcp.resource("directory://departments", mime_type="application/json")
def departments_resource() -> str:
    """Static resource: department names with headcount."""
    counts: dict[str, int] = {}
    for emp in EMPLOYEES:
        counts[emp["department"]] = counts.get(emp["department"], 0) + 1
    return json.dumps(counts, indent=2)


# Resource TEMPLATE: {emp_id} in the URI becomes the function argument.
# e.g. directory://employees/E003
@mcp.resource("directory://employees/{emp_id}", mime_type="application/json")
def employee_resource(emp_id: str) -> str:
    """Templated resource: one employee's record by ID."""
    emp = _find_by_id(emp_id)
    if emp is None:
        return json.dumps({"error": f"No employee with id {emp_id}"})
    return json.dumps(emp, indent=2)


# ==========================================================================
# TOOLS  (functions the LLM can call)
# The docstring + type hints become the tool description/schema that the
# LLM sees - so write them clearly!
# ==========================================================================
@mcp.tool()
def search_employees(
    query: str,
    department: Optional[str] = None,
    location: Optional[str] = None,
) -> list[dict]:
    """Search employees by free text (name, title, email or skill).

    Optionally narrow by department and/or location (case-insensitive).
    Returns a list of matching employee records.
    """
    q = query.lower().strip()
    results = []
    for emp in EMPLOYEES:
        # Build one searchable string from the interesting fields
        haystack = " ".join(
            [emp["name"], emp["title"], emp["email"], *emp["skills"]]
        ).lower()

        if q and q not in haystack:
            continue
        if department and emp["department"].lower() != department.lower():
            continue
        if location and emp["location"].lower() != location.lower():
            continue
        results.append(emp)
    return results


@mcp.tool()
def get_employee(emp_id: str) -> dict:
    """Get a single employee's full record by ID (e.g. 'E003')."""
    emp = _find_by_id(emp_id)
    return emp if emp else {"error": f"No employee with id {emp_id}"}


@mcp.tool()
def list_department_members(department: str) -> list[dict]:
    """List all employees in a department (e.g. 'Engineering', 'Sales')."""
    return [e for e in EMPLOYEES if e["department"].lower() == department.lower()]


@mcp.tool()
def get_direct_reports(manager_id: str) -> list[dict]:
    """List everyone who reports directly to the given manager ID."""
    return [e for e in EMPLOYEES if e["manager_id"] == manager_id.upper()]


# --------------------------------------------------------------------------
# Entry point. Default transport is stdio: the client launches this file as
# a subprocess, so you normally don't run it by hand.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()  # same as mcp.run(transport="stdio")