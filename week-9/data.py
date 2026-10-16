# Each employee is a plain dict -> easy to turn into JSON for MCP.
EMPLOYEES = [
    {"id": "E001", "name": "Aarav Shah",    "title": "CEO",                  "department": "Executive",   "email": "aarav.shah@acme.com",    "location": "Ahmedabad", "manager_id": None,   "skills": ["leadership", "strategy"]},
    {"id": "E002", "name": "Priya Patel",   "title": "VP Engineering",       "department": "Engineering", "email": "priya.patel@acme.com",   "location": "Ahmedabad", "manager_id": "E001", "skills": ["python", "architecture", "cloud"]},
    {"id": "E003", "name": "Rohan Mehta",   "title": "Senior Backend Engineer", "department": "Engineering", "email": "rohan.mehta@acme.com", "location": "Pune",      "manager_id": "E002", "skills": ["python", "fastapi", "postgresql"]},
    {"id": "E004", "name": "Sneha Iyer",    "title": "Frontend Engineer",    "department": "Engineering", "email": "sneha.iyer@acme.com",    "location": "Bengaluru", "manager_id": "E002", "skills": ["react", "typescript", "css"]},
    {"id": "E005", "name": "Karan Desai",   "title": "ML Engineer",          "department": "Engineering", "email": "karan.desai@acme.com",   "location": "Ahmedabad", "manager_id": "E002", "skills": ["python", "pytorch", "llm"]},
    {"id": "E006", "name": "Neha Kapoor",   "title": "HR Manager",           "department": "HR",          "email": "neha.kapoor@acme.com",   "location": "Mumbai",    "manager_id": "E001", "skills": ["recruiting", "onboarding"]},
    {"id": "E007", "name": "Vikram Singh",  "title": "Sales Director",       "department": "Sales",       "email": "vikram.singh@acme.com",  "location": "Delhi",     "manager_id": "E001", "skills": ["negotiation", "crm"]},
    {"id": "E008", "name": "Ananya Rao",    "title": "Account Executive",    "department": "Sales",       "email": "ananya.rao@acme.com",    "location": "Delhi",     "manager_id": "E007", "skills": ["crm", "presentations"]},
    {"id": "E009", "name": "Dev Joshi",     "title": "Finance Analyst",      "department": "Finance",     "email": "dev.joshi@acme.com",     "location": "Mumbai",    "manager_id": "E001", "skills": ["excel", "forecasting"]},
    {"id": "E010", "name": "Meera Nair",    "title": "DevOps Engineer",      "department": "Engineering", "email": "meera.nair@acme.com",    "location": "Pune",      "manager_id": "E002", "skills": ["kubernetes", "terraform", "aws"]},
]

# Registry of all directories. Future directories get added here.
DIRECTORIES = {
    "employees": EMPLOYEES,
    # "vendors": VENDORS,
    # "offices": OFFICES,
}