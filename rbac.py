"""
Role-based access control policy.

This is intentionally just data + one lookup helper. Every access decision
in the app should go through get_allowed_departments() so the policy stays
in exactly one place.

In a real deployment, DEMO_USERS would be replaced by real authentication
(SSO / JWT) that resolves a logged-in user to a role -- everything below
that point (ROLE_DEPARTMENTS onward) stays the same.
"""

ROLE_DEPARTMENTS: dict[str, list[str]] = {
    "finance": ["finance", "marketing", "general"],
    "marketing": ["marketing", "general"],
    "hr": ["hr", "general"],
    "engineering": ["engineering", "general"],
    "c_level": ["engineering", "finance", "general", "hr", "marketing"],
    "employee": ["general"],
}

DEMO_USERS: dict[str, dict] = {
    "priya": {"name": "Priya Nair", "title": "Finance Analyst", "role": "finance"},
    "arjun": {"name": "Arjun Verma", "title": "HR Manager", "role": "hr"},
    "meera": {"name": "Meera Iyer", "title": "Marketing Lead", "role": "marketing"},
    "dev": {"name": "Dev Kapoor", "title": "Engineering Lead", "role": "engineering"},
    "ananya": {"name": "Ananya Rao", "title": "CEO", "role": "c_level"},
    "sam": {"name": "Sam Fernandes", "title": "Support Associate", "role": "employee"},
}


def get_allowed_departments(role: str) -> list[str]:
    if role not in ROLE_DEPARTMENTS:
        raise ValueError(f"Unknown role: {role!r}")
    return ROLE_DEPARTMENTS[role]
