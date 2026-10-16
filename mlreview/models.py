from dataclasses import dataclass

@dataclass
class Finding:
    rule_id: str
    severity: str   
    line: int
    message: str
    fix: str