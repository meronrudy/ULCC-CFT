"""Minimal noncommutative algebraic elements for PGGS toy."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Op:
    name: str

def compose(a: Op, b: Op) -> Op:
    return Op(name=f"{a.name}∘{b.name}")
