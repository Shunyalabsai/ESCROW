"""Brute-force Kraft check of the value block under both naming rules."""
import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "code"))
from escrow import codes as C
from escrow.codes import ValueBlock

def sweep(tag):
    rng = random.Random(0)
    worst_dev, worst_case = 0.0, None
    worst_formula = 0.0
    for _ in range(4000):
        inv = rng.randrange(1, 40)
        b = ValueBlock()
        for _ in range(rng.randrange(0, 60)):
            b.observe(rng.randrange(inv))
        m = b.p_mass_full(inv)
        shipped_formula = 1.0 - b.u * (b.u + 0.5) / ((b.N + 1.0) * (inv + 1.0))
        if not C.TIGHT_NAMING:
            worst_formula = max(worst_formula, abs(m - shipped_formula))
        dev = abs(m - 1.0)
        if dev > worst_dev:
            worst_dev, worst_case = dev, dict(u=b.u, N=b.N, inv=inv, mass=m)
        # stages i+ii alone must always be exactly 1 (UT-20)
        assert abs(b.p_mass_check() - 1.0) < 1e-12
    print(f"{tag}: worst |mass - 1| = {worst_dev:.3e}  at {worst_case}")
    if not C.TIGHT_NAMING:
        print(f"   worst |mass - closed form 1-u(u+1/2)/((N+1)(inv+1))| = {worst_formula:.3e}")
    return worst_dev

print("flags:", C.flags())
d = sweep("SHIPPED naming log2(inv+1)" if not C.TIGHT_NAMING else "TIGHT naming log2(inv-u+1)")
if C.TIGHT_NAMING:
    assert d < 1e-12, d
    print("PASS: every block's three-stage predictive sums to exactly one")
else:
    print("shipped rule is a strict sub-probability, as E16 measured")
