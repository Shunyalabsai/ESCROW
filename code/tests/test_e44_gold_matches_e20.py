"""The gold builder E44 uses must be the one E20 was scored with, character for character.

E44 copies build_gold out of e20_headtohead.py instead of importing it, because importing that
module executes AutoPKG's prompt file. A copy can drift, and a drifted gold would make E44's floor
a floor for a different benchmark. This asserts it has not.
"""
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments")
PAT = re.compile(r"\ndef build_gold\(recs\):\n(?:.*?\n)*?    return keyfreq, pos, neg, len\(collide\)\n")


class TestGoldIsTheSame(unittest.TestCase):
    def test_identical(self):
        a = PAT.search(open(os.path.join(EXP, "e20_headtohead.py"), encoding="utf-8").read())
        b = PAT.search(open(os.path.join(EXP, "e44_string_algorithms_on_key_identity.py"),
                            encoding="utf-8").read())
        self.assertIsNotNone(a, "build_gold not found in e20_headtohead.py")
        self.assertIsNotNone(b, "build_gold not found in e44")
        self.assertEqual(a.group(0), b.group(0),
                         "E44's copy of build_gold has drifted from the one E20 was scored with")


if __name__ == "__main__":
    unittest.main()
