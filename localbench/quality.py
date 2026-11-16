"""Fixed quality set for comparing two quantizations of one model.

Each item has one exact answer and asks for the answer only, so grading is string
matching rather than a judge. The point is the *difference* between two variants of the
same model on the same items, not the absolute score: a 1B model gets some of these wrong
at any precision, and that is fine as long as the lower-precision variant doesn't get
noticeably more wrong.
"""
from __future__ import annotations

import re

ITEMS: list[tuple[str, str]] = [
    ("What is 17 + 26?", "43"), ("What is 9 * 8?", "72"), ("What is 144 / 12?", "12"),
    ("What is 1000 - 357?", "643"), ("What is 15% of 200?", "30"), ("What is 2 to the power 10?", "1024"),
    ("What is the square root of 81?", "9"), ("What is 7 * 13?", "91"), ("What is 250 + 375?", "625"),
    ("What is 3 * 3 * 3?", "27"), ("How many days are in a leap year?", "366"),
    ("How many minutes are in 3 hours?", "180"), ("How many sides does a hexagon have?", "6"),
    ("How many legs does a spider have?", "8"), ("How many continents are there?", "7"),
    ("What is the chemical symbol for gold?", "Au"), ("What is the chemical symbol for sodium?", "Na"),
    ("What planet is closest to the Sun?", "Mercury"), ("What is the largest planet in the Solar System?", "Jupiter"),
    ("What is the capital of Japan?", "Tokyo"), ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"), ("What is the capital of Kenya?", "Nairobi"),
    ("What is the capital of Italy?", "Rome"), ("Which ocean is the largest?", "Pacific"),
    ("What gas do plants absorb from the air?", "Carbon dioxide"), ("What is H2O commonly called?", "Water"),
    ("In which year did World War II end?", "1945"), ("How many bits are in a byte?", "8"),
    ("What is the binary number 1010 in decimal?", "10"), ("What is the hexadecimal number FF in decimal?", "255"),
    ("Which language runs natively in web browsers: Python or JavaScript?", "JavaScript"),
    ("What does HTTP status code 404 mean: Not Found or Forbidden?", "Not Found"),
    ("What is the opposite of 'hot'?", "Cold"), ("What is the plural of 'mouse'?", "Mice"),
    ("Spell the word 'cat' backwards.", "tac"), ("What is the first letter of the English alphabet?", "A"),
    ("What color do you get by mixing blue and yellow?", "Green"), ("How many hours are in a day?", "24"),
    ("What is 11 * 11?", "121"),
]
ANSWERS = dict(ITEMS)
SYSTEM = "Answer with only the answer: a number, a word, or a short name. No explanation."


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^(the answer is|answer:)\s*", "", text)
    return re.sub(r"[\s.,!\"'*`]+$", "", text)


def correct(answer: str, expected: str) -> bool:
    got, want = normalize(answer), normalize(expected)
    if got == want:
        return True
    # accept the expected answer as a whole word inside a short reply ("It's Tokyo.")
    return len(got) <= 4 * len(want) + 12 and re.search(rf"\b{re.escape(want)}\b", got) is not None
