import os

# تحميل المعرفة
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge.txt")

with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    KNOWLEDGE = f.read().split("\n")


def ask_ai(question: str):

    question = question.lower()

    best_lines = []

    for line in KNOWLEDGE:
        if any(word in line.lower() for word in question.split()):
            best_lines.append(line)

    if not best_lines:
        return "I could not find information about that in the arrhythmia knowledge base."

    return " ".join(best_lines[:6])