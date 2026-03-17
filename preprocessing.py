def norm_text(text):
    text = text.replace("\n", " ")
    return ''.join(c for c in text if c.isalnum() or c == ' ')


def tokenize(text):
    tokens = norm_text(text).lower().split()
    tokens = [t for t in tokens if not t.isdigit()]

    return tokens

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace("\n", " ")
    text = text.replace("[1]", "")

    return text
    