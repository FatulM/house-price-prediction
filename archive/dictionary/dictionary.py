from re import finditer


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


with open("words.txt", "r") as f:
    words = f.readlines()

all_words = set()
for word in words:
    word = word.strip()
    if len(word) > 1:
        all_words.add(word)
        for x in camel_case_split(word):
            x = x.strip().lower()
            if len(x) > 1:
                all_words.add(x)

for word in sorted(list(all_words)):
    print(f"<w>{word}</w>")
