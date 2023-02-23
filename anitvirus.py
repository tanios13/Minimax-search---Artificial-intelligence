
def filter_text(text):
    rules_map = {}

    for line in text.split("\n"):
        if line.startswith("#"):
            (key, value) = map(str.strip, line[1:].split("="))
            rules_map[key] = value

    # check for collision
    for key in rules_map.keys():
        for value in rules_map.values():
            if key in value:
                return False

    return True