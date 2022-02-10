import difflib

string1 = "apple pie available"
string2 = "come have some apple pies"

match = difflib.SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))

print(match)  # -> Match(a=0, b=15, size=9)
print(string1[match.a: match.a + match.size])  # -> apple pie
print(string2[match.b: match.b + match.size])  # -> apple pie
