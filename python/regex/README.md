# regex

1. link
   * [python-STL-re](https://docs.python.org/3/library/re.html)
   * [Regular Expression HOWTO](https://docs.python.org/3/howto/regex.html#regex-howto)
   * [pypi/regex](https://pypi.org/project/regex/)
   * [浅谈Python 正则表达式](https://zhuanlan.zhihu.com/p/26019553)
2. metacharacter: `. ^ $ * + ? { } [ ] \ | ( )`
3. `[]`
   * `[abc]`
   * `[A-Za-z]`
   * `[^abc]`
   * metacharacters are not active inside: `[abc$]`
4. `^`
5. `\d`, `\D`, `[0-9]`
6. `\s`, `\S`, `[ \t\n\r\f\v]`
7. `\w`, `\W`, `[a-zA-Z0-9_]`
8. `.` except newline character, `'re.DOTALL`
9. `*`, `+`
10. `re.IGNORECASE`
11. `\\\\section`
12. **lookahead assertions**
