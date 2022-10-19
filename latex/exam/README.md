# exam

1. link
   * [ctan/exam-randomizechoices](https://ctan.org/pkg/exam-randomizechoices)
   * [github/exam-randomizechoices](https://github.com/jesseopdenbrouw/exam-randomizechoices)

```bash
tlmgr install exam-randomizechoices

# randomize answer: use a different seed in main.tex
# or just comment it, the seed is different every minute
# \setrandomizerseed{998}

# show the answer with the question
# \documentclass[answers,addpoints]{exam}

# hide the answer
# \documentclass[addpoints]{exam}

# turn off the random behavior
# \usepackage[norandomize,nokeeplast]{exam-randomizechoices}

# turn on the random behavior
# \usepackage[randomize,nokeeplast]{exam-randomizechoices}
```
