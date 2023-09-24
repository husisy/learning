# learning

Just be happy. [中文版](README_zh.md)

## Repository Content and Organization

This repository contains various code snippets related to computer skills and personal notes. They are categorized and organized based on programming languages, including but not limited to:

```bash
.
├── python
│   ├── python
│   ├── pytorch
│   ├── tensorflow
│   └── scipy
├── linux
│   ├── docker
│   ├── git
│   └── shell
└── cpp
    ├── cpp
    ├── STL
    ├── cmake
    └── boost
```

Usually, the top-level directory indicates the programming language, and the second level represents libraries based on that language. The purpose of the files under these directories can generally be inferred from their names:

1. `xxx.md`: Explanatory text. For example, `README.md` typically includes official website links, documentation links, installation commands, and a minimum working example (MWE).
2. `draft_x.y`: Code snippets. For example, `draft00.py` is usually a series of code snippets. It's recommended to run them in an interactive environment (like `ipython/bash`) and observe the results.
3. `demo_x.y`: Encapsulated code snippets, usually refined from `draft_x.y`. For instance, `python/matplotlib/demo_3d.py` demonstrates 3D image plotting.
   * However, not all `drafts` are suitable for refining into `demo`.
4. `test_x.y`: Unit test code snippets, typically refined from `demo_x.y`. For example, `python/scipy/test_linalg.py` would test if the `U` matrix obtained from the `svd` decomposition is unitary.
   * When debugging a specific library error, I sometimes run these `test_x.y` files (enter `pytest -v xxx` in the command line) to ensure the library is installed correctly.
5. `ws_xx/` directories: When a feature cannot be demonstrated with a single `draft_x.y` file, it's placed in a separate directory.
   * `ws` is short for `workspace`.

## How to Use This Repository

1. To learn about a specific library: Locate the directory, first read `README.md`, which contains reference links and installation commands. Next, execute a series of code snippets in the `draft_x.y` series. These code snippets are often self-explanatory but may appear obscure and difficult to understand due to a lack of comments. It's recommended to simultaneously read the official documentation recorded in `README.md`.
2. Build project code from minimal code snippets: `draft_x.y` contains a series of minimal runnable code snippets, which are perfect foundational building blocks for constructing projects.
3. For uncertain code behaviors in the project code, write the minimal runnable code to determine its behavior and record it in this repository.
4. Build your own note system: Everyone's knowledge system varies greatly, so the note system they need will naturally differ. The important tools personal believed may not be suitable for other fields, but the way of organizing notes should be universal. Therefore, I strongly suggest that readers start building their own note system.

## Other

1. This repository **does not** accept pull requests.
   * The organization and content of this repository are largely determined by personal style and it's difficult to reach a consensus (on aspects like line width, spaces, line breaks, etc.).
   * Everyone's note system will vary greatly. I can hardly imagine anyone adding content to this repository. Perhaps it would be more appropriate to establish a personal note system from scratch.
   * If you really need to add content based on this repository, simply fork this repository and add content.
2. About the license:
   * Choosing a license is an extremely headache-inducing task. See [choose-a-license](https://choosealicense.com/) and [Open Source Guide](https://opensourceway.community/open-source-guide/legal/). It was only after I saw Linus's explanation of the GNUv2 protocol - "I give you source code, you give me your changes back, we are even." - that I liked this succinct explanation, so I chose GNUv2.
   * It must be admitted that most of the text in this repository is "stealing" from somewhere by reading the source code and documentation of other repositories, absorbing the key parts into this repository. The worst part is that some code is just copy-pasted without understanding (obvious plagiarism). I hope that I will have time to sort out this kind of code in the future. For the text where the license is incompatible, please inform me, and I will delete it.

## TODO

1. [ ] backward compatibility and forward compatibility
2. [ ] python-qt5
3. [ ] indexing and kernel
4. [ ] `python/deepxde`
5. [ ] [github/RLcode](https://github.com/louisnino/RLcode)
6. [ ] [github/tvm-learn](https://github.com/BBuf/tvm_learn)
7. [ ] [github/sisl](https://github.com/zerothi/sisl): python tight-binding, NEGF, DFT
8. [ ] quantum chemistry package
9. [ ] pde: [github/solver-in-the-loop](https://github.com/tum-pbs/Solver-in-the-Loop)
10. [ ] [missing-semester](https://missing-semester-cn.github.io/)
11. [ ] NJU/ics2023 [link](https://nju-projectn.github.io/ics-pa-gitbook/ics2021/)
12. [ ] [deepmd](https://docs.deepmodeling.org/projects/deepmd/en/master/index.html)
