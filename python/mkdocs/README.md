# mkdocs

1. link
   * [documentation](https://www.mkdocs.org/)
     * [link](https://squidfunk.github.io/mkdocs-material/publishing-your-site/) publishing your site, github-action
   * [github/mkdocs](https://github.com/mkdocs/mkdocs/)
   * [github/mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter)
   * [github/catalog](https://github.com/mkdocs/catalog) list of plugin
   * [github/pygments](https://github.com/pygments/pygments): generic syntax highligher
2. install
   * `conda install -c conda-forge mkdocs`
   * `pip install mkdocs`
3. `mkdocs.yml`
   * `site_name`: required
   * `site_url`: 默认值`https://example.com/`
4. ignore `.foo.md`, `.bar/baz.md`

```bash
mkdocs new my-project
mkdocs serve
mkdocs build

pip install mkdocs-jupyter mkdocs-material pymdown-extensions
```

```yaml
site_name: MkLorum
site_url: https://example.com/
nav:
  - Home: index.md
  - About: about.md
theme: readthedocs
```
