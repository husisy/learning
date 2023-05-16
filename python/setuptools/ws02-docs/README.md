# mypackage

```bash
conda install -c conda-forge mkdocstrings mkdocstrings-python
pip install mkdocstrings 'mkdocstrings[crystal,python]'
```

## documentation

```bash
# no need to run 'pip install .' first
mkdocs serve
```

## package install and uninstall

install

```bash
pip install .
```

run

```bash
python -c "import mypackage; mypackage.say()"
```

```python
import mypackage
mypackage.__version__
mypackage._package
```

run scripts

```bash
myscripts
```

uninstall

```bash
pip uninstall mypackage
```
