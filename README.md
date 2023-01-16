# Heartbeat detection

Set of scripts to detect heartbeat


## Preparing to launch

### Install package

```bash
pip install -e .[dev,notebook]
```

* dev - optional dependencies for development

* notebook - optional dependencies for ability to run project notebooks

If you're using windows do not forget to add extra argument for pip for gpu support:

```bash
pip install -e .[dev,notebook] --extra-index-url https://download.pytorch.org/whl/cu116
```

**NOTE**: numbers in "_cu116_" may vary from time to time or system to system, if this does not work go to [pytorch official site](https://pytorch.org/) to get correct link for prebuilt binaries
