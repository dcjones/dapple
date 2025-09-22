
from .elements import Element

# TODO: Should I bother defining shortcuts like this???

# Defining shortcuts for constructing svg tags as I need them
def g(**kwargs):
    return Element("g", **kwargs)
