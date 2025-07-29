
from xml.etree.ElementTree import Element

# Defining shortcuts for constructing svg tags as I need them
def g(**kwargs):
    return Element("g", **kwargs)
