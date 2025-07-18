
# Figuring some things out here:
#   - We should be able to build our tree using xml.etree.ElementTree, which
#     is pretty much our XML representation.
#
#

from .treemap import treemap


def plot(*args, **kwargs):
    # TODO:
    #   We treat args as though they are all plot elements
    #   like geometry scales and such
    #
    # kwargs can be used for like defaults=nothing and other options.

    pass
