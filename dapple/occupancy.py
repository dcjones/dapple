



from dapple.coordinates import AbsLengths


class Occupancy:
    """
    TODO: This will store some type of occupancy bit mask to help
    computing complex layout, like labeling and the such.
    """

    def __init__(self, width: AbsLengths, height: AbsLengths):
        self.width = width
        self.height = height
