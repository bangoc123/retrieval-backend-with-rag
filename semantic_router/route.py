from typing import List
class Route():
    def __init__(
        self,
        name: str = None,
        samples:List = []
    ):

        self.name = name
        self.samples = samples
