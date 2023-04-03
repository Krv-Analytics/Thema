"Dishwasher safe data container"


class Tupper:
    def __init__(self, dirty, clean):
        self._dirty = dirty
        self._clean = clean

    @property
    def dirty(self):
        return self._dirty

    @property
    def clean(self):
        return self._clean
