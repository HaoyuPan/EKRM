from pathlib import Path

from ekrm import EKRM


class DemoApplication(EKRM):
    groups = (
        EKRM.Group(name='other', index=1, color='#8ECFC9'),  # green
        EKRM.Group(name='feldspar', index=2, color='#FA7F6F'),  # red
        EKRM.Group(name='quartz', index=3, color='#FFBE7A'),  # yellow
        EKRM.Group(name='biotite', index=4, color='#999999'),  # black
    )


app = DemoApplication(Path(__file__).parent / 'befast-granite.jpg')
