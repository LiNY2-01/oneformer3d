import logging
from oneformer3d.spt.data import NAG
from oneformer3d.spt.transforms import Transform


log = logging.getLogger(__name__)


__all__ = ['HelloWorld']


class HelloWorld(Transform):
    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def _process(self, nag):
        log.info("\n**** Hello World ! ****\n")
        return nag
