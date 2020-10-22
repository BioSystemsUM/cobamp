import warnings
from .cobra import COBRAModelObjectReader
from .framed import FramedModelObjectReader
from .cobamp import CobampModelObjectReader
from .core import MatFormatReader, AbstractObjectReader

from ..wrappers import get_model_reader, model_readers

warnings.warn(
    '''\nThe wrappers.external_wrappers module will be deprecated in a future release in favour of the wrappers module. 
    Available ModelObjectReader classes can still be loaded using cobamp.wrappers.<class>. An appropriate model 
    reader can also be created using the get_model_reader function on cobamp.wrappers''')
