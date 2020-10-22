from importlib import import_module
from .core import MatFormatReader, ConstraintBasedModelSimulator
from .cobamp import CobampModelObjectReader

model_readers = {'numpy': MatFormatReader, 'cobamp.core.models': CobampModelObjectReader}

external_frameworks = {'cobra':'COBRAModelObjectReader', 'reframed':'FramedModelObjectReader'}
external_framework_readers = {}

available_readers_dict = {
	'numpy':'MatFormatReader',
	'cobamp.core.models':'CobampModelObjectReader',
	'cobra.core.model': 'COBRAModelObjectReader',
	'framed.model.cbmodel': 'FramedModelObjectReader'
}
for module_name, reader_name in external_frameworks.items():
	try:
		module = import_module(module_name, '')
		print(reader_name,'is available for',module_name)
		cobamp_module = import_module('.'+module_name, package='cobamp.wrappers')
		reader_class = getattr(cobamp_module, reader_name)
		globals().update({module_name: cobamp_module, reader_name: reader_class})
		external_framework_readers[reader_name] = reader_class
	except Exception as e:
		print(reader_name,'could not be loaded for',module_name)


available_readers_dict = {k:v for k,v in available_readers_dict.items() if v in globals()}

def get_model_reader(model_obj, **kwargs):
	if type(model_obj).__module__ in available_readers_dict.keys():
		return globals()[available_readers_dict[type(model_obj).__module__]](model_obj, **kwargs)
	else:
		raise TypeError('model_obj has an unknown type that could not be read with cobamp:', type(model_obj).__module__)


from . import method_wrappers
from .method_wrappers import KShortestEFMEnumeratorWrapper, KShortestEFPEnumeratorWrapper, KShortestMCSEnumeratorWrapper
from .method_wrappers import KShortestGenericMCSEnumeratorWrapper, KShortestGeneticMCSEnumeratorWrapper


