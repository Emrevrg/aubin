"""AUBIN by Norovox — açık-tabanlı, kalibre tipli karar modelleri."""
from .core import Aubin, AubinEnsemble, options
from .loop import AubinLoop, GemmaGenerator, OpenAIGenerator
from .control import AubinController
from .learn import AubinLearning, DecisionMemory, SelfCalibrator, SkillLibrary
from .engine import AubinEngine
from .omni import AubinOmni

__version__ = "0.1.0"
__all__ = ["Aubin", "AubinEnsemble", "options", "AubinLoop", "GemmaGenerator", "OpenAIGenerator", "AubinController", "AubinLearning", "DecisionMemory", "SelfCalibrator", "SkillLibrary", "AubinEngine", "AubinOmni"]
