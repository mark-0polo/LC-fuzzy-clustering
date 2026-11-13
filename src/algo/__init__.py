"""The ten benchmark algorithms, each with its own named source file."""
from .fcm import FuzzyCMeans
from .pcm import PossibilisticCMeans
from .pfcm import PossibilisticFuzzyCMeans
from .dpc import DensityPeaks
from .efcm import EntropyRegularizedFCM
from .mc_fcm import MultipleCoefficientFuzzyCMeans
from .lc_fixed_t import LCFixedT
from .lc_fixed_half_t import LCFixedHalfT
from .lc_soft_fixed_t import LCSoftFixedT
from .kmeans_pp import KMeansPlusPlus

__all__ = ['FuzzyCMeans', 'PossibilisticCMeans', 'PossibilisticFuzzyCMeans',
           'DensityPeaks', 'EntropyRegularizedFCM', 'MultipleCoefficientFuzzyCMeans',
           'LCFixedT', 'LCFixedHalfT', 'LCSoftFixedT', 'KMeansPlusPlus']
