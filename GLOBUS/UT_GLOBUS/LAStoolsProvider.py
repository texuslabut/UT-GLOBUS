import sys
sys.path.insert(1, '/home/users3/hk25639/Desktop/LAStoolsPluginQGIS3-master/')
from PyQt5.QtGui import QIcon
from qgis.core import QgsProcessingProvider
from processing.core.ProcessingConfig import Setting, ProcessingConfig
from LAStools import LAStoolsUtils
from LAStools.LAStoolsProduction.txt2lasPro import txt2lasPro
from LAStools.LAStoolsProduction.lasgridPro import lasgridPro
from LAStools import resources


class LAStoolsProvider(QgsProcessingProvider):

    def __init__(self):
        QgsProcessingProvider.__init__(self)

    def load(self):
        ProcessingConfig.settingIcons[self.name()] = self.icon()
        ProcessingConfig.addSetting(Setting(self.name(), 'LASTOOLS_ACTIVATED', 'Activate', True))
        ProcessingConfig.addSetting(
            Setting(self.name(), 'LASTOOLS_FOLDER', 'LAStools folder', "/home/users3/hk25639/LAStools", valuetype=Setting.FOLDER))
        ProcessingConfig.addSetting(Setting(self.name(), 'WINE_FOLDER', 'Wine folder', "", valuetype=Setting.FOLDER))
        ProcessingConfig.readSettings()
        self.refreshAlgorithms()
        return True

    def unload(self):
        ProcessingConfig.removeSetting('LASTOOLS_ACTIVATED')
        ProcessingConfig.removeSetting('LASTOOLS_FOLDER')
        ProcessingConfig.removeSetting('WINE_FOLDER')
        pass

    def isActive(self):
        return ProcessingConfig.getSetting('LASTOOLS_ACTIVATED')

    def setActive(self, active):
        ProcessingConfig.setSettingValue('LASTOOLS_ACTIVATED', active)

    def loadAlgorithms(self)
        self.algs = [lasgridPro(), txt2lasPro()]

        for alg in self.algs:
            self.addAlgorithm(alg)

        self.algs = [hugeFileClassify(), hugeFileGroundClassify(), hugeFileNormalize(), flightlinesToCHM_FirstReturn(),
                     flightlinesToCHM_HighestReturn(), flightlinesToCHM_SpikeFree(),
                     flightlinesToDTMandDSM_FirstReturn(), flightlinesToDTMandDSM_SpikeFree(),
                     flightlinesToMergedCHM_FirstReturn(), flightlinesToMergedCHM_HighestReturn(),
                     flightlinesToMergedCHM_PitFree(), flightlinesToMergedCHM_SpikeFree()]

        for alg in self.algs:
            self.addAlgorithm(alg)

    def icon(self):
        return QIcon("/home/users3/hk25639/Desktop/LAStoolsPluginQGIS3-master/LAStools/LAStools.png")

    def id(self):
        return 'LAStools'

    def name(self):
        return 'LAStools'

    def longName(self):
        return 'LAStools LiDAR and point cloud processing'
