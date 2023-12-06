from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingProvider
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterRasterLayer
from qgis.core import QgsProcessingParameterFeatureSink
from qgis.core import QgsCoordinateReferenceSystem
import processing


class Canopy_ground(QgsProcessingAlgorithm):

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer('GEDI', 'GEDI', optional=True, defaultValue=None))
        self.addParameter(QgsProcessingParameterRasterLayer('canopy', 'canopy', defaultValue=None))
        self.addParameter(QgsProcessingParameterRasterLayer('ground', 'ground', defaultValue=None))
        self.addParameter(QgsProcessingParameterFeatureSink('Points', 'points', type=QgsProcessing.TypeVectorPoint, createByDefault=True, defaultValue='./vector.gpkg'))

    def processAlgorithm(self, parameters, context, model_feedback):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(3, model_feedback)
        results = {}
        outputs = {}

        # Raster calculator
        alg_params = {
            'CELLSIZE': 0,
            'CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
            'EXPRESSION': '(("canopy@1"-"ground@1")>=0)*("canopy@1"-"ground@1")',
            'EXTENT': None,
            'LAYERS': [parameters['canopy'],parameters['ground']],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['RasterCalculator'] = processing.run('qgis:rastercalculator', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}
        # Merge
        alg_params = {
            'DATA_TYPE': 5,  # Float32
            'EXTRA': '',
            'INPUT': [parameters['GEDI'],outputs['RasterCalculator']['OUTPUT']],
            'NODATA_INPUT': None,
            'NODATA_OUTPUT': 0,
            'OPTIONS': '',
            'PCT': False,
            'SEPARATE': False,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['Merge'] = processing.run('gdal:merge', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        # Raster pixels to points
        alg_params = {
            'FIELD_NAME': 'VALUE',
            'INPUT_RASTER': outputs['Merge']['OUTPUT'],
            'RASTER_BAND': 1,
            'OUTPUT': parameters['Points']
        }
        outputs['RasterPixelsToPoints'] = processing.run('native:pixelstopoints', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['Points'] = outputs['RasterPixelsToPoints']['OUTPUT']
        return results

    def name(self):
        return 'canopy_ground'

    def displayName(self):
        return 'canopy_ground'

    def group(self):
        return ''

    def groupId(self):
        return ''

    def createInstance(self):
        return Canopy_ground()

