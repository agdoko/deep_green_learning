import apache_beam as beam
import apache_beam.transforms.window as window
import ee
import numpy as np
import sys
from apache_beam.options.pipeline_options import PipelineOptions
sys.path.insert(0, '/Users/felix/code/agdoko/deep_green_learning')
import params

# Initialize Earth Engine
ee.Initialize()

# Define your processing function
def process_images(elements):
    processed_images = []
    for element in elements:
        image_uri = f'gs://{params.BUCKET}/' + element   # Assuming each line in the input file is an image URI
        image = ee.Image.loadGeoTIFF(image_uri)

        # Perform your processing on the image
        # For example, compute NDVI
        NDVI = image.normalizedDifference(['B4', 'B8'])

        # Convert the processed image to an ndarray
        #ndvi_array = ndvi.toArray()

        #B4 = image['B4'].astype(float).toArray()
        #B8 = image['B8'].astype(float).toArray()

        # Calculate NDVI - basically the normalised difference between Red and NIR bands
        #NDVI = (B8 - B4) / (B8 + B4 + 1e-10)  # adding a small constant to avoid division by zero

        # Append the numpy array to the list
        processed_images.append(NDVI)

        #feature_stacked_array = np.stack(process_images, axis=0)

        # Add the processed image to the list

    return processed_images

# Define custom options for the Dataflow pipeline
#class CustomOptions(PipelineOptions):
#    @classmethod
#    def _add_argparse_args(cls, parser):
#        parser.add_argument('--output', dest='output', required=True, help='Output GCS file path')

# Define the Dataflow pipeline
def run():
    pipeline_options = PipelineOptions()

    with beam.Pipeline(options=pipeline_options) as p:
        # List objects (images) in the GCS folder
        input_uri = "/Users/felix/code/agdoko/deep_green_learning/features.txt"
        # Define the output GCS path
        output_uri = 'gs://dgl_cloud/Features_preprocessed/' # Adjust the GCS bucket and output folder names
        processed_images = (
            p
            | 'ReadInputFile' >> beam.io.ReadFromText(input_uri)
            | 'Window' >> beam.WindowInto(window.FixedWindows(60))
            | 'ProcessImages' >> beam.Map(process_images)
        )




        # You can write the processed data to GCS or another storage location
        _ = processed_images | "WriteResults" >> beam.io.fileio.WriteToFiles(
            output_uri,
            file_naming=beam.io.fileio.destination_prefix(),
            coder=beam.coders.BytesCoder(),
        )

if __name__ == '__main__':
    run()
