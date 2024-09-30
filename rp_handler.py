import time
import uuid
import requests
import traceback
import json
import base64
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from requests.adapters import HTTPAdapter, Retry
from schemas.input import INPUT_SCHEMA  # Adjust according to your project structure
from comfy_api_simplified import ComfyApiWrapper, ComfyWorkflowWrapper

BASE_URI = 'http://127.0.0.1:3000'
VOLUME_MOUNT_PATH = '/runpod-volume'
TIMEOUT = 200  # Changed timeout to 200 seconds

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
logger = RunPodLogger()

# ---------------------------------------------------------------------------- #
#                               ComfyUI Functions                              #
# ---------------------------------------------------------------------------- #

def wait_for_service(url):
    retries = 0
    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:
                logger.info('Service not ready yet. Retrying...')
        except Exception as err:
            logger.error(f'Error: {err}')
        time.sleep(0.2)

def send_post_request(payload):
    api = ComfyApiWrapper(BASE_URI)

    # Queue and wait for images
    def new_queue_and_wait_images(self, prompt: dict, output_node_title: str) -> dict:
        prompt_id = self.queue_prompt_and_wait(prompt)
        history = self.get_history(prompt_id)
        image_node_id = prompt.get_node_id(output_node_title)
        images = history[prompt_id]["outputs"][image_node_id]["images"]
        return {
            image["filename"]: self.get_image(image["filename"], image["subfolder"], image["type"])
            for image in images
        }

    api.queue_and_wait_images = new_queue_and_wait_images.__get__(api, ComfyApiWrapper)

    results = api.queue_and_wait_images(payload, "Image Save")

    # Return the first image found as base64
    for filename, image_bytes in results.items():
        return base64.b64encode(image_bytes).decode("utf-8")

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    job_id = event['id']

    try:
        validated_input = validate(event['input'], INPUT_SCHEMA)

        if 'errors' in validated_input:
            return {
                'error': '\n'.join(validated_input['errors'])
            }

        payload = validated_input['validated_input']
        workflow_name = payload['workflow']
        payload = payload['payload']

        logger.info(f'Workflow: {workflow_name}', job_id)

        logger.debug('Queuing prompt')
        response = send_post_request(payload)

        return {
            'image': response  # Returning base64 encoded image
        }
    
    except Exception as e:
        logger.error(f'An exception was raised: {e}', job_id)

        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }

if __name__ == '__main__':
    wait_for_service(url=f'{BASE_URI}/system_stats')
    logger.info('ComfyUI API is ready')
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
