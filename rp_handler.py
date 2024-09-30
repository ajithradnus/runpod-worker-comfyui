import os
import time
import requests
import traceback
import json
import base64
import uuid
import logging
import logging.handlers
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from requests.adapters import HTTPAdapter, Retry

BASE_URI = 'http://127.0.0.1:3000'
VOLUME_MOUNT_PATH = '/runpod-volume'
LOG_FILE = 'comfyui-worker.log'
TIMEOUT = 600
LOG_LEVEL = 'INFO'

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
rp_logger = RunPodLogger()

# Input schema to validate incoming requests
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "workflow": {
            "type": "object",
            "description": "The workflow to execute"
        }
    },
    "required": ["workflow"]
}

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

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                rp_logger.info('Service not ready yet. Retrying...')
        except Exception as err:
            rp_logger.error(f'Error: {err}')

        time.sleep(0.2)


def send_post_request(endpoint, payload):
    return session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )

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

        # Extract workflow from validated input
        workflow = validated_input['validated_input']['workflow']

        rp_logger.info('Processing workflow...', job_id)

        # Prepare the request to ComfyUI
        response = send_post_request('prompt', {'prompt': workflow})

        if response.status_code == 200:
            resp_json = response.json()
            prompt_id = resp_json['prompt_id']
            rp_logger.info(f'Prompt queued successfully: {prompt_id}', job_id)

            # Wait for completion
            retries = 0
            while True:
                if retries == 0 or retries % 15 == 0:
                    rp_logger.info(f'Getting status of prompt: {prompt_id}', job_id)

                r = session.get(f'{BASE_URI}/history/{prompt_id}', timeout=TIMEOUT)
                resp_json = r.json()

                if r.status_code == 200 and len(resp_json):
                    break

                time.sleep(0.2)
                retries += 1

            status = resp_json[prompt_id]['status']

            if status['status_str'] == 'success' and status['completed']:
                # Job was processed successfully
                outputs = resp_json[prompt_id]['outputs']

                if len(outputs):
                    rp_logger.info(f'Images generated successfully for prompt: {prompt_id}', job_id)
                    images = []

                    for image_filename in outputs:
                        filename = image_filename['filename']
                        image_path = f'{VOLUME_MOUNT_PATH}/ComfyUI/output/{filename}'

                        with open(image_path, 'rb') as image_file:
                            images.append(base64.b64encode(image_file.read()).decode('utf-8'))

                        rp_logger.info(f'Deleting output file: {image_path}', job_id)
                        os.remove(image_path)

                    return {
                        'images': images
                    }
                else:
                    raise RuntimeError(f'No output found for prompt id: {prompt_id}')
            else:
                # Job did not process successfully
                return {
                    'error': f'Job did not process successfully for prompt id: {prompt_id}',
                    'status': status
                }

        else:
            try:
                response_content = response.json()
            except Exception:
                response_content = str(response.content)

            rp_logger.error(f'HTTP Status code: {response.status_code}', job_id)
            rp_logger.error(response_content, job_id)

            return {
                'error': f'HTTP status code: {response.status_code}',
                'output': response_content
            }

    except Exception as e:
        rp_logger.error(f'An exception was raised: {e}', job_id)

        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }


if __name__ == '__main__':
    # Setup log file
    logging.getLogger().setLevel(LOG_LEVEL)
    log_handler = logging.handlers.WatchedFileHandler(f'{VOLUME_MOUNT_PATH}/{LOG_FILE}')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    log_handler.setFormatter(formatter)
    logging.getLogger().addHandler(log_handler)

    # Set up RunPod logger
    rp_logger.set_level(LOG_LEVEL)

    wait_for_service(url=f'{BASE_URI}/system_stats')
    rp_logger.info('ComfyUI API is ready')
    rp_logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
