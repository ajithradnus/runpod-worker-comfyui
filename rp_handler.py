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
from schemas.input import INPUT_SCHEMA
import signal

BASE_URI = 'http://127.0.0.1:3000'
VOLUME_MOUNT_PATH = '/runpod-volume'
LOG_FILE = 'comfyui-worker.log'
TIMEOUT = 600
LOG_LEVEL = 'INFO'

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
rp_logger = RunPodLogger()

# ---------------------------------------------------------------------------- #
#                               ComfyUI Functions                              #
# ---------------------------------------------------------------------------- #

def wait_for_service(url, timeout=200):
    retries = 0
    start_time = time.time()

    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Check if the timeout is exceeded
            if time.time() - start_time > timeout:
                raise TimeoutError("Service did not start within the timeout period.")

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                rp_logger.info('Service not ready yet. Retrying...')
        except Exception as err:
            rp_logger.error(f'Error: {err}')

        time.sleep(0.2)


def send_get_request(endpoint):
    return session.get(
        url=f'{BASE_URI}/{endpoint}',
        timeout=TIMEOUT
    )


def send_post_request(endpoint, payload):
    return session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )


def create_unique_filename_prefix(payload):
    """Generate a unique filename prefix for the payload."""
    unique_prefix = str(uuid.uuid4())
    payload['prompt']['filename_prefix'] = unique_prefix


def get_workflow_payload(workflow_name, payload):
    """Generate workflow payload based on workflow_name and input payload."""
    if workflow_name == 'txt2img':
        return get_txt2img_payload(payload)
    elif workflow_name == 'img2img':
        return get_img2img_payload(payload)
    # Add more workflows as needed
    else:
        raise ValueError(f"Unsupported workflow: {workflow_name}")


def get_txt2img_payload(payload):
    """Generate payload for txt2img workflow."""
    return {
        'prompt': {
            'prompt_text': payload.get('prompt_text', ''),
            'steps': payload.get('steps', 20),
            'sampler_name': payload.get('sampler_name', 'Euler a'),
            'width': payload.get('width', 512),
            'height': payload.get('height', 512),
            # Add more default parameters as needed
        }
    }


def get_img2img_payload(payload):
    """Generate payload for img2img workflow."""
    return {
        'prompt': {
            'prompt_text': payload.get('prompt_text', ''),
            'init_image': payload.get('init_image', ''),
            'strength': payload.get('strength', 0.75),
            'steps': payload.get('steps', 20),
            'sampler_name': payload.get('sampler_name', 'Euler a'),
            'width': payload.get('width', 512),
            'height': payload.get('height', 512),
            # Add more default parameters as needed
        }
    }

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Job execution exceeded the timeout limit.")

signal.signal(signal.SIGALRM, timeout_handler)

def handler(event):
    job_id = event['id']
    timeout_seconds = 200  # Set timeout duration

    signal.alarm(timeout_seconds)  # Set the alarm for the handler

    try:
        validated_input = validate(event['input'], INPUT_SCHEMA)

        if 'errors' in validated_input:
            return {
                'error': '\n'.join(validated_input['errors'])
            }

        payload = validated_input['validated_input']
        workflow_name = payload['workflow']
        payload = payload['payload']

        if workflow_name == 'default':
            workflow_name = 'txt2img'

        rp_logger.info(f'Workflow: {workflow_name}', job_id)

        if workflow_name != 'custom':
            try:
                payload = get_workflow_payload(workflow_name, payload)
            except Exception as e:
                rp_logger.error(f'Unable to load workflow payload for: {workflow_name}', job_id)
                raise

        create_unique_filename_prefix(payload)
        rp_logger.debug('Queuing prompt', job_id)

        queue_response = send_post_request(
            'prompt',
            {
                'prompt': payload
            }
        )

        if queue_response.status_code == 200:
            resp_json = queue_response.json()
            prompt_id = resp_json['prompt_id']
            rp_logger.info(f'Prompt queued successfully: {prompt_id}', job_id)
            retries = 0

            while True:
                # Only log every 15 retries so the logs don't get spammed
                if retries == 0 or retries % 15 == 0:
                    rp_logger.info(f'Getting status of prompt: {prompt_id}', job_id)

                r = send_get_request(f'history/{prompt_id}')
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
                    image_filenames = get_filenames(outputs)
                    images = []

                    for image_filename in image_filenames:
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
                for message in status['messages']:
                    key, value = message

                    if key == 'execution_error':
                        if 'node_type' in value and 'exception_message' in value:
                            node_type = value['node_type']
                            exception_message = value['exception_message']
                            raise RuntimeError(f'{node_type}: {exception_message}')
                        else:
                            error_msg = f'Job did not process successfully for prompt_id: {prompt_id}'
                            logging.error(error_msg)
                            logging.info(f'{job_id}: Response JSON: {resp_json}')
                            raise RuntimeError(error_msg)

        else:
            try:
                queue_response_content = queue_response.json()
            except Exception as e:
                queue_response_content = str(queue_response.content)

            rp_logger.error(f'HTTP Status code: {queue_response.status_code}', job_id)
            rp_logger.error(queue_response_content, job_id)

            return {
                'error': f'HTTP status code: {queue_response.status_code}',
                'output': queue_response_content
            }
    except TimeoutException:
        rp_logger.error(f"Job {job_id} exceeded the timeout of {timeout_seconds} seconds")
        return {
            'error': f"Job exceeded the timeout of {timeout_seconds} seconds.",
            'refresh_worker': True
        }
    except Exception as e:
        rp_logger.error(f'An exception was raised: {e}', job_id)

        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }
    finally:
        signal.alarm(0)  # Cancel the alarm


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
