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
TIMEOUT = 200  # Timeout for API calls in seconds
LOG_LEVEL = 'INFO'

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
rp_logger = RunPodLogger()

# Function to wait for ComfyUI service to be ready
def wait_for_service(url):
    retries = 0
    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:
                rp_logger.info('Service not ready yet. Retrying...')
        except Exception as err:
            rp_logger.error(f'Error: {err}')
        time.sleep(0.2)

# Send GET request with timeout
def send_get_request(endpoint):
    return session.get(
        url=f'{BASE_URI}/{endpoint}',
        timeout=TIMEOUT
    )

# Send POST request with timeout
def send_post_request(endpoint, payload):
    return session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )

# Helper functions for constructing the payload
def get_workflow_payload(workflow_name, payload):
    with open(f'/workflows/{workflow_name}.json', 'r') as json_file:
        workflow = json.load(json_file)

    if workflow_name == 'txt2img':
        workflow = get_txt2img_payload(workflow, payload)

    return workflow

def get_txt2img_payload(workflow, payload):
    workflow["3"]["inputs"]["seed"] = payload["seed"]
    workflow["3"]["inputs"]["steps"] = payload["steps"]
    workflow["3"]["inputs"]["cfg"] = payload["cfg_scale"]
    workflow["3"]["inputs"]["sampler_name"] = payload["sampler_name"]
    workflow["4"]["inputs"]["ckpt_name"] = payload["ckpt_name"]
    workflow["5"]["inputs"]["batch_size"] = payload["batch_size"]
    workflow["5"]["inputs"]["width"] = payload["width"]
    workflow["5"]["inputs"]["height"] = payload["height"]
    workflow["6"]["inputs"]["text"] = payload["prompt"]
    workflow["7"]["inputs"]["text"] = payload["negative_prompt"]
    return workflow

# Main handler function
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

        if workflow_name == 'default':
            workflow_name = 'txt2img'

        rp_logger.info(f'Workflow: {workflow_name}', job_id)

        if workflow_name != 'custom':
            try:
                payload = get_workflow_payload(workflow_name, payload)
            except Exception as e:
                rp_logger.error(f'Unable to load workflow payload for: {workflow_name}', job_id)
                raise

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

            # Polling for result with timeout
            start_time = time.time()
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > TIMEOUT:
                    rp_logger.error(f'Timeout reached for prompt ID: {prompt_id}', job_id)
                    return {
                        'error': 'Request timed out after 200 seconds.'
                    }

                r = send_get_request(f'history/{prompt_id}')
                resp_json = r.json()

                if r.status_code == 200 and len(resp_json):
                    break

                time.sleep(0.2)

            status = resp_json[prompt_id]['status']

            if status['status_str'] == 'success' and status['completed']:
                # Job was processed successfully
                outputs = resp_json[prompt_id]['outputs']
                if outputs:
                    rp_logger.info(f'Images generated successfully for prompt: {prompt_id}', job_id)

                    # Convert images to base64
                    images = []
                    for output in outputs:
                        filename = output['images'][0]['filename']
                        image_path = f'{VOLUME_MOUNT_PATH}/ComfyUI/output/{filename}'

                        with open(image_path, 'rb') as image_file:
                            images.append(base64.b64encode(image_file.read()).decode('utf-8'))

                        # Delete the output image to save space
                        rp_logger.info(f'Deleting output file: {image_path}', job_id)
                        os.remove(image_path)

                    return {
                        'images': images
                    }
                else:
                    raise RuntimeError(f'No output found for prompt id: {prompt_id}')
            else:
                # Handle errors from the job
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
