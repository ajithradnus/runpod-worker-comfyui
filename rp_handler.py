import os
import json
import base64
import time
from runpod.serverless.utils.rp_validator import validate
from comfyui import api as comfy_api

# Define the input schema for validation
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "workflow": {
            "type": "object",
            "additionalProperties": True
        }
    },
    "required": ["workflow"]
}

def handler(event):
    try:
        # Validate the input using the schema
        validated_input = validate(event.get('input', {}), INPUT_SCHEMA)

        if 'errors' in validated_input:
            return {
                'error': '\n'.join(validated_input['errors'])
            }

        # Extract the workflow from the validated input
        workflow = validated_input.get('workflow')

        # Start timer for timeout
        start_time = time.time()

        # Process the workflow with ComfyUI
        result = comfy_api.execute_workflow(workflow)

        # Check if processing exceeds timeout limit of 200 seconds
        elapsed_time = time.time() - start_time
        if elapsed_time > 200:
            return {
                "error": "Workflow execution time exceeded 200 seconds and was canceled."
            }

        # Convert the generated image to a base64 string
        if "output" in result and result["output"].get("images"):
            image = result["output"]["images"][0]
            with open(image, "rb") as img_file:
                base64_string = base64.b64encode(img_file.read()).decode('utf-8')

            # Remove the generated image to save storage space
            os.remove(image)

            return {
                "base64_image": base64_string
            }

        return {
            "error": "No image generated."
        }

    except Exception as e:
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    event = {
        "input": {
            "workflow": {
                # Your JSON workflow here
            }
        }
    }
    print(handler(event))
