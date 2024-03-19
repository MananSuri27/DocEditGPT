import base64
import requests
import re
import pandas as pd 
from tqdm import tqdm
import os


document_edit_introduction_prompt = """
Task Description:
Your task is to develop an HTML+CSS document that recreates a given document image with desired changes based on a provided command. The command consists of an action, a component, initial state, and final state, allowing for various editing operations on specific components within the document.

Inputs:
1. Document Image: A document image with a red bounding box indicating the region to be edited.
2. Command: The command format is as follows: action_para [ ] component_para [ ] initial_state [ ] final_state [ ]. The taxonomy of actions includes Add, Delete, Copy, Move, Replace, Split, Merge, and Modify. For example, "Add paragraph before heading Lorem Ipsum" or "Delete table in section 2".

Outputs:
1. HTML+CSS Document: Generate an HTML+CSS document that reflects the desired changes specified in the command. Ensure that the edited components are accurately modified from the initial state to the final state as specified in the command.

Please ensure that your output HTML+CSS document maintains the structural integrity of the original document while incorporating the specified changes accurately.
"""


reformulation_prompt = """
Given a user request, and a corresponding command for document editing, use the user request to improve the command and make it more specific.
The command is of the form: action_para [ ] component_para [ ] intial_state [ ] final_state [ ]. Where action_para is performed on component_para, changing it from intial_state to final_state.

When you are modifying the command, do not modify the action_para and component_para. Just make intial_state and final_state more specific. If no changes are needed in intial_state and final_state do not change them.
Purpose of this operation is to generate a better command for a document editing software. 
"""

html_instruction_prompt = """
Create an HTML document replicating the provided image. 
Guidelines for generating HTML document from image:
1. Include details such as full text, bullets, tables, and layout. 
2. Use appropriate tags for lists, bullets, paragraphs and tables.
3. Preserve the relative size, alignment and placement of elements on the page.
4. Use inline CSS (i.e. css defined using style attribute in tags) for denoting the styles. For each element, consider the alignment, colour and other relevant attributes.
5. Utilize flexbox for layouts.
6. Wherever images or visual elements are involved, use placeholder boxes with appropriately scaled sizes.
7. Pay special attention to how page numbers, headings footers are aligned (left/right/center), position of text with respect to images (below/above/to the side).
"""

# OpenAI API Key
api_key = ""


def cleanHTML(llm_response):
  pattern = re.compile(r'<!DOCTYPE html>(.*?)</html>', re.DOTALL)
  match = pattern.search(llm_response)
  cleaned_html = match.group(1).strip() if match else None

  return f"<!DOCTYPE html>{cleaned_html}</html>"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def reformulate_command(user_request, command):
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4",
    "messages": [
      {
        "role": "user",
        "content": 
         [
          {
            "type": "text",
            "text": f"{reformulation_prompt} \n User Request:{user_request} \n Command:{command} \n Optimised Command:"
          },

        ]
      }
    ],
    "max_tokens": 200,
    "temperature": 0
  }


  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  new_command = response.json()["choices"][0]["message"]["content"]

  return new_command
  
def document_editin(image_path, command = None):
  # Getting the base64 string
  base64_image = encode_image(image_path)

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": 
         [
           {
            "type": "text",
            "text": document_edit_introduction_prompt
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          },
          {
            "type": "text",
            "text": html_instruction_prompt
          },
          {
            "type": "text",
            "text": f"The task is to edit the document. Pay attention to the red box in the image. This box is where the change needs to be performed. The change is described by the command: [{command}], where action_para represents what action needs to be performed, component_para refers to what component needs to be changed. the change needs to be from initial_state--> final_state. Donot draw the red box. Include all text content in the document."
          }
        ]
      }
    ],
    "max_tokens": 4000,
    "temperature": 0
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  html = response.json()["choices"][0]["message"]["content"]
  clean_html = cleanHTML(html) 

  return cleanHTML(response.json()["choices"][0]["message"]["content"])



      

