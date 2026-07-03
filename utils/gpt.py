#Copyright 2026 Mücahit Sahin
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import re
import sys
import numpy as np
import json
import uuid
import time
import string
import openai
from dotenv import load_dotenv


def load_api_key():
    load_dotenv()
    openai.api_key = os.getenv('OPEN_AI_KEY')


def add_batch(batchfile=None, messages=[], model="gpt-5.4"):
    if batchfile == None:
        return 'Specify the batchfile to add the batch!'
    
    batch = {
        'custom_id': 'request-'+str(uuid.uuid4()),
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {
            'model': model,
            'messages': messages,
            'temperature': 0.2,
            'seed': 999
        }
    }
    
    if os.path.isfile(batchfile):
        with open(batchfile, 'a') as fp:
            fp.write('\n')
            json.dump(batch, fp)
            fp.close()
    else:
        with open(batchfile, 'w') as fp:
            json.dump(batch, fp)
            fp.close()


def ask_gpt(prompt=None, system_prompt=None, model="gpt-5.4", batchfile=None):
    client = openai.OpenAI(api_key=openai.api_key)

    messages = [
        {"role": "user", "content": prompt}
    ]

    if not system_prompt == None:
        messages.insert(0, {"role": "system", "content": system_prompt})

    
    if not batchfile == None:
        return 'For batching, run the corresponding files under /batches directory! (make sure batchfiles are created)'
    else:
        if prompt == None:
            return 'Please specify a user prompt'
        else:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.02,
                    seed=999
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error: {e} \n")
                return 'Failed to make LLM-prediction'
