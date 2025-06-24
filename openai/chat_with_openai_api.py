import os
from openai import OpenAI

class MyOpenAI():
    def __init__(self, model_name = "gpt-4o-mini"):
        self.model_name = model_name
        self.key_name = "OPENAI_API_KEY"

    # should import OPENAI_API_KEY first
    def check_key(self):
        open_api_key = os.getenv(self.key_name)
        if open_api_key is None:
            print(f"{self.key_name} is not set. Try get/set from local file.")
            return False
        return True


    def generate(self, input: str,):
        assert self.check_key()
        model = OpenAI()
        completion = model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input},
            ],
        )
        response = completion.choices[0].message.content
        return response

