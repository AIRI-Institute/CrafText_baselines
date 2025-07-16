import os
import time

from openai import OpenAI
from yandex_cloud_ml_sdk import YCloudML


class InputAPI:
    """API that reads user input from stdin."""

    @staticmethod
    def api_request(prompt):
        print(prompt)
        user_input = input('Write your action: ')
        return user_input


class DummyAPI:
    """Dummy API that returns a static response."""

    @staticmethod
    def api_request(prompt):
        return "step X: LEFT"


class OpenAIAPI:
    """Handles OpenAI API calls."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("Missing environment variable: OPENAI_API_KEY")

    def api_request(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class YandexAPI:
    """Handles Yandex API calls."""

    def __init__(self):
        self.sdk = self.get_yandex_sdk()

    @staticmethod
    def get_yandex_sdk():
        folder_id = os.getenv("YANDEX_FOLDER_API_KEY")
        auth = os.getenv("YANDEX_AUTH_API_KEY")
        if not folder_id or not auth:
            raise ValueError("Missing environment variables: YANDEX_FOLDER_API_KEY and/or YANDEX_AUTH_API_KEY")
        return YCloudML(folder_id=folder_id, auth=auth)

    def api_request(self, prompt):
        model = self.sdk.models.completions("llama").configure(temperature=0.5)
        operation = model.run_deferred([{"role": "system", "text": prompt}])
        while operation.get_status().is_running:
            time.sleep(5)
        result = operation.get_result()
        return result[0].text if result and hasattr(result[0], "text") else ""


class APIProvider:
    """Aggregates all available APIs into a unified interface."""

    def __init__(self):
        self.apis = {
            "dummy": DummyAPI(),
            "input": InputAPI(),
            "openai": OpenAIAPI(),
            "yandex": YandexAPI(),
        }

    def api_request(self, prompt, api="dummy"):
        if api not in self.apis:
            raise ValueError("Invalid API type")
        return self.apis[api].api_request(prompt)


def main():
    # Example usage
    api_provider = APIProvider()
    print(api_provider.api_request("Translate this", api="dummy"))  # Dummy API
    print(api_provider.api_request("Tell me a joke", api="openai"))  # OpenAI API (GPT-4o mini)
    # print(api_provider.api_request("Translate to Russian", api="yandex"))  # Yandex API


if __name__ == '__main__':
    main()
