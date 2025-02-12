import os
from dotenv import load_dotenv

class ConfigLoader:
    def __init__(self, env_path="./.env"):
        """
        Initialize the ConfigLoader and load the environment variables.
        :param env_path: Path to the .env file relative to the current directory.
        """
        self.current_directory = os.getcwd()
        self.dotenv_path = os.path.join(self.current_directory, env_path)
        self.load_env()

    def load_env(self):
        """Load the .env file."""
        if os.path.exists(self.dotenv_path):
            load_dotenv(self.dotenv_path)
        else:
            raise FileNotFoundError(f".env file not found at {self.dotenv_path}")

    def get_api_key(self, key_name):
        """
        Retrieve an API key from the environment variables.

        :param key_name: The name of the environment variable.
        :return: The value of the environment variable or None if not found.
        """
        return os.getenv(key_name)