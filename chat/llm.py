from chat.config import Configuration


class LLMClient:
    """
    This class is to communicate and interact with the LLM
    """

    def __init__(self):
        config = Configuration()
        self.client = config.client
        self.model = config.model_name

    def _send_request(self, messages, tools=None):
        """
        This function creates a connection between the LLM model and the user
        """
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            tools=tools
        )
        return response

    def _parse_response(self, response):
        """
        This function parses and translates the output of the LLM
        """
        return response.choices[0].message

    def generate(self, messages, tools=None):
        """
        This function is to generate a response from the LLM to the user's
        prompt
        """
        try:
            response = self._send_request(messages, tools=tools)
            assistant_message = self._parse_response(response)
            return assistant_message

        except Exception as e:
            print(f"LLM Error: {e}")
            raise

    def _handle_error(self, error):
        """
        This function is to handle any errors that may occur when
        interacting with the LLM or the user, such as insufficient tokens,
        connection timeout, etc.
        """
        print(f"Error: {error}")
