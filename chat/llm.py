# from chat.config import Configuration


# class LLMClient:
#     """
#     This class is to communicate and interact with the LLM
#     """

#     def __init__(self):
#         config = Configuration()
#         self.client = config.client
#         self.model = config.model_name

#     def _send_request(self, messages, tools=None):
#         """
#         This function creates a connection between the LLM model and the user
#         """
#         response = self.client.chat.completions.create(
#             messages=messages, model=self.model, tools=(tools if tools
#                                                         else None))
#         return response

#     def _parse_response(self, response):
#         """
#         This function parses and translates the output of the LLM
#         """
#         return response.choices[0].message

#     def generate(self, messages, tools=None):
#         """
#         This function is to generate a response from the LLM to the user's
#         prompt
#         """
#         try:
#             response = self._send_request(messages, tools=tools)
#             assistant_message = self._parse_response(response)
#             return assistant_message

#         except Exception as e:
#             print(f"LLM Error: {e}")
#             raise

#     def _handle_error(self, error):
#         """
#         This function is to handle any errors that may occur when
#         interacting with the LLM or the user, such as insufficient tokens,
#         connection timeout, etc.
#         """
#         print(f"Error: {error}")

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
            messages=messages, model=self.model, tools=(tools if tools
                                                        else None))
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

        CHANGED: added a fallback retry. Some tool-calling models, when
        asked (via a system prompt) to answer in structured JSON while
        real tools are also available, will occasionally invent a fake
        tool call (e.g. named "JSON") to wrap that structured answer in,
        instead of just returning it as plain message content. That
        fake tool isn't registered, so Groq's API rejects the request
        server-side with a 400 error -- and because the rejection
        happens inside the API call itself, we never get a normal
        response object back to recover the model's intended answer
        from. The only reliable fix is to retry the identical
        conversation with tools disabled: with nothing to call, the
        model is forced to answer in plain text instead.

        This only fires when tools were actually offered on the failed
        call (retrying a tools=None call with tools=None again would be
        pointless) and the error looks like this specific failure mode
        -- matched loosely on the API's own wording, since the exact
        invented tool name varies by model/run (it isn't always "JSON").
        Any other kind of LLM error still propagates normally.
        """
        try:
            response = self._send_request(messages, tools=tools)
            return self._parse_response(response)

        except Exception as e:
            if tools and self._is_tool_hallucination_error(e):
                print(f"LLM Error (retrying without tools): {e}")
                try:
                    response = self._send_request(messages, tools=None)
                    return self._parse_response(response)
                except Exception as retry_error:
                    print(f"LLM Error (retry without tools also failed): "
                          f"{retry_error}")
                    raise

            print(f"LLM Error: {e}")
            raise

    def _is_tool_hallucination_error(self, error):
        """
        Loosely detect the "model invented a fake tool to wrap a
        structured answer in" failure mode, without depending on the
        exact tool name it happened to invent this time.
        """
        message = str(error).lower()
        return ("tool call validation failed" in message
                or "was not in request.tools" in message)

    def _handle_error(self, error):
        """
        This function is to handle any errors that may occur when
        interacting with the LLM or the user, such as insufficient tokens,
        connection timeout, etc.
        """
        print(f"Error: {error}")
