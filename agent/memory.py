class Memory:
    """
    This class is used to store the memory of the agent,
    what messages have been given and what actions have been taken
    """

    def __init__(self, user="New User"):
        self.user = user
        self.messages = []

    def add(self, message):
        """
        Adding the message to the history of the chat
        """
        self.messages.append(message)

    def get_messages(self):
        """
        Returns the message history of the user and the model
        """
        return self.messages

    def clear_history(self):
        """
        This function clears the history of the chat
        """
        self.messages.clear()
