from enum import Enum


class IntentEnums(Enum):
    GREETINGS = "greeting"
    GRATITUDE = "gratitude"
    ASKING = "asking_mental_health_question"
    GOODBYE = "goodbye"
    OUT_OF_SCOPE = "out_of_scope"
