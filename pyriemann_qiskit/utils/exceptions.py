class InitAlgoNotImplemented(NotImplementedError):
    def __init__(self):
        NotImplementedError.__init__("Init algo method was not implemented")


class BinaryClassificationOnly(Exception):
    def __init__(self):
        Exception.__init__("Only binary classification \
                            is currently supported.")


class IncorrectRepsParameter(ValueError):
    def __init__(self, reps):
        Exception.__init__("Parameter reps must be superior \
                            or equal to 1 (Got %d)" % reps)
