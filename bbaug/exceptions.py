""" Custom Exceptions """


class BaseError(Exception):
    """ Base Exception """


class InvalidMagnitude(BaseError):
    """ Error if magnitude is too large or too small """
