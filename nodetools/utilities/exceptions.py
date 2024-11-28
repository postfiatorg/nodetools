# GOOGLE DOCUMENT EXCEPTIONS

class GoogleDocNotFoundException(Exception):
    """ This exception is raised when the Google Doc is not found """
    def __init__(self, google_url):
        self.google_url = google_url
        super().__init__(f"Google Doc not found: {google_url}")

class InvalidGoogleDocException(Exception):
    """ This exception is raised when the google doc is not valid """
    def __init__(self, google_url):
        self.google_url = google_url
        super().__init__(f"Invalid Google Doc URL: {google_url}")

class GoogleDocDoesNotContainXrpAddressException(Exception):
    """ This exception is raised when the google doc does not contain the XRP address """
    def __init__(self, xrp_address):
        self.xrp_address = xrp_address
        super().__init__(f"Google Doc does not contain expected XRP address: {xrp_address}")

class GoogleDocIsNotFundedException(Exception):
    """ This exception is raised when the google doc's XRP address is not funded """
    def __init__(self, google_url):
        self.google_url = google_url
        super().__init__(f"Google Doc's XRP address is not funded: {google_url}")

class GoogleDocIsNotSharedException(Exception):
    """ This exception is raised when the google doc is not shared """
    def __init__(self, google_url):
        self.google_url = google_url
        super().__init__(f"Google Doc is not shared: {google_url}")

# XRP ACCOUNT EXCEPTIONS

class XRPAccountNotFoundException(Exception):
    """ This exception is raised when the XRP account is not found """
    def __init__(self, address):
        self.address = address
        super().__init__(f"XRP account not found: {address}")

class InsufficientXrpBalanceException(Exception):
    """ This exception is raised when the user does not have enough XRP balance """
    def __init__(self, xrp_address):
        self.xrp_address = xrp_address
        super().__init__(f"Insufficient XRP balance: {xrp_address}")

# MEMO EXCEPTIONS

class HandshakeRequiredException(Exception):
    """ This exception is raised when a handshake is required for an encrypted memo to be sent"""
    def __init__(self, destination):
        self.destination = destination
        super().__init__(f"Cannot encrypt message: no handshake received from {destination}")
