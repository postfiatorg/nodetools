import threading
import unittest
from nodetools.ai.openai import OpenAIRequestTool
from nodetools.utilities.db_manager import DBConnectionManager
from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities
from nodetools.utilities.credentials import CredentialManager
import getpass

class TestSingletons(unittest.TestCase):
    def setUp(self):
        # Initialize CredentialManager first as other singletons need it
        password = getpass.getpass("Enter your password: ")
        self.cred_manager = CredentialManager(password=password)
        self.singleton_classes = [
            (OpenAIRequestTool, "OpenAIRequestTool"),
            (DBConnectionManager, "DBConnectionManager"),
            (GenericPFTUtilities, "GenericPFTUtilities")
        ]

    def test_singleton_threading(self):
        for singleton_class, class_name in self.singleton_classes:
            with self.subTest(F"Testing {class_name} singleton pattern"):
                instances = []
                def create_instance():
                    instance = singleton_class()
                    instances.append(instance)

                # Create and run multiple threads
                threads = [threading.Thread(target=create_instance) for _ in range(5)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                # Verify all instances are the same object
                first_instance = instances[0]
                for instance in instances[1:]:
                    self.assertIs(instance, first_instance)

if __name__ == '__main__':
    unittest.main()