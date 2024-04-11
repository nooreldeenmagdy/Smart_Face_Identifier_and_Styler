import unittest
from unittest.mock import patch
import identifier

class TestIdentifier(unittest.TestCase):

		#is_image_file method


		# Valid Image File
        def test_is_image_file(self):
                file_name = "image.jpg"

                self.assertTrue(identifier.is_image_file(file_name))


        # Not Valid Image File
        def test_not_image_file(self):
                file_name = "music.mp3"

                self.assertFalse(identifier.is_image_file(file_name))

        # File with No Extension
        def test_no_extension_file(self):
                file_name = "file"

                self.assertFalse(identifier.is_image_file(file_name))






		#get_faces_data method
"""
        def test_have_directory(self):
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
				image_dir = os.path.join(BASE_DIR, 'faces')
				known_encodings = []
				known_names = []

				with open('scanned-faces', 'rt') as f:
					scanned = f.readlines()
					scanned = [i.split('\n')[0] for i in scanned]

                self.assertTrue(identifier.get_faces_data("ab"))


"""


if __name__ == '__main__':
    unittest.main()
	        

