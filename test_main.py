import unittest
from unittest.mock import patch
from main import HomeScreen

class TestMain(unittest.TestCase):


    # Once BeforeAll
    @classmethod
    def setUpClass(cls):
        print('setupClass')
        
    # Once AfterALL
    @classmethod
    def tearDownClass(cls):
        print('teardownClass')

    # Everytime Before
    def setUp(self):
        print('setUp')
        
        self.hm1 = HomeScreen()
        
    # Everytime After
    def tearDown(self):
        print('tearDown\n')



    #update camera method
    def test_update_cam_on(self):
        print('test_cam_on')

        self.assertTrue(self.hm1.update_cam(dt=0.5))



    def test_update_cam_off(self):
        print('test_cam_off')
        self.hm1.not_testing = False
        self.hm1.capture.release()
        self.assertFalse(self.hm1.update_cam(dt=0.5))




    #glasses method
    def test_glasses_withFace(self):
        print('glasses have face')
        self.hm1.glasses(dt=0.5)
        self.assertTrue(self.hm1.have_face)

    def test_glasses_withnoFace(self):

        print('glasses have no face')
        self.hm1.glasses(dt=0.5)
        self.assertFalse(self.hm1.have_face)


        #detect face method

    def test_face_detected(self):

        print('Detected Face')
        self.hm1.detect_faces(dt=0.5)
        self.assertTrue(self.hm1.have_face)

    def test_no_face_detected(self):

        print('No Face Detected')
        self.hm1.detect_faces(dt=0.5)
        self.assertFalse(self.hm1.have_face)



	#identify face method
    def test_face_identified(self):

        print('Saved Face')
        self.hm1.identify_faces(dt=0.5)
        self.assertTrue(self.hm1.saved_face)

    def test_no_face_identified(self):

        print('No Face Identified')
        self.hm1.identify_faces(dt=0.5)
        self.assertFalse(self.hm1.saved_face)

    def test_no_face_in_identified(self):

        print('No Face Identified')
        self.hm1.identify_faces(dt=0.5)
        self.assertFalse(self.hm1.have_face)




if __name__ == '__main__':
    unittest.main()
	        