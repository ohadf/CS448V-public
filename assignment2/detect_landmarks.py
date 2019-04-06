import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt


class FaceLandmarkDetector:
    """Detect face landmarks using dlib"""
    PREDICTOR_PATH = './data/shape_predictor_68_face_landmarks.dat'

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)

    @staticmethod
    def plot_landmarks(img, landmarks, show_plot=True):
        # convert from BGR (opencv) to RGB (matplotlib)
        img = img[..., ::-1]

        # display face image
        plt.imshow(img)

        # plot landmarks
        plt.scatter(x=landmarks[:, 0], y=landmarks[:, 1], c='r', s=20, edgecolors='k')

        if show_plot:
            plt.show()

    def predict(self, img):
        n_landmarks = 68
        img_size = img.shape[0:2]
        # Generate 68x2 array with random landmark locations. This should be replaced with detected landmarks.
        # It is ok to assume each image contains exactly one face (and raise an error otherwise).
        return np.hstack((np.random.randint(img_size[1], size=(n_landmarks, 1)), np.random.randint(img_size[0], size=(n_landmarks, 1))))


if __name__ == '__main__':
    face_filename = './data/head1.jpg'
    face = cv2.imread(face_filename)

    d = FaceLandmarkDetector()
    landmarks = d.predict(face)

    d.plot_landmarks(face, landmarks)




