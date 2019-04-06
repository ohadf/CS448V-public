import cv2
from detect_landmarks import FaceLandmarkDetector
import extrapolate_vector_field as evf
from warp_image import ImageWarper
import numpy as np


def generate_face_warp_video(face1_filename, face2_filename, out_filename, n_frames=30, fps=30):
    # Read face photos. We assume they have the same size
    face1 = cv2.imread(face1_filename)
    face2 = cv2.imread(face2_filename)
    assert face1.shape == face2.shape
    img_size = face1.shape[0:2]

    # Here you will need to tie everything together: landmarks, warp fields and warping.

    out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_size[1], img_size[0]))

    warp_amounts = np.linspace(0., 1., n_frames)
    for i, warp_amount in enumerate(warp_amounts):
        # We alpha blend the original images. Replace this to produce a warping effect
        face_out = (1 - warp_amount) * face1 + warp_amount * face2

        # write video frame
        out.write(face_out.astype(np.uint8))

    out.release()


if __name__ == '__main__':
    face1_filename = './data/head1.jpg'
    face2_filename = './data/head2.jpg'
    out_filename = './data/out.mp4'

    generate_face_warp_video(face1_filename, face2_filename, out_filename)
