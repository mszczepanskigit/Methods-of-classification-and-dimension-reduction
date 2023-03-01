import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as imageio


def reconstruct(u, s, vt, p):
    return u[:, :p] @ np.diag(s[:p]) @ vt[:p, :]


if __name__ == "__main__":

    image_1 = np.array(imageio.imread('../Images/baboon_color.bmp', mode='RGB'))  # 500x480
    image_2 = np.array(imageio.imread('../Images/cells2.jpg', mode="RGB"))  # 420x321
    image_3 = np.array(imageio.imread('../Images/einstein.jpg', mode="RGB"))  # 186x182

    image_1_R = image_1[:, :, 0]  # R
    image_1_G = image_1[:, :, 1]  # G
    image_1_B = image_1[:, :, 2]  # B

    image_2_R = image_2[:, :, 0]
    image_2_G = image_2[:, :, 1]
    image_2_B = image_2[:, :, 2]

    image_3_R = image_3[:, :, 0]
    image_3_G = image_3[:, :, 1]
    image_3_B = image_3[:, :, 2]

    U1r, S1r, VT1r = np.linalg.svd(image_1_R, full_matrices=False)
    U1g, S1g, VT1g = np.linalg.svd(image_1_G, full_matrices=False)
    U1b, S1b, VT1b = np.linalg.svd(image_1_B, full_matrices=False)

    U2r, S2r, VT2r = np.linalg.svd(image_2_R, full_matrices=False)
    U2g, S2g, VT2g = np.linalg.svd(image_2_G, full_matrices=False)
    U2b, S2b, VT2b = np.linalg.svd(image_2_B, full_matrices=False)

    U3r, S3r, VT3r = np.linalg.svd(image_3_R, full_matrices=False)
    U3g, S3g, VT3g = np.linalg.svd(image_3_G, full_matrices=False)
    U3b, S3b, VT3b = np.linalg.svd(image_3_B, full_matrices=False)

    plt.imshow(reconstruct(U1r, S1r, VT1r, p=480))
    plt.show()

    plt.imshow(reconstruct(U2g, S2g, VT2g, p=2))
    plt.show()

    plt.imshow(reconstruct(U3r, S3r, VT3r, p=30))
    plt.show()