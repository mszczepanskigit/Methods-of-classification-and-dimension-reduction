import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as imageio


def reconstruct(u, s, vt, p):
    return u[:, :p] @ np.diag(s[:p]) @ vt[:p, :]


if __name__ == "__main__":
    image = np.array(imageio.imread('../Images/nuclei1b.jpg', mode="RGB"))  # 483x465
    # image = np.array(imageio.imread('../Images/street.jpg', mode="RGB"))  # 1600x1000
    print(image.shape[0], image.shape[1])
    image_reshaped = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    print(image_reshaped.shape[0], image_reshaped.shape[1])

    U, S, VT = np.linalg.svd(image_reshaped, full_matrices=False)

    U_unshaped = U.reshape(image.shape[0], image.shape[1], image.shape[2])

    image_to_compare = np.array(imageio.imread('../Images/nuclei1b.jpg', mode="L"))
    # image = np.array(imageio.imread('../Images/street.jpg', mode="L"))

    f1 = plt.figure(1)
    plt.title("U_unshaped")
    plt.imshow(U_unshaped)

    f2 = plt.figure(2)
    plt.title("U_unshaped R")
    plt.imshow(U_unshaped[:, :, 0])

    f3 = plt.figure(3)
    plt.title("U_unshaped G")
    plt.imshow(U_unshaped[:, :, 1])

    f4 = plt.figure(4)
    plt.title("U_unshaped B")
    plt.imshow(U_unshaped[:, :, 2])

    f5 = plt.figure(5)
    plt.title("mode='L'")
    plt.imshow(image_to_compare)
    plt.show()