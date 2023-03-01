from sklearn.datasets import fetch_lfw_people

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data
Y = lfw_people.target
n_samples, h, w = lfw_people.images.shape