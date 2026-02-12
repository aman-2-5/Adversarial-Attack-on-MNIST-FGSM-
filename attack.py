import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("models/mnist_victim_model.keras")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# Load MNIST test data
mnist = tf.keras.datasets.mnist
(_, _), (test_images, test_labels) = mnist.load_data()
test_images = test_images / 255.0

index = 12
image = test_images[index]
label = test_labels[index]

image_tensor = tf.expand_dims(tf.cast(image, tf.float32), axis=0)

def create_adversarial_pattern(input_image, input_label):
    input_label = tf.convert_to_tensor([input_label], dtype=tf.int64)

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

perturbations = create_adversarial_pattern(image_tensor, label)

epsilon = 0.15

adv_image = image_tensor + epsilon * perturbations
adv_image = tf.clip_by_value(adv_image, 0, 1)

print("True Label:", label)

# Save Original
prediction_orig = model.predict(image_tensor)
plt.imshow(image_tensor[0], cmap="gray")
plt.title(f"Original\nPrediction: {np.argmax(prediction_orig)}")
plt.axis("off")
plt.savefig("screenshots/original.png")
plt.close()

# Save Adversarial
prediction_adv = model.predict(adv_image)
plt.imshow(adv_image[0], cmap="gray")
plt.title(f"Adversarial\nPrediction: {np.argmax(prediction_adv)}")
plt.axis("off")
plt.savefig("screenshots/adversarial.png")
plt.close()

# Save Perturbation
plt.imshow(perturbations[0], cmap="gray")
plt.title("Perturbation")
plt.axis("off")
plt.savefig("screenshots/perturbation.png")
plt.close()

print("Images saved successfully.")
