# WIDERSNET CNN MODEL

import os
import sys
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import matplotlib
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import confusion_matrix, classification_report
from scipy.optimize import differential_evolution

# Set environment variable to disable OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Use a non-interactive backend for matplotlib
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Flask configuration
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Model configuration
MODEL_SAVE_PATH = 'model/cifar10_model1.keras'
model_input_shape = (32, 32, 3)
num_classes = 10

def create_widersnet_model(input_shape=(32, 32, 3), num_classes=10):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # First Wide Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second Wide Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Third Wide Convolutional Block
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Global Average Pooling and Dense Layers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

# Load or create the model
def load_model_fn():
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            logging.info("Model loaded from disk.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            model = create_widersnet_model(input_shape=model_input_shape, num_classes=num_classes)
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    else:
        model = create_widersnet_model(input_shape=model_input_shape, num_classes=num_classes)
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        logging.info("New model created and compiled.")
    return model

model = load_model_fn()

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load and preprocess image
def load_and_preprocess_image(filepath, target_shape=(32, 32, 3)):
    try:
        with Image.open(filepath) as img:
            img = img.convert('RGB').resize(target_shape[:2])
            return np.array(img) / 255.0
    except UnidentifiedImageError:
        logging.error(f"Unidentified or corrupted image: {filepath}")
    except Exception as e:
        logging.error(f"Error processing image {filepath}: {e}")
    return None

# Load CIFAR-10 data
def load_cifar10_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    return (train_images, train_labels), (test_images, test_labels)

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        logging.info("Rendering train.html")
        return render_template('train.html')

    try:
        logging.info("Starting model training...")

        # Load CIFAR-10 dataset
        (train_images, train_labels), (val_images, val_labels) = load_cifar10_data()

        # Convert the data to tensors and batch them
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(50000).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)

        # Use the WideResNet model
        model = create_widersnet_model(input_shape=(32, 32, 3), num_classes=10)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

        logging.info("Model compiled.")

        # Implement Early Stopping
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            callbacks=[early_stop]
        )

        # Save the model
        model.save(MODEL_SAVE_PATH)
        logging.info(f"Model trained and saved to {MODEL_SAVE_PATH}")

        # Get validation labels and predictions
        val_predictions = model.predict(val_images)
        val_predicted_labels = np.argmax(val_predictions, axis=1)

        # Confusion matrix and classification report
        conf_matrix = confusion_matrix(val_labels, val_predicted_labels)
        class_report = classification_report(val_labels, val_predicted_labels)
        logging.info("Confusion matrix and classification report generated.")

        # Extract metrics from history
        training_loss = history.history['loss'][-1]
        validation_loss = history.history['val_loss'][-1]
        training_accuracy = history.history['accuracy'][-1]
        validation_accuracy = history.history['val_accuracy'][-1]

        logging.info("Training metrics extracted.")

        return render_template(
            'train_result.html',
            training_loss=training_loss,
            validation_loss=validation_loss,
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            confusion_matrix=conf_matrix.tolist(),  # Convert to list for JSON serialization
            classification_report=class_report.replace('\n', '<br>')
        )
    except Exception as e:
        logging.error(f"Training error: {e}")
        return f"Error during training: {e}", 500





def record_predictions_to_csv(filepath, image_index, confidence_score, predicted_label, true_label):
    import os, csv

    # Check if file exists
    file_exists = os.path.isfile(filepath)

    # Open the file in append mode
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file does not exist
        if not file_exists:
            writer.writerow(["Image Index", "True Label", "Predicted Label", "Confidence Score"])

        # Write the prediction data
        writer.writerow([image_index, true_label, predicted_label, confidence_score])


@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    if request.method == 'GET':
        logging.info("Rendering evaluate.html")
        return render_template('evaluate.html')

    try:
        logging.info("Starting model evaluation...")

        # Load CIFAR-10 test data
        _, (test_images, test_labels) = load_cifar10_data()

        # Convert to batches
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

        # Load the trained model
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)

        # Evaluate the model
        loss, accuracy = model.evaluate(test_dataset, verbose=1)
        logging.info(f"Evaluation - Loss: {loss}, Accuracy: {accuracy}")

        # Record only test dataset predictions to CSV
        csv_filepath = "cifar_predictions1.csv"
        if os.path.exists(csv_filepath):
            os.remove(csv_filepath)  # Remove existing file to ensure fresh data

        image_index = 0
        for batch_images, batch_labels in test_dataset:
            predictions = model.predict(batch_images)

            for idx, (true_label, predicted_probs) in enumerate(zip(batch_labels.numpy(), predictions)):
                confidence_score = np.max(predicted_probs)
                predicted_label = np.argmax(predicted_probs)
                # Record the test images predictions to the CSV
                record_predictions_to_csv(
                    csv_filepath, image_index, confidence_score, predicted_label, true_label
                )
                image_index += 1  # Increment the index for each test image

        logging.info(f"Predictions recorded in {csv_filepath}.")

        # Confusion matrix and classification report
        test_predictions = np.concatenate(
            [model.predict(batch_images) for batch_images, _ in test_dataset], axis=0
        )
        test_predicted_labels = np.argmax(test_predictions, axis=1)
        conf_matrix = confusion_matrix(test_labels, test_predicted_labels)
        class_report = classification_report(test_labels, test_predicted_labels)

        logging.info("Confusion matrix and classification report generated.")

        return render_template(
            'evaluate_result.html',
            loss=loss,
            accuracy=accuracy,
            confusion_matrix=conf_matrix.tolist(),  # Convert to list for JSON serialization
            classification_report=class_report.replace('\n', '<br>'),
        )
    except Exception as e:
        logging.error(f"Evaluation error: {e}")
        return f"Error during evaluation: {e}", 500











# Temperature scaling function for logits
def temperature_scaling(logits, temperature):
    return logits / temperature

# Objective function for untargeted attack with temperature scaling
def attack_objective_untargeted_with_temp(params, image, original_label, temperature=1.0):
    x, y, r, g, b = params
    
    x, y = np.clip([int(x), int(y)], 0, [image.shape[0] - 1, image.shape[1] - 1])
    r, g, b = np.clip([r, g, b], 0.0, 1.0)
    
    perturbed_image = np.copy(image)
    perturbed_image[x, y] = [r, g, b]
    
    logits = model.predict(perturbed_image[np.newaxis, :], verbose=0)
    scaled_logits = temperature_scaling(logits, temperature)
    
    softmax_probs = tf.nn.softmax(scaled_logits).numpy()
    confidence = softmax_probs[0, original_label]
    
    return -confidence

# Objective function for targeted attack with temperature scaling
def attack_objective_targeted_with_temp(params, image, target_label, temperature=1.0):
    x, y, r, g, b = params
    
    x, y = np.clip([int(x), int(y)], 0, [image.shape[0] - 1, image.shape[1] - 1])
    r, g, b = np.clip([r, g, b], 0.0, 1.0)
    
    perturbed_image = np.copy(image)
    perturbed_image[x, y] = [r, g, b]
    
    logits = model.predict(perturbed_image[np.newaxis, :], verbose=0)
    scaled_logits = temperature_scaling(logits, temperature)
    
    softmax_probs = tf.nn.softmax(scaled_logits).numpy()
    confidence = softmax_probs[0, target_label]
    
    return -confidence

# Differential evolution attack function
def differential_evolution_attack(image, original_label, targeted=False, target_label=None, max_iter=100):
    height, width, _ = image.shape
    bounds = [(0, height - 1), (0, width - 1), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
    if targeted:
        if target_label is None:
            raise ValueError("Target label must be specified for targeted attacks.")
        args = (image, target_label)
        objective_func = attack_objective_targeted_with_temp
    else:
        args = (image, original_label)
        objective_func = attack_objective_untargeted_with_temp
    
    result = differential_evolution(
        objective_func,
        bounds,
        args=args,
        strategy='best1bin',
        maxiter=max_iter,
        popsize=15,
        tol=0.01,
        mutation=(0.7, 1.2),
        recombination=0.7,
        seed=42,
        disp=False
    )
    
    x_opt, y_opt, r_opt, g_opt, b_opt = result.x
    x_opt, y_opt = int(x_opt), int(y_opt)
    r_opt, g_opt, b_opt = r_opt, g_opt, b_opt
    
    perturbed_image = np.copy(image)
    perturbed_image[x_opt, y_opt] = [r_opt, g_opt, b_opt]
    
    rgb_values = (np.array([r_opt, g_opt, b_opt]) * 255).astype(int).tolist()
    
    return perturbed_image, (x_opt, y_opt), rgb_values

# Visualize attack
def visualize_attack(original_image, perturbed_image, perturbation_coords, rgb_values, attack_type='untargeted'):
    assert original_image.shape == perturbed_image.shape, "Original and perturbed images must have the same dimensions."
    
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(perturbed_image)
    plt.scatter(perturbation_coords[1], perturbation_coords[0], c='red', s=50, label='Perturbation Point')
    plt.title(f'Perturbed Image ({attack_type.capitalize()} Attack)')
    plt.axis('off')
    
    x_text = min(max(perturbation_coords[1] + 10, 0), perturbed_image.shape[1] - 1)
    y_text = min(max(perturbation_coords[0] + 10, 0), perturbed_image.shape[0] - 1)

    plt.text(
        x_text,
        y_text,
        f'RGB: {rgb_values}',
        color='yellow',
        fontsize=8,
        ha='left',
        va='bottom',
        bbox=dict(facecolor='black', alpha=0.5, pad=1)
    )
    
    plt.legend()

    if not os.path.exists(app.config['STATIC_FOLDER']):
        os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
    
    perturbed_image_filename = f'perturbation_visualization_{attack_type}.png'
    perturbed_image_path = os.path.join(app.config['STATIC_FOLDER'], perturbed_image_filename)
    
    try:
        plt.tight_layout()
        plt.savefig(perturbed_image_path)
        logging.debug(f"Perturbed image saved to: {perturbed_image_path}")
    except Exception as e:
        logging.error(f"Failed to save perturbed image: {str(e)}")
        raise
    finally:
        plt.close(fig)
    
    return perturbed_image_filename

# Flask Routes
@app.route('/')
def index():
    logging.info("Rendering index.html")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        logging.warning("No file part in the request")
        return "No file part in the request", 400

    file = request.files['file']

    if file.filename == '':
        logging.warning("No selected file")
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"Original image saved to: {filepath}")

        try:
            # Preprocess and predict the original image
            image = load_and_preprocess_image(filepath, target_shape=(224, 224))
            if image is None:
                return "Failed to process image.", 400
            predictions = model.predict(image[np.newaxis, :], verbose=0)
            original_label = np.argmax(predictions[0])
            original_confidence = predictions[0].max()
            logging.info(f"Original prediction: Label={original_label}, Confidence={original_confidence}")

            # Perform Untargeted Attack
            perturbed_image_untargeted, coords_untargeted, rgb_untargeted = differential_evolution_attack(
                image, original_label, targeted=False, max_iter=100
            )
            perturbed_predictions_untargeted = model.predict(perturbed_image_untargeted[np.newaxis, :], verbose=0)
            perturbed_label_untargeted = np.argmax(perturbed_predictions_untargeted)
            perturbed_confidence_untargeted = perturbed_predictions_untargeted[0].max()
            logging.info(f"Untargeted attack prediction: Label={perturbed_label_untargeted}, Confidence={perturbed_confidence_untargeted}")

            # Perform Targeted Attack
            target_label = 1 - original_label  # Simple target for demonstration
            perturbed_image_targeted, coords_targeted, rgb_targeted = differential_evolution_attack(
                image, original_label, targeted=True, target_label=target_label, max_iter=100
            )
            perturbed_predictions_targeted = model.predict(perturbed_image_targeted[np.newaxis, :], verbose=0)
            perturbed_label_targeted = np.argmax(perturbed_predictions_targeted)
            perturbed_confidence_targeted = perturbed_predictions_targeted[0].max()
            logging.info(f"Targeted attack prediction: Label={perturbed_label_targeted}, Confidence={perturbed_confidence_targeted}")

            # Save visualizations
            perturbed_image_path_untargeted = visualize_attack(
                image, perturbed_image_untargeted, coords_untargeted, rgb_untargeted, attack_type='untargeted'
            )
            perturbed_image_path_targeted = visualize_attack(
                image, perturbed_image_targeted, coords_targeted, rgb_targeted, attack_type='targeted'
            )

            # Determine attack success
            success_untargeted = perturbed_label_untargeted == original_label
            success_targeted = perturbed_label_targeted != target_label
            logging.info(f"Untargeted attack success: {success_untargeted}")
            logging.info(f"Targeted attack success: {success_targeted}")

            # Prepare information for display
            perturbation_info_untargeted = {'coords': coords_untargeted, 'rgb': rgb_untargeted}
            perturbation_info_targeted = {'coords': coords_targeted, 'rgb': rgb_targeted}

            original_image_static_path = os.path.join(app.config['STATIC_FOLDER'], filename)
            if filepath != original_image_static_path:
                with Image.open(filepath) as img:
                    img.save(original_image_static_path)
                logging.info(f"Original image copied to: {original_image_static_path}")
                
            return render_template(
                'result.html',
                original_label=original_label,
                original_confidence=round(original_confidence * 100, 2),
                perturbed_label_untargeted=perturbed_label_untargeted,
                perturbed_confidence_untargeted=round(perturbed_confidence_untargeted * 100, 2),
                success_untargeted=success_untargeted,
                perturbed_image_path_untargeted=perturbed_image_path_untargeted,
                perturbation_info_untargeted=perturbation_info_untargeted,
                perturbed_label_targeted=perturbed_label_targeted,
                perturbed_confidence_targeted=round(perturbed_confidence_targeted * 100, 2),
                success_targeted=success_targeted,
                perturbed_image_path_targeted=perturbed_image_path_targeted,
                perturbation_info_targeted=perturbation_info_targeted,
                original_image_path=filename
            )
        except Exception as e:
            logging.error(f"Error during attack processing: {e}")
            return "Error during attack processing.", 500
    else:
        return "Invalid file type. Please upload an image.", 400












    # if args.train:
    #     train_model()
    # if args.evaluate:
    #     evaluate_model()

# Run the Flask app
if __name__ == '__main__':
    logging.info("Starting Flask app...")
    app.run(debug=True)
    logging.info("Flask app is running.")
sys.stdout.flush()