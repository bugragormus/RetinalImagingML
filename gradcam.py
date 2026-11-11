"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Retinal Pathology
Provides visual interpretability (XAI) for ophthalmic fundus image classification.
"""

import numpy as np
import cv2

try:
    import tensorflow as tf
except ImportError:
    tf = None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Computes Grad-CAM heatmap for a given input image and trained Keras model.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for Grad-CAM computation.")

    # Auto-detect target conv layer if not specified
    if last_conv_layer_name is None:
        # Search for InceptionV3 'mixed10' or last Conv2D layer in the architecture
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):  # Nested Sequential or functional model
                for sublayer in reversed(layer.layers):
                    if "mixed10" in sublayer.name or isinstance(sublayer, tf.keras.layers.Conv2D):
                        last_conv_layer_name = sublayer.name
                        break
            elif "mixed10" in layer.name or isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    # If the model has a nested base model (like InceptionV3 inside Sequential)
    base_model = None
    for layer in model.layers:
        if "inception" in layer.name.lower() or hasattr(layer, "get_layer"):
            try:
                layer.get_layer(last_conv_layer_name)
                base_model = layer
                break
            except Exception:
                pass

    if base_model is not None:
        # Handle Sequential(InceptionV3 -> GAP -> Dense -> Dense)
        grad_model = tf.keras.models.Model(
            inputs=[base_model.input],
            outputs=[base_model.get_layer(last_conv_layer_name).output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs = grad_model(img_array)
            tape.watch(conv_outputs)
            
            # Forward pass through top layers
            x = conv_outputs
            for layer in model.layers[1:]:
                x = layer(x)
            
            preds = x
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
    else:
        # Standard flat functional model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    # Global Average Pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only positive features contributing to the class
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def generate_gradcam_overlay(orig_img_rgb, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Superimposes the Grad-CAM heatmap over the original RGB image.
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (orig_img_rgb.shape[1], orig_img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Colorize heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on original image
    superimposed_img = np.float32(heatmap_colored) * alpha + np.float32(orig_img_rgb) * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img, heatmap_colored
