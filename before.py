import numpy as np
from PIL import Image


def process_and_save_fourier_image(image_path, output_spectrum_path, output_reconstructed_path, delta):
    # --- 1. Read Image and Convert to Grayscale ---
    image = Image.open(image_path).convert('L')
    img_array = np.array(image)

    # --- 2. Compute the 2D FFT ---
    f_transform = np.fft.fft2(img_array)

    # My changes
    # f_transform[0] = 1000
    print(f_transform[0][0])
    # f_transform[0][0] = 0+0j

    for y in range(image.height):
        for x in range(image.width):
            x_offset = abs(x - image.width/2)
            y_offset = abs(y - image.height/2)

            if (x_offset < delta*image.width/2 or y_offset < delta*image.height/2):
              f_transform[y][x] = 0

    # We store the original complex transform for inverse FFT later
    # We shift it only for visualization/saving the spectrum image
    f_transform_shifted = np.fft.fftshift(f_transform)

    # --- 3. Prepare Magnitude Spectrum for Saving/Visualization ---
    # Calculate magnitude and apply logarithmic scaling
    # Adding 1 to avoid log(0)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

    normalized_spectrum = (magnitude_spectrum / magnitude_spectrum.max()) * 255
    spectrum_image_array = normalized_spectrum.astype(np.uint8)

    Image.fromarray(spectrum_image_array).save(output_spectrum_path)

    # --- 5. Compute Inverse FFT to Reconstruct the Original Image ---
    # We use the *unshifted* complex data from step 2 for the inverse transform
    # np.fft.ifft2 returns complex numbers; we take the real part
    img_reconstructed_complex = np.fft.ifft2(f_transform)
    # Take the magnitude of the result
    img_reconstructed = np.abs(img_reconstructed_complex)

    # Ensure the values are in the valid range for an 8-bit image (0-255)
    img_reconstructed = np.clip(img_reconstructed, 0, 255)
    img_reconstructed_array = img_reconstructed.astype(np.uint8)

    # --- 6. Save the Reconstructed Image Using PIL ---
    img_reconstructed_pil = Image.fromarray(img_reconstructed_array, mode='L')
    img_reconstructed_pil.save(output_reconstructed_path)


for i in range(10):
    process_and_save_fourier_image("random.jpg", "spectrum" + str(i) + ".jpg", "rebuilt" + str(i) + ".jpg", (1-1/(2**i)))
