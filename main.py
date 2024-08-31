import numpy as np
import matplotlib.pyplot as plt

# Görüntü boyutları
image_width = 256
image_height = 256

# Görüntü sayısı
num_images = 10000

# Rastgele renkli ve renksiz görüntülerin oluşturulması
def generate_random_images(num_images, width, height, channels):
    """
    Rastgele görüntüler oluşturur.

    Args:
        num_images: Oluşturulacak görüntü sayısı.
        width: Görüntü genişliği.
        height: Görüntü yüksekliği.
        channels: Renk kanalı sayısı (1: grayscale, 3: RGB).

    Returns:
        Numpy array olarak rastgele görüntüler.
    """
    return np.random.randint(0, 256, size=(num_images, width, height, channels))

# Renksiz görüntülerin oluşturulması
grayscale_images = generate_random_images(num_images, image_width, image_height, 1)

# Renkli görüntülerin oluşturulması
rgb_images = generate_random_images(num_images, image_width, image_height, 3)

# Veri noktası hesaplaması
def calculate_data_points(images):
    """
    Verilen görüntü seti için veri noktası sayısını hesaplar.

    Args:
        images: Numpy array olarak görüntüler.

    Returns:
        Her bir görüntü için veri noktası sayısı ve toplam veri noktası sayısı.
    """
    num_images, width, height, channels = images.shape
    n_x = width * height * channels  # Her bir görüntü için veri noktası sayısı
    total_data_points = n_x * num_images  # Toplam veri noktası sayısı
    return n_x, total_data_points

# Renksiz görüntülerin veri noktası hesaplaması
n_x_grayscale, total_data_grayscale = calculate_data_points(grayscale_images)
print(f"Renksiz görüntüler için her bir görüntüdeki veri noktası sayısı (n_x): {n_x_grayscale}")
print(f"Renksiz görüntüler için toplam veri noktası sayısı: {total_data_grayscale}")

# Renkli görüntülerin veri noktası hesaplaması
n_x_rgb, total_data_rgb = calculate_data_points(rgb_images)
print(f"Renkli görüntüler için her bir görüntüdeki veri noktası sayısı (n_x): {n_x_rgb}")
print(f"Renkli görüntüler için toplam veri noktası sayısı: {total_data_rgb}")

# Görüntülerin görselleştirilmesi
def display_image(image, title="Görüntü"):
    """
    Verilen görüntüyü görselleştirir.

    Args:
        image: Numpy array olarak görüntü.
        title: Görüntü başlığı.
    """
    plt.imshow(image.squeeze(), cmap='gray' if image.shape[-1] == 1 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Örnek görüntüleri görselleştirme
print("Renksiz bir görüntü örneği:")
display_image(grayscale_images[0], "Renksiz Görüntü (Grayscale)")

print("Renkli bir görüntü örneği:")
display_image(rgb_images[0], "Renkli Görüntü (RGB)")
