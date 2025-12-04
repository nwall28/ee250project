import requests
import base64


def test_with_file(image_path):
    """Test with an existing image file"""
    print(f"Reading {image_path}...")

    # Read the image file
    with open(image_path, 'rb') as f:
        image_data = f.read()

    # Convert to base64
    img_base64 = base64.b64encode(image_data).decode('utf-8')

    # Send to API
    print("Sending to API...")
    response = requests.post(
        'http://localhost:5000/image',
        json={'image': img_base64, 'format': 'jpeg'}
    )

    if response.status_code == 201:
        print("SUCCESS!")
        print(f"Response: {response.json()}")
        print("\nView at: http://localhost:5000/")
    else:
        print(f"Failed: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    test_with_file('IMG_9721.jpg')