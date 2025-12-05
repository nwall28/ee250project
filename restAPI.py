from flask import Flask, request, jsonify, send_file, send_from_directory
import base64
from io import BytesIO
import time
import pathlib

app = Flask(__name__)
thisdir = pathlib.Path(__file__).parent.absolute()

latest_image = None
last_updated = None  # Timestamp of when image was last updated

@app.route('/image', methods=['POST'])
def send_image_route():
    """
    Summary: Receives and stores the latest image from the camera (replaces previous)
    """
    global latest_image, last_updated
    
    try:
        image_entry = request.get_json()

        # Validate required fields
        if not image_entry or 'image' not in image_entry:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image and store in memory
        image_data = base64.b64decode(image_entry['image'])
        latest_image = image_data
        last_updated = time.time()
        
        print("Latest image updated")
        
        res = jsonify({'status': 'success', 'message': 'Image updated'})
        res.status_code = 201  # Status code for "created"
        return res

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# Serve static files (CSS, JS)
@app.route('/app.css')
def serve_css():
    return send_from_directory(thisdir, 'app.css')

@app.route('/src/main.js')
def serve_js():
    return send_from_directory(thisdir / 'src', 'main.js')

@app.route('/')
def index():
    """Display the latest image using index.html structure"""
    if latest_image is None:
        # No image yet - show waiting message with same styling
        return """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="UTF-8" />
            <link rel="stylesheet" href="app.css">
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>SlapABigBack.com</title>
          </head>
          
          <body>
            <h1 class="id">SlapABigBack.com</h1>
            <div class="main-container">
                <div id="suspect">
                    <p style="color: white; font-size: 24px;">Your snacks are safe. So far...</p>
                </div>
            </div>
            
            <script>
                // Check for new images every 2 seconds
                setInterval(() => {
                    fetch('/check_update')
                        .then(res => res.json())
                        .then(data => {
                            if (data.has_image) {
                                console.log('New thief detected! Refreshing...');
                                location.reload();
                            }
                        });
                }, 2000);
            </script>
          </body>
        </html>
        """
    
    # Image exists - show it using index.html structure
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <link rel="stylesheet" href="app.css">
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>SlapABigBack.com - ALERT!</title>
        <style>
            #suspect img {{
                max-width: 100%;
                height: auto;
                display: block;
            }}
        </style>
      </head>
      
      <body>
        <h1 class="id">SlapABigBack.com</h1>
        <div class="main-container">
            <div id="suspect">
                <img src="/latest_image?t={last_updated}" alt="Suspect">
            </div>
            <div id="suspect">
                <h1 class="id">Was found stealing your food.</h1>
            </div>
        </div>
        
        <script type="module" src="src/main.js"></script>
        <script>
            let lastUpdate = {last_updated};
            
            // Check for new images every 2 seconds
            setInterval(() => {{
                fetch('/check_update')
                    .then(res => res.json())
                    .then(data => {{
                        if (data.last_updated > lastUpdate) {{
                            console.log('New thief detected! Refreshing...');
                            location.reload();
                        }}
                    }});
            }}, 2000);
        </script>
      </body>
    </html>
    """

@app.route('/check_update')
def check_update():
    """Check if there's a new image"""
    return jsonify({
        'has_image': latest_image is not None,
        'last_updated': last_updated if last_updated else 0
    })

@app.route('/latest_image')
def get_latest_image():
    """Serve the latest image"""
    if latest_image is None:
        return jsonify({'error': 'No image available'}), 404
    
    return send_file(
        BytesIO(latest_image),
        mimetype='image/jpeg',
        as_attachment=False
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)