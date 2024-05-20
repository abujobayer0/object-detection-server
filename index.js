const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const cocoSsd = require("@tensorflow-models/coco-ssd");
const app = express();
const upload = multer({ dest: "uploads/" });
const cors = require("cors");
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 4000;
let model;

cocoSsd.load().then((loadedModel) => {
  model = loadedModel;
  console.log("Model loaded");
});

app.post("/detect", upload.single("image"), async (req, res) => {
  if (!model) {
    return res
      .status(500)
      .send("Model is not loaded yet, please try again later.");
  }

  const imagePath = req.file.path;
  const image = await loadImage(imagePath);

  const predictions = await model.detect(image);
  res.json(predictions);
});

const loadImage = async (path) => {
  const imageBuffer = require("fs").readFileSync(path);
  const tfimage = tf.node.decodeImage(imageBuffer);
  return tfimage;
};

app.get("/", (req, res) => {
  res.type("html").send(html);
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
const html = `<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection Server</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            font-family: 'Arial, sans-serif';
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            background: linear-gradient(to right, #2c3e50, #4ca1af);
            padding: 20px;
            text-align: center;
            color: #fff;
        }
        .header h1 {
            margin: 0;
            font-size: 3em;
        }
        .header p {
            margin: 10px 0 0;
            font-size: 1.2em;
        }
        .container {
            padding: 20px;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            font-size: 2em;
            margin-bottom: 20px;
        }
        .section p {
            font-size: 1.2em;
            line-height: 1.6em;
            margin-bottom: 20px;
        }
        .features {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .feature {
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            flex: 1 1 calc(33% - 40px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .feature h3 {
            margin-top: 0;
            font-size: 1.5em;
        }
        .cta {
            text-align: center;
            background: linear-gradient(to right, #4ca1af, #2c3e50);
            color: #fff;
            padding: 40px 20px;
        }
        .cta h2 {
            margin-top: 0;
            font-size: 2.5em;
        }
        .cta p {
            font-size: 1.2em;
        }
        .cta a {
            display: inline-block;
            background: #fff;
            color: #2c3e50;
            padding: 15px 30px;
            font-size: 1.2em;
            border-radius: 5px;
            text-decoration: none;
            margin-top: 20px;
        }
        .footer {
            background: #2c3e50;
            color: #fff;
            text-align: center;
            padding: 20px;
        }
        .footer p {
            margin: 0;
            font-size: 1em;
        }
        @media (max-width: 768px) {
            .features {
                flex-direction: column;
            }
            .feature {
                flex: 1 1 100%;
            }
        }
        .api-endpoints {
          margin-top: 40px;
      }
      .api-endpoint {
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      .api-endpoint h3 {
          margin-top: 0;
          font-size: 1.5em;
      }
      .api-endpoint code {
          display: block;
          background: #f5f5f5;
          padding: 10px;
          border-radius: 4px;
          margin-top: 10px;
          white-space: pre-wrap;
      }
    </style>
</head>
<body>
    <div class="header">
        <h1>Welcome to Our Object Detection Server</h1>
        <p>High Performance, Secure, and Reliable Object Detection Solutions</p>
    </div>
    <div class="container">
        <div class="section">
            <h2>About Our Server</h2>
            <p>Our server offers unmatched performance and reliability for all your object detection needs. With state-of-the-art technology and top-notch security, we ensure that your data is safe and accessible at all times.</p>
        </div>
        <div class="section api-endpoints">
        <h2>API Endpoints</h2>
        <div class="api-endpoint">
            <h3>Object Detection</h3>
            <p>Endpoint: <code>POST /detect</code></p>
            <p>Upload an image to detect objects within it.</p>
            <code>
curl -X POST http://localhost:4000/detect \\
 -H "Content-Type: multipart/form-data" \\
 -F "image=@path/to/your/image.jpg"
            </code>
        </div>
        <!-- Additional endpoints can be added similarly -->
    </div>
    <div class="footer">
        <p>&copy; 2024 Abu Jobayer. All rights reserved.</p>
    </div>
</body>
</html>
`;
