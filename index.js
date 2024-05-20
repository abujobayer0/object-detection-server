const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const cocoSsd = require("@tensorflow-models/coco-ssd");
const app = express();
const upload = multer({ dest: "uploads/" });
const cors = require("cors");

const PORT = process.env.PORT || 4000;
let model;

cocoSsd.load().then((loadedModel) => {
  model = loadedModel;
  console.log("Model loaded");
});

app.use(cors());

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
  res.send("Running server");
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
