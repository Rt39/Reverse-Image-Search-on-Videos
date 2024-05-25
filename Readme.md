# Reverse Image search on Videos

A system based on object detection and autoencoders for video reverse searching.

## Run instruction

Run the whole project by using

```bash
docker compose up -d
```

Start the Jupyter Notebook server with http://localhost:8888/tree and direct to `Image_Encoder.ipynb`, run the whole notebook and at the last cell it will start a local server which provides image searching. Get to the server at http://localhost:5000/ start image searching.

Stop the project by

```bash
docker compose down
```

and delete the unneed images.

## Description

The running **results** can be found in `data/results`. You could delete the result and rerun for it.