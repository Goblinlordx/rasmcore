# rasmcore-image — Python SDK

Image processing SDK powered by WebAssembly.

## Install

```bash
pip install rasmcore-image
```

## Usage (Dynamic)

```python
from rasmcore import RcImage

img = RcImage.load("rasmcore_image.wasm", open("photo.png", "rb").read())
jpeg = img.apply("blur", radius=3.0).encode("jpeg", quality=85)
```

## Usage (Typed)

```bash
rasmcore-image-typegen manifest.json --output rasmcore_typed.py
```

```python
from rasmcore_typed import RcImage

jpeg = RcImage.load("module.wasm", data).blur(radius=3.0).to_jpeg(quality=85)
```
