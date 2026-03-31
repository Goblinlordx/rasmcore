# rasmcore-image-go — Go SDK

Image processing SDK powered by WebAssembly.

## Install

```bash
go get github.com/ArtProcessors/rasmcore-image-go
```

## Usage (Dynamic)

```go
img, err := rasmcore.Load("rasmcore_image.wasm", pngData)
jpeg, err := img.Apply("blur", map[string]any{"radius": 3.0}).Encode("jpeg", nil)
```

## Usage (Typed)

```bash
go run github.com/ArtProcessors/rasmcore-image-go/cmd/typegen manifest.json -o rasmcore_typed.go
```

```go
jpeg, err := rasmcore.Load("module.wasm", data).Blur(3.0).ToJPEG(nil)
```
