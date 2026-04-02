package rcimg

// Chain builds a sequence of image processing operations using a fluent API.
// Create a Chain via Pipeline.Read(), add operations, then call a Write method.
//
// Example:
//
//	result, err := pipe.Read(imageBytes).Invert().Blur(5.0).WritePNG()
type Chain struct {
	pipe  *Pipeline
	input []byte
	ops   []Op
	err   error
}

// Read starts a new processing chain from raw image bytes (PNG, JPEG, etc.).
func (p *Pipeline) Read(input []byte) *Chain {
	return &Chain{pipe: p, input: input}
}

// add appends an operation to the chain and returns the chain for fluent chaining.
func (c *Chain) add(name string, params ...float32) *Chain {
	if c.err != nil {
		return c
	}
	c.ops = append(c.ops, Op{Name: name, Params: params})
	return c
}

// WritePNG encodes the chain result as PNG.
func (c *Chain) WritePNG() ([]byte, error) {
	if c.err != nil {
		return nil, c.err
	}
	return c.pipe.Process(c.input, c.ops, OutputConfig{Format: "png"})
}

// WriteJPEG encodes the chain result as JPEG with the given quality (1-100).
func (c *Chain) WriteJPEG(quality uint8) ([]byte, error) {
	if c.err != nil {
		return nil, c.err
	}
	return c.pipe.Process(c.input, c.ops, OutputConfig{Format: "jpeg", Quality: &quality})
}

// WriteWebP encodes the chain result as WebP with the given quality (1-100).
func (c *Chain) WriteWebP(quality uint8) ([]byte, error) {
	if c.err != nil {
		return nil, c.err
	}
	return c.pipe.Process(c.input, c.ops, OutputConfig{Format: "webp", Quality: &quality})
}

// Write encodes the chain result in the specified format with optional quality.
func (c *Chain) Write(format string, quality *uint8) ([]byte, error) {
	if c.err != nil {
		return nil, c.err
	}
	return c.pipe.Process(c.input, c.ops, OutputConfig{Format: format, Quality: quality})
}
