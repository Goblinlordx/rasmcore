//go:build !cgo

package rcimg

import "fmt"

type ffiBackend struct{}

func newFFIBackend(_ Options) (*ffiBackend, error) {
	return nil, fmt.Errorf("FFI backend requires CGO_ENABLED=1 and librcimg installed")
}

func (f *ffiBackend) Process(_ []byte, _ []Op, _ OutputConfig) ([]byte, error) {
	return nil, fmt.Errorf("FFI backend not available")
}

func (f *ffiBackend) Close() error { return nil }
