package rcimg

import (
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// wasmtimeBackend executes pipelines via wasmtime CLI subprocess.
type wasmtimeBackend struct {
	wasmPath     string
	cwasmPath    string
	wasmtimePath string
}

func newWasmtimeBackend(opts Options) (*wasmtimeBackend, error) {
	wasmPath, err := findWASM(opts.WASMPath)
	if err != nil {
		return nil, err
	}
	wtPath, err := findWasmtime(opts.WasmtimePath)
	if err != nil {
		return nil, err
	}
	return &wasmtimeBackend{
		wasmPath:     wasmPath,
		wasmtimePath: wtPath,
	}, nil
}

func findWASM(override string) (string, error) {
	candidates := []string{
		override,
		os.Getenv("RASMCORE_WASM_PATH"),
		filepath.Join(".", "rasmcore_image.wasm"),
		filepath.Join("target", "wasm32-wasip1", "release", "rasmcore_image.wasm"),
		filepath.Join("target", "wasm32-wasip1", "debug", "rasmcore_image.wasm"),
	}
	for _, p := range candidates {
		if p == "" {
			continue
		}
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf(
		"rasmcore_image.wasm not found; set RASMCORE_WASM_PATH or build with: " +
			"cargo component build -p rasmcore-image --release",
	)
}

func findWasmtime(override string) (string, error) {
	if override != "" {
		if _, err := os.Stat(override); err == nil {
			return override, nil
		}
		return "", fmt.Errorf("wasmtime not found at %q", override)
	}
	path, err := exec.LookPath("wasmtime")
	if err != nil {
		return "", fmt.Errorf("wasmtime CLI not found in PATH; install from https://wasmtime.dev")
	}
	return path, nil
}

func (w *wasmtimeBackend) Process(input []byte, ops []Op, output OutputConfig) ([]byte, error) {
	if len(ops) == 0 {
		return nil, fmt.Errorf("at least one operation is required")
	}
	if output.Format == "" {
		output.Format = "png"
	}

	inputB64 := base64.StdEncoding.EncodeToString(input)

	opStrs := make([]string, len(ops))
	for i, op := range ops {
		paramStrs := make([]string, len(op.Params))
		for j, v := range op.Params {
			paramStrs[j] = fmt.Sprintf("%g", v)
		}
		opStrs[i] = fmt.Sprintf("{name: \"%s\", params: [%s]}", op.Name, strings.Join(paramStrs, ", "))
	}

	qualityWave := "none"
	if output.Quality != nil {
		qualityWave = fmt.Sprintf("some(%d)", *output.Quality)
	}

	invoke := fmt.Sprintf(
		`process-chain("%s", [%s], "%s", %s)`,
		inputB64,
		strings.Join(opStrs, ", "),
		output.Format,
		qualityWave,
	)

	runPath := w.wasmPath
	args := []string{"run", "--invoke", invoke}
	if w.cwasmPath != "" {
		runPath = w.cwasmPath
		args = append(args, "--allow-precompiled")
	}
	args = append(args, runPath)
	cmd := exec.Command(w.wasmtimePath, args...)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("wasmtime failed: %s\n%s", ee.Error(), string(ee.Stderr))
		}
		return nil, fmt.Errorf("wasmtime failed: %w", err)
	}

	result := strings.TrimSpace(string(out))
	if strings.HasPrefix(result, `ok("`) && strings.HasSuffix(result, `")`) {
		b64 := result[4 : len(result)-2]
		return base64.StdEncoding.DecodeString(b64)
	}
	if strings.HasPrefix(result, "err(") {
		return nil, fmt.Errorf("pipeline error: %s", result[4:len(result)-1])
	}
	return nil, fmt.Errorf("unexpected wasmtime output: %s", result)
}

func (w *wasmtimeBackend) Close() error {
	return nil
}

// Compile pre-compiles the WASM for faster startup.
func (w *wasmtimeBackend) Compile() error {
	cwasmPath := w.wasmPath + ".cwasm"
	if info, err := os.Stat(cwasmPath); err == nil {
		if srcInfo, err2 := os.Stat(w.wasmPath); err2 == nil {
			if info.ModTime().After(srcInfo.ModTime()) {
				w.cwasmPath = cwasmPath
				return nil
			}
		}
	}
	cmd := exec.Command(w.wasmtimePath, "compile", w.wasmPath, "-o", cwasmPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("wasmtime compile failed: %s\n%s", err, string(out))
	}
	w.cwasmPath = cwasmPath
	return nil
}
