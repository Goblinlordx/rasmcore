// Package rasmcore provides manifest types for WASM module introspection.
//
// These types are module-agnostic. Any WASM component that exposes
// GetFilterManifest() is a valid module.
package rasmcore

import "encoding/json"

// ParamMeta describes a single operation parameter.
type ParamMeta struct {
	Name    string      `json:"name"`
	Type    string      `json:"type"`
	Min     *float64    `json:"min"`
	Max     *float64    `json:"max"`
	Step    *float64    `json:"step"`
	Default interface{} `json:"default"`
	Label   string      `json:"label"`
	Hint    string      `json:"hint"`
}

// OperationMeta describes a single filter/transform operation.
type OperationMeta struct {
	Name      string      `json:"name"`
	Category  string      `json:"category"`
	Group     string      `json:"group"`
	Variant   string      `json:"variant"`
	Reference string      `json:"reference"`
	Params    []ParamMeta `json:"params"`
}

// FilterManifest describes all operations a WASM module supports.
type FilterManifest struct {
	Filters []OperationMeta `json:"filters"`
}

// ParseManifest parses a FilterManifest from the JSON returned by GetFilterManifest().
func ParseManifest(data []byte) (*FilterManifest, error) {
	var m FilterManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}
