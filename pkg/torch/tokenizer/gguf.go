// Pure Go GGUF metadata reader. Parses the key-value header from GGUF files
// to extract tokenizer vocabulary, merge rules, and special token IDs.
package tokenizer

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// TokenType represents the type of a token in the vocabulary.
type TokenType int32

const (
	TokenTypeNormal      TokenType = 1
	TokenTypeUnknown     TokenType = 2
	TokenTypeControl     TokenType = 3
	TokenTypeUserDefined TokenType = 4
	TokenTypeUnused      TokenType = 5
	TokenTypeByte        TokenType = 6
)

const ggufMagic = 0x46554747 // "GGUF" little-endian

// GGUF metadata value types.
const (
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12
)

// ReadGGUFMetadata reads all key-value metadata from a GGUF file header.
// Returns the metadata map. Only reads the header, does not load tensor data.
func ReadGGUFMetadata(r io.ReadSeeker) (map[string]interface{}, error) {
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("not a GGUF file (magic: 0x%08X)", magic)
	}

	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", version)
	}

	// Skip tensor count.
	var nTensors uint64
	if err := binary.Read(r, binary.LittleEndian, &nTensors); err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}

	// Number of metadata key-value pairs.
	var nKV uint64
	if err := binary.Read(r, binary.LittleEndian, &nKV); err != nil {
		return nil, fmt.Errorf("read kv count: %w", err)
	}

	metadata := make(map[string]interface{}, nKV)
	for i := uint64(0); i < nKV; i++ {
		key, err := readGGUFString(r)
		if err != nil {
			return nil, fmt.Errorf("read key %d: %w", i, err)
		}
		value, err := readGGUFValue(r)
		if err != nil {
			return nil, fmt.Errorf("read value for %q: %w", key, err)
		}
		metadata[key] = value
	}

	return metadata, nil
}

// readGGUFString reads a GGUF string (uint64 length + bytes).
func readGGUFString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1<<20 { // sanity limit: 1MB per string
		return "", fmt.Errorf("string too long: %d bytes", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readGGUFValue reads a typed GGUF value.
func readGGUFValue(r io.Reader) (interface{}, error) {
	var valueType uint32
	if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
		return nil, err
	}

	switch valueType {
	case ggufTypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat32:
		var v uint32
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		return math.Float32frombits(v), nil
	case ggufTypeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case ggufTypeString:
		return readGGUFString(r)
	case ggufTypeArray:
		return readGGUFArray(r)
	case ggufTypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat64:
		var v uint64
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		return math.Float64frombits(v), nil
	default:
		return nil, fmt.Errorf("unknown GGUF type: %d", valueType)
	}
}

// readGGUFArray reads a typed GGUF array.
func readGGUFArray(r io.Reader) (interface{}, error) {
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return nil, err
	}
	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}

	switch elemType {
	case ggufTypeString:
		arr := make([]string, count)
		for i := uint64(0); i < count; i++ {
			s, err := readGGUFString(r)
			if err != nil {
				return nil, fmt.Errorf("array string %d: %w", i, err)
			}
			arr[i] = s
		}
		return arr, nil
	case ggufTypeFloat32:
		arr := make([]float32, count)
		for i := uint64(0); i < count; i++ {
			var v uint32
			if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
				return nil, err
			}
			arr[i] = math.Float32frombits(v)
		}
		return arr, nil
	case ggufTypeInt32:
		arr := make([]int32, count)
		for i := uint64(0); i < count; i++ {
			if err := binary.Read(r, binary.LittleEndian, &arr[i]); err != nil {
				return nil, err
			}
		}
		return arr, nil
	case ggufTypeUint32:
		arr := make([]uint32, count)
		for i := uint64(0); i < count; i++ {
			if err := binary.Read(r, binary.LittleEndian, &arr[i]); err != nil {
				return nil, err
			}
		}
		return arr, nil
	default:
		// For other types, read as raw interface slice.
		arr := make([]interface{}, count)
		for i := uint64(0); i < count; i++ {
			v, err := readGGUFValueByType(r, elemType)
			if err != nil {
				return nil, fmt.Errorf("array elem %d: %w", i, err)
			}
			arr[i] = v
		}
		return arr, nil
	}
}

// readGGUFValueByType reads a GGUF value of a known type (no type prefix).
func readGGUFValueByType(r io.Reader, valueType uint32) (interface{}, error) {
	switch valueType {
	case ggufTypeUint8:
		var v uint8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt8:
		var v int8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeUint16:
		var v uint16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt16:
		var v int16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeUint64:
		var v uint64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt64:
		var v int64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	default:
		return nil, fmt.Errorf("unsupported array element type: %d", valueType)
	}
}

// toInt32 converts various numeric types to int32.
func toInt32(v interface{}) int32 {
	switch val := v.(type) {
	case int32:
		return val
	case uint32:
		return int32(val)
	case int64:
		return int32(val)
	case uint64:
		return int32(val)
	case int:
		return int32(val)
	default:
		return -1
	}
}
