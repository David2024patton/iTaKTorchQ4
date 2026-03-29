package llama

import (
	"errors"
	"unsafe"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/utils"
	"github.com/ebitengine/purego"
)

// --- purego direct-call function pointers ---
var (
	adapterLoraInitFn               func(model Model, pathLora *byte) AdapterLora
	adapterLoraFreeFn               func(adapter AdapterLora)
	adapterMetaValStrFn             func(adapter AdapterLora, key *byte, buf *byte, bufSize uintptr) int32
	adapterMetaCountFn              func(adapter AdapterLora) int32
	adapterMetaKeyByIndexFn         func(adapter AdapterLora, i int32, buf *byte, bufSize uintptr) int32
	adapterMetaValStrByIndexFn      func(adapter AdapterLora, i int32, buf *byte, bufSize uintptr) int32
	setAdaptersLoraFn               func(ctx Context, adapters *AdapterLora, nAdapters uintptr, scales *float32) int32
	adapterGetAloraNInvocationFn    func(adapter AdapterLora) uint64
	adapterGetAloraInvocationTokFn  func(adapter AdapterLora) *Token
)

var (
	errInvalidAdapter = errors.New("invalid LoRA adapter")
)

func loadLoraFuncs(lib uintptr) error {
	purego.RegisterLibFunc(&adapterLoraInitFn, lib, "llama_adapter_lora_init")
	purego.RegisterLibFunc(&adapterLoraFreeFn, lib, "llama_adapter_lora_free")
	purego.RegisterLibFunc(&adapterMetaValStrFn, lib, "llama_adapter_meta_val_str")
	purego.RegisterLibFunc(&adapterMetaCountFn, lib, "llama_adapter_meta_count")
	purego.RegisterLibFunc(&adapterMetaKeyByIndexFn, lib, "llama_adapter_meta_key_by_index")
	purego.RegisterLibFunc(&adapterMetaValStrByIndexFn, lib, "llama_adapter_meta_val_str_by_index")
	purego.RegisterLibFunc(&setAdaptersLoraFn, lib, "llama_set_adapters_lora")
	purego.RegisterLibFunc(&adapterGetAloraNInvocationFn, lib, "llama_adapter_get_alora_n_invocation_tokens")
	purego.RegisterLibFunc(&adapterGetAloraInvocationTokFn, lib, "llama_adapter_get_alora_invocation_tokens")
	return nil
}

// AdapterLoraInit loads a LoRA adapter from file and applies it to the model.
func AdapterLoraInit(model Model, pathLora string) (AdapterLora, error) {
	if model == 0 {
		return 0, errors.New("invalid model")
	}

	file := &[]byte(pathLora + "\x00")[0]
	adapter := adapterLoraInitFn(model, file)
	return adapter, nil
}

// AdapterLoraFree manually frees a LoRA adapter.
func AdapterLoraFree(adapter AdapterLora) error {
	if adapter == 0 {
		return errInvalidAdapter
	}
	adapterLoraFreeFn(adapter)
	return nil
}

// AdapterMetaValStr gets metadata value as a string by key name.
func AdapterMetaValStr(adapter AdapterLora, key string) (string, bool) {
	if adapter == 0 {
		return "", false
	}
	buf := make([]byte, 32768)
	b := unsafe.SliceData(buf)

	keyPtr, _ := utils.BytePtrFromString(key)
	result := adapterMetaValStrFn(adapter, keyPtr, b, uintptr(len(buf)))
	if result < 0 {
		return "", false
	}

	value := make([]byte, result)
	copy(value, buf[:result])
	return string(value), true
}

// AdapterMetaCount gets the number of metadata key/value pairs.
func AdapterMetaCount(adapter AdapterLora) int32 {
	if adapter == 0 {
		return 0
	}
	return adapterMetaCountFn(adapter)
}

// AdapterMetaKeyByIndex gets metadata key name by index.
func AdapterMetaKeyByIndex(adapter AdapterLora, i int32) (string, bool) {
	if adapter == 0 {
		return "", false
	}
	buf := make([]byte, 128)
	b := unsafe.SliceData(buf)

	result := adapterMetaKeyByIndexFn(adapter, i, b, uintptr(len(buf)))
	if result < 0 {
		return "", false
	}

	value := make([]byte, result)
	copy(value, buf[:result])
	return string(value), true
}

// AdapterMetaValStrByIndex gets metadata value as a string by index.
func AdapterMetaValStrByIndex(adapter AdapterLora, i int32) (string, bool) {
	if adapter == 0 {
		return "", false
	}
	buf := make([]byte, 32768)
	b := unsafe.SliceData(buf)

	result := adapterMetaValStrByIndexFn(adapter, i, b, uintptr(len(buf)))
	if result < 0 {
		return "", false
	}

	value := make([]byte, result)
	copy(value, buf[:result])
	return string(value), true
}

// SetAdaptersLora sets LoRa adapters on the context.
func SetAdaptersLora(ctx Context, adapters []AdapterLora, scales []float32) int32 {
	if ctx == 0 || len(adapters) == 0 || len(adapters) != len(scales) {
		return -1
	}

	adaptersPtr := unsafe.SliceData(adapters)
	scalesPtr := unsafe.SliceData(scales)
	return setAdaptersLoraFn(ctx, adaptersPtr, uintptr(len(adapters)), scalesPtr)
}

// AdapterGetAloraNInvocationTokens returns the number of invocation tokens for the adapter.
func AdapterGetAloraNInvocationTokens(adapter AdapterLora) uint64 {
	if adapter == 0 {
		return 0
	}
	return adapterGetAloraNInvocationFn(adapter)
}

// AdapterGetAloraInvocationTokens returns a slice of invocation tokens for the adapter.
func AdapterGetAloraInvocationTokens(adapter AdapterLora) []Token {
	n := AdapterGetAloraNInvocationTokens(adapter)
	if n == 0 {
		return nil
	}

	ptr := adapterGetAloraInvocationTokFn(adapter)
	if ptr == nil {
		return nil
	}

	return unsafe.Slice(ptr, n)
}
