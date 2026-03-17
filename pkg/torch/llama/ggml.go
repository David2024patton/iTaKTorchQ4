package llama

import (
	"errors"

	"github.com/David2024patton/iTaKTorch/pkg/torch/utils"
	"github.com/ebitengine/purego"
)

type (
	GGMLBackendDeviceType int32
	GGMLBackendDevice     uintptr
	GGMLBackendReg        uintptr
	GGMLType              int32
)

const (
	GGMLBackendDeviceTypeCPU   GGMLBackendDeviceType = iota
	GGMLBackendDeviceTypeGPU
	GGMLBackendDeviceTypeIGPU
	GGMLBackendDeviceTypeACCEL
)

const (
	GGMLTypeF32     GGMLType = 0
	GGMLTypeF16     GGMLType = 1
	GGMLTypeQ4_0    GGMLType = 2
	GGMLTypeQ4_1    GGMLType = 3
	GGMLTypeQ5_0    GGMLType = 6
	GGMLTypeQ5_1    GGMLType = 7
	GGMLTypeQ8_0    GGMLType = 8
	GGMLTypeQ8_1    GGMLType = 9
	GGMLTypeQ2_K    GGMLType = 10
	GGMLTypeQ3_K    GGMLType = 11
	GGMLTypeQ4_K    GGMLType = 12
	GGMLTypeQ5_K    GGMLType = 13
	GGMLTypeQ6_K    GGMLType = 14
	GGMLTypeQ8_K    GGMLType = 15
	GGMLTypeIQ2_XXS GGMLType = 16
	GGMLTypeIQ2_XS  GGMLType = 17
	GGMLTypeIQ3_XXS GGMLType = 18
	GGMLTypeIQ1_S   GGMLType = 19
	GGMLTypeIQ4_NL  GGMLType = 20
	GGMLTypeIQ3_S   GGMLType = 21
	GGMLTypeIQ2_S   GGMLType = 22
	GGMLTypeIQ4_XS  GGMLType = 23
	GGMLTypeI8      GGMLType = 24
	GGMLTypeI16     GGMLType = 25
	GGMLTypeI32     GGMLType = 26
	GGMLTypeI64     GGMLType = 27
	GGMLTypeF64     GGMLType = 28
	GGMLTypeIQ1_M   GGMLType = 29
	GGMLTypeBF16    GGMLType = 30
	GGMLTypeTQ1_0   GGMLType = 34
	GGMLTypeTQ2_0   GGMLType = 35
	GGMLTypeMXFP4   GGMLType = 39
	GGMLTypeCOUNT   GGMLType = 40
)

// --- purego direct-call function pointers ---
var (
	ggmlBackendLoadAllFn       func()
	ggmlBackendLoadAllFromPath func(path *byte)
	ggmlBackendUnloadFn        func(reg GGMLBackendReg)
	ggmlBackendDevCountFn      func() uintptr
	ggmlBackendDevGetFn        func(index uintptr) GGMLBackendDevice
	ggmlBackendDevByNameFn     func(name *byte) GGMLBackendDevice
	ggmlBackendDevByTypeFn     func(devType int32) GGMLBackendDevice
	ggmlBackendRegCountFn      func() uintptr
	ggmlBackendRegGetFn        func(index uintptr) GGMLBackendReg
	ggmlBackendRegByNameFn     func(name *byte) GGMLBackendReg
)

func loadGGML(lib uintptr) error {
	purego.RegisterLibFunc(&ggmlBackendLoadAllFn, lib, "ggml_backend_load_all")
	purego.RegisterLibFunc(&ggmlBackendLoadAllFromPath, lib, "ggml_backend_load_all_from_path")
	purego.RegisterLibFunc(&ggmlBackendUnloadFn, lib, "ggml_backend_unload")
	purego.RegisterLibFunc(&ggmlBackendDevCountFn, lib, "ggml_backend_dev_count")
	purego.RegisterLibFunc(&ggmlBackendDevGetFn, lib, "ggml_backend_dev_get")
	purego.RegisterLibFunc(&ggmlBackendDevByNameFn, lib, "ggml_backend_dev_by_name")
	purego.RegisterLibFunc(&ggmlBackendDevByTypeFn, lib, "ggml_backend_dev_by_type")
	purego.RegisterLibFunc(&ggmlBackendRegCountFn, lib, "ggml_backend_reg_count")
	purego.RegisterLibFunc(&ggmlBackendRegGetFn, lib, "ggml_backend_reg_get")
	purego.RegisterLibFunc(&ggmlBackendRegByNameFn, lib, "ggml_backend_reg_by_name")
	return nil
}

// GGMLBackendLoadAll loads all backends using the default search paths.
func GGMLBackendLoadAll() {
	ggmlBackendLoadAllFn()
}

// GGMLBackendLoadAllFromPath loads all backends from a specific path.
func GGMLBackendLoadAllFromPath(path string) error {
	if path == "" {
		return errors.New("invalid path")
	}
	p := &[]byte(path + "\x00")[0]
	ggmlBackendLoadAllFromPath(p)
	return nil
}

// GGMLBackendUnload unloads a backend if loaded dynamically and unregisters it.
func GGMLBackendUnload(reg GGMLBackendReg) {
	if reg == 0 {
		return
	}
	ggmlBackendUnloadFn(reg)
}

// GGMLBackendDeviceCount returns the number of backend devices.
func GGMLBackendDeviceCount() uint64 {
	return uint64(ggmlBackendDevCountFn())
}

// GGMLBackendDeviceGet returns the backend device at the given index.
func GGMLBackendDeviceGet(index uint64) GGMLBackendDevice {
	return ggmlBackendDevGetFn(uintptr(index))
}

// GGMLBackendDeviceByName returns the backend device by its name.
func GGMLBackendDeviceByName(name string) GGMLBackendDevice {
	namePtr, _ := utils.BytePtrFromString(name)
	return ggmlBackendDevByNameFn(namePtr)
}

// GGMLBackendDeviceByType returns the backend device by its type.
func GGMLBackendDeviceByType(devType GGMLBackendDeviceType) GGMLBackendDevice {
	return ggmlBackendDevByTypeFn(int32(devType))
}

// GGMLBackendRegCount returns the number of backend registrations.
func GGMLBackendRegCount() uint64 {
	return uint64(ggmlBackendRegCountFn())
}

// GGMLBackendRegGet returns the backend registration at the given index.
func GGMLBackendRegGet(index uint64) GGMLBackendReg {
	return ggmlBackendRegGetFn(uintptr(index))
}

// GGMLBackendRegByName returns the backend registration by its name.
func GGMLBackendRegByName(name string) GGMLBackendReg {
	namePtr, _ := utils.BytePtrFromString(name)
	return ggmlBackendRegByNameFn(namePtr)
}
