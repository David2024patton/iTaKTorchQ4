package llama

import (
	"flag"
	"os"
	"strings"
	"testing"
)

var (
	benchModel    Model
	benchCtx      Context
	benchTemplate string
	benchReady    bool
)

var (
	nCtx       int
	device     string
	nGpuLayers int
	nThreads   int
)

func init() {
	flag.IntVar(&nCtx, "nctx", 8192, "number of context tokens for llama.Context")
	flag.StringVar(&device, "device", "", "comma-separated list of devices to use for benchmarking (e.g. 'CUDA0')")
	flag.IntVar(&nGpuLayers, "ngpulayers", -1, "number of layers to offload to GPU (-1 = default/auto)")
	flag.IntVar(&nThreads, "nthreads", 0, "number of CPU threads (0 = auto)")
}

func TestMain(m *testing.M) {
	flag.Parse() // Parse flags before running tests

	code := m.Run()

	if benchReady {
		benchmarkTeardown()
	}

	os.Exit(code)
}

func benchmarkSetupOnce(b *testing.B) {
	if benchReady {
		return
	}

	modelFile := benchmarkModelFileName(b)

	benchmarkSetup(b)

	mparams := ModelDefaultParams()
	mparams.UseMmap = 0

	if nGpuLayers >= 0 {
		mparams.NGpuLayers = int32(nGpuLayers)
	} else {
		// Auto-offload: load all layers to GPU when available.
		// Without this, CUDA/Vulkan backends get 0 GPU layers = CPU only.
		mparams.NGpuLayers = 999
	}

	if device != "" {
		devs := []GGMLBackendDevice{}
		devices := strings.Split(device, ",")
		for _, d := range devices {
			dev := GGMLBackendDeviceByName(d)
			if dev == 0 {
				b.Fatalf("unknown device: %s", d)
			}
			devs = append(devs, dev)
		}

		mparams.SetDevices(devs)
	}

	model, err := ModelLoadFromFile(modelFile, mparams)
	if err != nil {
		b.Fatalf("ModelLoadFromFile failed: %v", err)
	}
	benchModel = model

	params := ContextDefaultParams()
	params.NBatch = 4096
	params.NCtx = uint32(nCtx)

	// Enable flash attention for GPU (20-30% faster attention computation).
	params.FlashAttentionType = FlashAttentionTypeEnabled

	// Quantize KV cache to q8_0 (halves VRAM usage with minimal quality loss).
	params.TypeK = GGMLTypeQ8_0
	params.TypeV = GGMLTypeQ8_0

	if nThreads > 0 {
		params.NThreads = int32(nThreads)
		params.NThreadsBatch = int32(nThreads)
	}

	ctx, err := InitFromModel(model, params)
	if err != nil {
		b.Fatalf("InitFromModel failed: %v", err)
	}
	benchCtx = ctx

	benchTemplate = ModelChatTemplate(model, "")

	benchReady = true
}

func benchmarkTeardown() {
	Free(benchCtx)
	ModelFree(benchModel)

	LogSet(LogNormal)
	Close()
}
