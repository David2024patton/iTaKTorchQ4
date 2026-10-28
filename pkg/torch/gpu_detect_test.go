package torch

import "testing"

// Regression tests for the 2026-10-28 AMD detection failure: on Linux the
// prober never read AMD VRAM, so an RX 7900 XTX inventoried as 0 MiB, the
// best-discrete scan skipped it, and AutoConfig forced CPU-only.

func TestAMDDetectionHelpers(t *testing.T) {
	const gib = 1024 * 1024 * 1024

	t.Run("pciSlotFromPath", func(t *testing.T) {
		cases := map[string]string{
			"/sys/devices/pci0000:00/0000:00:01.0/0000:03:00.0/drm/card2": "03:00.0",
			"/sys/devices/pci0000:00/0000:00:02.0/drm/card1":              "00:02.0",
			"/sys/class/drm/card2":                                        "",
			"":                                                            "",
		}
		for in, want := range cases {
			if got := pciSlotFromPath(in); got != want {
				t.Errorf("pciSlotFromPath(%q) = %q, want %q", in, got, want)
			}
		}
	})

	t.Run("lspciSlot", func(t *testing.T) {
		cases := map[string]string{
			"03:00.0 VGA compatible controller: Advanced Micro Devices": "03:00.0",
			"0000:03:00.0 VGA compatible controller: AMD":               "03:00.0",
			"": "",
		}
		for in, want := range cases {
			if got := lspciSlot(in); got != want {
				t.Errorf("lspciSlot(%q) = %q, want %q", in, got, want)
			}
		}
	})

	t.Run("isDRMCardName", func(t *testing.T) {
		for _, name := range []string{"card0", "card1", "card12"} {
			if !isDRMCardName(name) {
				t.Errorf("isDRMCardName(%q) = false, want true", name)
			}
		}
		for _, name := range []string{"card0-DP-1", "card", "renderD128", "cardA"} {
			if isDRMCardName(name) {
				t.Errorf("isDRMCardName(%q) = true, want false", name)
			}
		}
	})

	t.Run("amdDedicatedVRAM", func(t *testing.T) {
		xtx := "Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]"
		if !amdDedicatedVRAM(xtx, 24560*1024*1024) {
			t.Error("RX 7900 XTX with 24 GiB must count as discrete")
		}
		if amdDedicatedVRAM("AMD Radeon 780M Graphics", 2*gib) {
			t.Error("780M APU with a 2 GiB carveout must not count as discrete")
		}
		if amdDedicatedVRAM("AMD Radeon Vega 8 Graphics", 512*1024*1024) {
			t.Error("Vega APU must not count as discrete")
		}
		if amdDedicatedVRAM(xtx, 0) {
			t.Error("zero VRAM must not count as discrete")
		}
		if !amdDedicatedVRAM("AMD Radeon RX 7600", 8*gib) {
			t.Error("RX 7600 with 8 GiB must count as discrete")
		}
	})
}
