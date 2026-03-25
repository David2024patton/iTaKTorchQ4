//go:build !amd64

package native

// Dot computes the dot product of two float32 slices using standard Go scalar math.
// This is the fallback for ARM or 32-bit platforms.
func Dot(a, b []float32) float32 {
	var sum float32
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}
