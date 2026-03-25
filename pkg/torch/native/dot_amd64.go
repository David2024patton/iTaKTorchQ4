//go:build amd64

package native

// Dot computes the dot product of two float32 slices using AVX2.
// The slices must have the same length. It uses raw hardware assembly.
func Dot(a, b []float32) float32
