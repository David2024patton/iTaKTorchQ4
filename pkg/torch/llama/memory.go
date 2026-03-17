package llama

import (
	"errors"

	"github.com/ebitengine/purego"
)

// --- purego direct-call function pointers ---
var (
	memoryClearFn      func(mem Memory, data uint8)
	memorySeqRmFn      func(mem Memory, seqID int32, p0, p1 int32) uint8
	memorySeqCpFn      func(mem Memory, seqIDSrc, seqIDDst int32, p0, p1 int32)
	memorySeqKeepFn    func(mem Memory, seqID int32)
	memorySeqAddFn     func(mem Memory, seqID int32, p0, p1, delta int32)
	memorySeqDivFn     func(mem Memory, seqID int32, p0, p1 int32, d int32)
	memorySeqPosMinFn  func(mem Memory, seqID int32) int32
	memorySeqPosMaxFn  func(mem Memory, seqID int32) int32
	memoryCanShiftFn   func(mem Memory) uint8
)

func loadMemoryFuncs(lib uintptr) error {
	purego.RegisterLibFunc(&memoryClearFn, lib, "llama_memory_clear")
	purego.RegisterLibFunc(&memorySeqRmFn, lib, "llama_memory_seq_rm")
	purego.RegisterLibFunc(&memorySeqCpFn, lib, "llama_memory_seq_cp")
	purego.RegisterLibFunc(&memorySeqKeepFn, lib, "llama_memory_seq_keep")
	purego.RegisterLibFunc(&memorySeqAddFn, lib, "llama_memory_seq_add")
	purego.RegisterLibFunc(&memorySeqDivFn, lib, "llama_memory_seq_div")
	purego.RegisterLibFunc(&memorySeqPosMinFn, lib, "llama_memory_seq_pos_min")
	purego.RegisterLibFunc(&memorySeqPosMaxFn, lib, "llama_memory_seq_pos_max")
	purego.RegisterLibFunc(&memoryCanShiftFn, lib, "llama_memory_can_shift")
	return nil
}

var (
	errInvalidMemory = errors.New("invalid memory handle")
)

// MemoryClear clears the memory contents.
// If data == true, the data buffers will also be cleared together with the metadata.
func MemoryClear(mem Memory, data bool) error {
	if mem == 0 {
		return errInvalidMemory
	}
	var d uint8
	if data {
		d = 1
	}
	memoryClearFn(mem, d)
	return nil
}

// MemorySeqRm removes all tokens that belong to the specified sequence and have positions in [p0, p1).
// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails.
func MemorySeqRm(mem Memory, seqID SeqId, p0, p1 Pos) (bool, error) {
	if mem == 0 {
		return false, errInvalidMemory
	}
	return memorySeqRmFn(mem, int32(seqID), int32(p0), int32(p1)) != 0, nil
}

// MemorySeqCp copies all tokens from one sequence to another.
func MemorySeqCp(mem Memory, seqIDSrc, seqIDDst SeqId, p0, p1 Pos) error {
	if mem == 0 {
		return errInvalidMemory
	}
	memorySeqCpFn(mem, int32(seqIDSrc), int32(seqIDDst), int32(p0), int32(p1))
	return nil
}

// MemorySeqKeep removes all tokens that do not belong to the specified sequence.
func MemorySeqKeep(mem Memory, seqID SeqId) error {
	if mem == 0 {
		return errInvalidMemory
	}
	memorySeqKeepFn(mem, int32(seqID))
	return nil
}

// MemorySeqAdd adds a relative position delta to tokens in the specified sequence and range.
func MemorySeqAdd(mem Memory, seqID SeqId, p0, p1, delta Pos) error {
	if mem == 0 {
		return errInvalidMemory
	}
	memorySeqAddFn(mem, int32(seqID), int32(p0), int32(p1), int32(delta))
	return nil
}

// MemorySeqDiv divides the positions of tokens in the specified sequence and range by a factor.
func MemorySeqDiv(mem Memory, seqID SeqId, p0, p1 Pos, d int) error {
	if mem == 0 {
		return errInvalidMemory
	}
	memorySeqDivFn(mem, int32(seqID), int32(p0), int32(p1), int32(d))
	return nil
}

// MemorySeqPosMin returns the smallest position in the memory for the specified sequence.
func MemorySeqPosMin(mem Memory, seqID SeqId) (Pos, error) {
	if mem == 0 {
		return 0, errInvalidMemory
	}
	return Pos(memorySeqPosMinFn(mem, int32(seqID))), nil
}

// MemorySeqPosMax returns the largest position in the memory for the specified sequence.
func MemorySeqPosMax(mem Memory, seqID SeqId) (Pos, error) {
	if mem == 0 {
		return 0, errInvalidMemory
	}
	return Pos(memorySeqPosMaxFn(mem, int32(seqID))), nil
}

// MemoryCanShift checks if the memory supports shifting.
func MemoryCanShift(mem Memory) (bool, error) {
	if mem == 0 {
		return false, errInvalidMemory
	}
	return memoryCanShiftFn(mem) != 0, nil
}
