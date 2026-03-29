package torch

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// PERSONAS matches the Python-side high-fidelity voice definitions.
var PERSONAS = map[string]string{
	"Vivian":  "A professional, clear, and warm American female voice. Natural and authoritative.",
	"Dylan":   "An energetic, youthful, and friendly American male voice. engaging and bright.",
	"Ryan":    "A deep, resonant, and calm British male voice. Sophisticated and steady.",
	"Serena":  "A warm, bilingual Spanish-accented female voice. Smooth and inviting.",
	"Camille": "A soft, expressive female voice with a charming French accent.",
}

func (s *Server) handleSpeech(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, "failed to read body")
		return
	}
	defer r.Body.Close()

	var req SpeechRequest
	if err := json.Unmarshal(body, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	if req.Input == "" {
		s.writeError(w, http.StatusBadRequest, "input text is required")
		return
	}

	// 1. Build Persona Instruction
	instruct := buildSpeechPersona(req)

	// 2. Dispatch to Engine
	// Note: Currently, iTaK Torch (Go) is a text inference engine.
	// We are implementing the shell here while the GGUF-TTS kernels are finalized.
	// For "No Python" mode, we will eventually call s.engine.Synthesize().
	
	fmt.Printf("[iTaK Torch] Speech: arch=%s voice=%s input_len=%d\n", req.Arch, req.Voice, len(req.Input))
	if instruct != "" {
		fmt.Printf("[iTaK Torch] Speech: instruct=\"%s\"\n", instruct)
	}

	// TODO: Integrate GGUF-TTS synthesis here.
	// For now, return a placeholder error that explains we are in the Go Transition Phase.
	s.writeError(w, http.StatusNotImplemented, "Go-Native GGUF-TTS inference is currently in the Unification Phase. Use 'itak' arch (Python sidecar) for live synthesis during migration.")
}

func buildSpeechPersona(req SpeechRequest) string {
	var parts []string

	// Base Persona
	if p, ok := PERSONAS[req.Voice]; ok {
		parts = append(parts, p)
	}

	// Speed
	if req.Speed > 0 && req.Speed != 1.0 {
		if req.Speed < 0.8 {
			parts = append(parts, "Speak very slowly.")
		} else if req.Speed > 1.2 {
			parts = append(parts, "Speak quickly.")
		}
	}

	return strings.Join(parts, " ")
}
