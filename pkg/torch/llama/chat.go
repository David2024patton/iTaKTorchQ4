package llama

import (
	"unsafe"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/utils"
	"github.com/ebitengine/purego"
)

// --- purego direct-call function pointers ---
var (
	chatApplyTemplateFn    func(tmpl *byte, chat unsafe.Pointer, nMsg uintptr, addAss uint8, buf *byte, length int32) int32
	chatBuiltinTemplatesFn func(output **byte, length uintptr) int32
)

func loadChatFuncs(lib uintptr) error {
	purego.RegisterLibFunc(&chatApplyTemplateFn, lib, "llama_chat_apply_template")
	purego.RegisterLibFunc(&chatBuiltinTemplatesFn, lib, "llama_chat_builtin_templates")
	return nil
}

// NewChatMessage creates a new ChatMessage.
func NewChatMessage(role, content string) ChatMessage {
	r, err := utils.BytePtrFromString(role)
	if err != nil {
		return ChatMessage{}
	}

	c, err := utils.BytePtrFromString(content)
	if err != nil {
		return ChatMessage{}
	}

	return ChatMessage{Role: r, Content: c}
}

// ChatApplyTemplate applies a chat template to a slice of ChatMessage.
func ChatApplyTemplate(template string, chat []ChatMessage, addAssistantPrompt bool, buf []byte) int32 {
	tmpl, err := utils.BytePtrFromString(template)
	if err != nil {
		return 0
	}

	if len(chat) == 0 {
		return 0
	}

	c := unsafe.Pointer(&chat[0])
	nMsg := uintptr(len(chat))

	out := unsafe.SliceData(buf)
	bLen := int32(len(buf))

	var addAss uint8
	if addAssistantPrompt {
		addAss = 1
	}

	return chatApplyTemplateFn(tmpl, c, nMsg, addAss, out, bLen)
}

// ChatBuiltinTemplates returns a slice of built-in chat template names.
func ChatBuiltinTemplates() []string {
	// get the needed size
	var cOutput *byte
	count := chatBuiltinTemplatesFn(&cOutput, 0)

	if count == 0 {
		return nil
	}

	// now get the actual templates
	cStrings := make([]*byte, count)
	cFinalOutput := unsafe.SliceData(cStrings)

	chatBuiltinTemplatesFn(cFinalOutput, uintptr(count))

	templates := make([]string, count)
	for i, cStr := range cStrings {
		if cStr != nil {
			templates[i] = utils.BytePtrToString(cStr)
		}
	}

	return templates
}
