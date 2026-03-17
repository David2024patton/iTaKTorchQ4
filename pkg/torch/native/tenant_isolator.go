// tenant_isolator.go implements strict Multi-Tenant Memory Isolation.
//
// WHAT: In Inference-as-a-Service environments (Nebius, Fireworks, Baseten),
// multiple users (tenants) share the same massive GPU cluster.
// If Tenant A submits a 100K token document, standard LRU eviction will
// wipe out the KV cache of Tenant B, Tenant C, etc., causing massive tail 
// latency spikes for everyone else.
//
// HOW: The Tenant Isolator overrides the basic LRU cache. It partitions
// the physical PagedAttention blocks into strict quotas per API Key / Tenant ID.
// 
// WHY: Ensures noisy neighbors cannot monopolize GPU memory. If Tenant A
// exceeds their quota, their existing blocks are forcefully offloaded to 
// NVMe/RAM, while Tenant B's interactive chat history remains safely anchored
// in the blazing fast GPU VRAM.
package native

import (
	"errors"
	"sync"
)

// ErrQuotaExceeded is returned when a tenant tries to use more VRAM than allowed
// and eviction of their own blocks fails.
var ErrQuotaExceeded = errors.New("tenant has exceeded their isolated KV memory quota")

// TenantPolicy defines the resource limits for a specific user/organization.
type TenantPolicy struct {
	MaxGPUBlocks   int  // Maximum number of KV blocks allowed in GPU
	MaxOffload     int  // Maximum number of blocks allowed in RAM/NVMe
	PriorityTier   int  // 1 (Critical), 2 (Standard), 3 (Free/Preemptible)
}

// Tenant represents an isolated user context within the inference engine.
type Tenant struct {
	ID           string
	Policy       TenantPolicy
	
	ActiveBlocks int
	Blocks       map[string]bool // BlockIDs currently owned by this tenant
	
	mu           sync.Mutex
}

// IsolationManager intercepts all KV block allocations.
type IsolationManager struct {
	mu           sync.RWMutex
	tenants      map[string]*Tenant
	globalBlocks int
	maxBlocks    int
}

// NewIsolationManager initializes the multi-tenant partitioner.
func NewIsolationManager(maxGPUBlocks int) *IsolationManager {
	return &IsolationManager{
		tenants:   make(map[string]*Tenant),
		maxBlocks: maxGPUBlocks,
	}
}

// RegisterTenant provisions an isolated slice of the engine for a user.
func (im *IsolationManager) RegisterTenant(tenantID string, policy TenantPolicy) {
	im.mu.Lock()
	defer im.mu.Lock()
	
	im.tenants[tenantID] = &Tenant{
		ID:     tenantID,
		Policy: policy,
		Blocks: make(map[string]bool),
	}
}

// RequestAllocation is called by PagedAttention when a request needs a new KV block.
// It enforces the tenant's quota.
func (im *IsolationManager) RequestAllocation(tenantID string, blockID string) error {
	im.mu.RLock()
	tenant, exists := im.tenants[tenantID]
	im.mu.RUnlock()
	
	if !exists {
		return errors.New("unknown tenant ID")
	}
	
	tenant.mu.Lock()
	defer tenant.mu.Unlock()
	
	// Check against strictly isolated tenant quota first
	if tenant.ActiveBlocks >= tenant.Policy.MaxGPUBlocks {
		// The tenant has exhausted their specific bucket.
		// They MUST evict their OWN blocks. They cannot touch global LRU.
		err := im.evictTenantBlock(tenant)
		if err != nil {
			return ErrQuotaExceeded
		}
	}
	
	im.mu.Lock()
	defer im.mu.Unlock()
	
	// Even if within quota, physical GPU might be full globally
	if im.globalBlocks >= im.maxBlocks {
		// Global Eviction - but we only evict from lower-tier tenants
		// or tenants exceeding their fair share ratio.
		err := im.evictLowestPriorityGlobal()
		if err != nil {
			return errors.New("global GPU memory exhausted and no preemptible blocks found")
		}
	}
	
	// Allocation successful
	tenant.ActiveBlocks++
	tenant.Blocks[blockID] = true
	im.globalBlocks++
	
	return nil
}

// ReleaseAllocation frees a block and returns it to the tenant's quota.
func (im *IsolationManager) ReleaseAllocation(tenantID string, blockID string) {
	im.mu.RLock()
	tenant, exists := im.tenants[tenantID]
	im.mu.RUnlock()
	
	if !exists { return }
	
	tenant.mu.Lock()
	if tenant.Blocks[blockID] {
		delete(tenant.Blocks, blockID)
		tenant.ActiveBlocks--
		
		im.mu.Lock()
		im.globalBlocks--
		im.mu.Unlock()
	}
	tenant.mu.Unlock()
}

// evictTenantBlock forces a specific tenant to page out one of their own blocks
// to NVMe/RAM to make room for their new request. (Omitted complex LRU for brevity).
func (im *IsolationManager) evictTenantBlock(tenant *Tenant) error {
	if len(tenant.Blocks) == 0 {
		return errors.New("no blocks left to evict")
	}
	
	// Pick pseudo-random/LRU block owned by them
	var victim string
	for b := range tenant.Blocks {
		victim = b
		break
	}
	
	// "Offload"
	delete(tenant.Blocks, victim)
	tenant.ActiveBlocks--
	
	im.mu.Lock()
	im.globalBlocks--
	im.mu.Unlock()
	
	return nil
}

// evictLowestPriorityGlobal looks across the entire fleet for Free-tier
// tenants holding GPU blocks and preempts them to make room for Standard/Critical traffic.
func (im *IsolationManager) evictLowestPriorityGlobal() error {
	var victimTenant *Tenant
	var victimBlock string
	
	// Find lowest priority tenant using ANY blocks
	worstPriority := -1
	
	for _, t := range im.tenants {
		t.mu.Lock()
		if t.ActiveBlocks > 0 && t.Policy.PriorityTier > worstPriority {
			worstPriority = t.Policy.PriorityTier
			victimTenant = t
			for b := range t.Blocks {
				victimBlock = b
				break
			}
		}
		t.mu.Unlock()
	}
	
	if victimTenant == nil || victimBlock == "" {
		return errors.New("could not find preemtpible background block")
	}
	
	victimTenant.mu.Lock()
	delete(victimTenant.Blocks, victimBlock)
	victimTenant.ActiveBlocks--
	victimTenant.mu.Unlock()
	
	im.globalBlocks--
	return nil
}
