// cluster_discovery.go implements LAN auto-discovery and health management
// for the distributed Torch cluster.
//
// HOW DISCOVERY WORKS:
//   1. On startup, each node broadcasts a UDP beacon to 255.255.255.255:39272
//      every 10 seconds containing its name and HTTP address.
//   2. All nodes listen on the same UDP port for incoming beacons.
//   3. When a new beacon is received, the node automatically registers the
//      peer by calling /v1/capabilities to probe its hardware.
//   4. A background goroutine removes peers that haven't been seen in 2 minutes.
//
// This means zero manual configuration - just start Torch on each machine
// and they find each other automatically.
package torch

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"
)

const (
	// discoveryPort is the UDP port for broadcast beacons.
	discoveryPort = 39272

	// beaconInterval is how often we announce ourselves.
	beaconInterval = 10 * time.Second

	// healthCheckInterval is how often we probe all peers.
	healthCheckInterval = 30 * time.Second

	// peerTimeout is how long before a peer is removed from the registry.
	peerTimeout = 2 * time.Minute
)

// discoveryBeacon is the UDP payload broadcast to the LAN.
type discoveryBeacon struct {
	Name    string `json:"name"`    // e.g. "beast"
	Address string `json:"address"` // e.g. "192.168.0.100:39271"
	Version string `json:"version"` // e.g. "torch/1.0"
}

// clusterDiscovery manages LAN discovery and health monitoring.
type clusterDiscovery struct {
	mu       sync.Mutex
	stopCh   chan struct{}
	server   *Server
	selfAddr string // our own address (to ignore self-beacons)
	selfName string
}

// StartClusterDiscovery begins broadcasting and listening for peers.
// Call this after the HTTP server is started.
func (s *Server) StartClusterDiscovery(selfName, selfAddr string) {
	d := &clusterDiscovery{
		stopCh:   make(chan struct{}),
		server:   s,
		selfAddr: selfAddr,
		selfName: selfName,
	}

	// Start UDP listener.
	go d.listenForBeacons()

	// Start UDP broadcaster.
	go d.broadcastBeacon()

	// Start health checker / cleanup.
	go d.healthLoop()

	s.debugf("[DISCOVERY] Started: name=%s addr=%s beacon_port=%d", selfName, selfAddr, discoveryPort)
}

// broadcastBeacon sends a UDP beacon to the LAN broadcast address.
func (d *clusterDiscovery) broadcastBeacon() {
	beacon := discoveryBeacon{
		Name:    d.selfName,
		Address: d.selfAddr,
		Version: "torch/1.0",
	}
	payload, _ := json.Marshal(beacon)

	ticker := time.NewTicker(beaconInterval)
	defer ticker.Stop()

	// Send one immediately.
	d.sendBeacon(payload)

	for {
		select {
		case <-d.stopCh:
			return
		case <-ticker.C:
			d.sendBeacon(payload)
		}
	}
}

// sendBeacon broadcasts a single UDP beacon packet.
func (d *clusterDiscovery) sendBeacon(payload []byte) {
	conn, err := net.Dial("udp4", fmt.Sprintf("255.255.255.255:%d", discoveryPort))
	if err != nil {
		return // silently skip if network is down
	}
	defer conn.Close()
	conn.Write(payload)
}

// listenForBeacons listens for discovery beacons from other Torch nodes.
func (d *clusterDiscovery) listenForBeacons() {
	addr := &net.UDPAddr{Port: discoveryPort}
	conn, err := net.ListenUDP("udp4", addr)
	if err != nil {
		d.server.debugf("[DISCOVERY] Failed to listen on UDP %d: %v", discoveryPort, err)
		return
	}
	defer conn.Close()

	buf := make([]byte, 1024)
	for {
		select {
		case <-d.stopCh:
			return
		default:
		}

		conn.SetReadDeadline(time.Now().Add(5 * time.Second))
		n, _, err := conn.ReadFromUDP(buf)
		if err != nil {
			continue // timeout or error, just retry
		}

		var beacon discoveryBeacon
		if err := json.Unmarshal(buf[:n], &beacon); err != nil {
			continue
		}

		// Ignore our own beacons.
		if beacon.Address == d.selfAddr {
			continue
		}

		// Check if we already know this peer.
		cluster.mu.RLock()
		_, known := cluster.peers[beacon.Address]
		cluster.mu.RUnlock()

		if known {
			// Update LastSeen.
			cluster.mu.Lock()
			if p, ok := cluster.peers[beacon.Address]; ok {
				p.LastSeen = time.Now()
			}
			cluster.mu.Unlock()
			continue
		}

		// New peer! Probe its capabilities.
		go d.registerPeer(beacon)
	}
}

// registerPeer probes a newly-discovered peer and adds it to the registry.
func (d *clusterDiscovery) registerPeer(beacon discoveryBeacon) {
	capURL := fmt.Sprintf("http://%s/v1/capabilities", beacon.Address)
	client := &http.Client{Timeout: 5 * time.Second}

	resp, err := client.Get(capURL)
	if err != nil {
		d.server.debugf("[DISCOVERY] Failed to probe %s (%s): %v", beacon.Name, beacon.Address, err)
		return
	}
	defer resp.Body.Close()

	var caps Capabilities
	if err := json.NewDecoder(resp.Body).Decode(&caps); err != nil {
		d.server.debugf("[DISCOVERY] Invalid capabilities from %s: %v", beacon.Name, err)
		return
	}

	peer := &ClusterPeer{
		Address:      beacon.Address,
		Name:         beacon.Name,
		Capabilities: &caps,
		LastSeen:     time.Now(),
		Healthy:      true,
	}

	cluster.mu.Lock()
	cluster.peers[beacon.Address] = peer
	cluster.mu.Unlock()

	d.server.debugf("[DISCOVERY] Auto-registered peer: %s (%s) strategy=%s max_parallel=%d",
		beacon.Name, beacon.Address, caps.Strategy, caps.MaxParallel)
}

// healthLoop periodically probes all peers and removes dead ones.
func (d *clusterDiscovery) healthLoop() {
	ticker := time.NewTicker(healthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-d.stopCh:
			return
		case <-ticker.C:
			d.checkAndCleanPeers()
		}
	}
}

// checkAndCleanPeers probes each peer's /health endpoint and removes
// peers that haven't responded in peerTimeout.
func (d *clusterDiscovery) checkAndCleanPeers() {
	cluster.mu.RLock()
	peers := make([]*ClusterPeer, 0, len(cluster.peers))
	for _, p := range cluster.peers {
		peers = append(peers, p)
	}
	cluster.mu.RUnlock()

	if len(peers) == 0 {
		return
	}

	client := &http.Client{Timeout: clusterHealthTimeout}
	var wg sync.WaitGroup

	for _, peer := range peers {
		wg.Add(1)
		go func(p *ClusterPeer) {
			defer wg.Done()

			resp, err := client.Get(fmt.Sprintf("http://%s/health", p.Address))
			if err != nil {
				p.Healthy = false
			} else {
				resp.Body.Close()
				p.Healthy = resp.StatusCode == http.StatusOK
				if p.Healthy {
					p.LastSeen = time.Now()
				}
			}
		}(peer)
	}

	wg.Wait()

	// Remove peers that haven't been seen in peerTimeout.
	cluster.mu.Lock()
	for addr, p := range cluster.peers {
		if time.Since(p.LastSeen) > peerTimeout {
			d.server.debugf("[DISCOVERY] Removing dead peer: %s (%s) last_seen=%s ago",
				p.Name, addr, time.Since(p.LastSeen).Round(time.Second))
			delete(cluster.peers, addr)
		}
	}
	cluster.mu.Unlock()
}
