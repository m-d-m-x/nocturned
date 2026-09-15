package utils

// Bluetooth
type BluetoothDeviceInfo struct {
	Address           string `json:"address"`
	Name              string `json:"name"`
	Alias             string `json:"alias"`
	Class             string `json:"class"`
	Icon              string `json:"icon"`
	Paired            bool   `json:"paired"`
	Trusted           bool   `json:"trusted"`
	Blocked           bool   `json:"blocked"`
	Connected         bool   `json:"connected"`
	LegacyPairing     bool   `json:"legacyPairing"`
	BatteryPercentage int    `json:"batteryPercentage,omitempty"`
}

type PairingRequest struct {
	Device      string
	Passkey     string
	RequestType string
}

// WebSocket
type WebSocketEvent struct {
	Type    string      `json:"type"`
	Payload interface{} `json:"payload"`
}

type PairingStartedPayload struct {
	Address    string `json:"address"`
	PairingKey string `json:"pairingKey"`
}

type DeviceConnectedPayload struct {
	Address string               `json:"address"`
	Device  *BluetoothDeviceInfo `json:"device,omitempty"`
}

type DeviceDisconnectedPayload struct {
	Address string `json:"address"`
}

type DevicePairedPayload struct {
	Device *BluetoothDeviceInfo `json:"device"`
}

type NetworkConnectedPayload struct {
	Address string `json:"address"`
}

type VoiceTranscriptPayload struct {
	Text      string `json:"text,omitempty"`
	Error     string `json:"error,omitempty"`
	Provider  string `json:"provider,omitempty"`
	ElapsedMs int64  `json:"elapsedMs,omitempty"`
}

// VoiceStatePayload reports capture lifecycle transitions the client did not
// initiate - most importantly the silence detector auto-stopping - so the UI
// can follow along without owning the stop itself.
type VoiceStatePayload struct {
	State string `json:"state"` // wake | recording | transcribing | discarded | cancelled

	// Present on "transcribing" so the client can log what was actually
	// captured: why it stopped, how long, how big, and the RMS that tripped it.
	Reason     string  `json:"reason,omitempty"` // silence | cap | manual
	DurationMs int64   `json:"durationMs,omitempty"`
	Bytes      int64   `json:"bytes,omitempty"`
	PeakRMS    float64 `json:"peakRms,omitempty"`
	FloorRMS   float64 `json:"floorRms,omitempty"`
	Threshold  float64 `json:"threshold,omitempty"`
	Windows    uint32  `json:"windows,omitempty"`

	// Present on "wake": how confident the detector was, so the client can
	// surface or log a marginal trigger.
	Score float32 `json:"score,omitempty"`
}
