import { useState, useEffect, useMemo } from 'react';
import './App.css';

/** Memory mode: discrete GPU or unified memory. */
type MemoryMode = 'DISCRETE_GPU' | 'UNIFIED_MEMORY';

/** Model quantization set: F32, F16, Q8, Q6, Q5, Q4, Q3, Q2, GPTQ, AWQ. */
type ModelQuantization =
  | 'F32'
  | 'F16'
  | 'Q8'
  | 'Q6'
  | 'Q5'
  | 'Q4'
  | 'Q3'
  | 'Q2'
  | 'GPTQ'
  | 'AWQ';

/** KV cache quantization: F32, F16, Q8, Q5, Q4. */
type KvCacheQuantization = 'F32' | 'F16' | 'Q8' | 'Q5' | 'Q4';

/** Recommendation for final output. */
interface Recommendation {
  gpuType: string;         // e.g., 'Single 24GB GPU' or 'Unified memory...'
  vramNeeded: string;      // e.g., "32.5"
  fitsUnified: boolean;    // relevant if memoryMode = 'UNIFIED_MEMORY'
  systemRamNeeded: number; // in GB
  gpusRequired: number;    // discrete GPUs required (0 if doesn't fit)
}

// SVG Icons for UI
const ModelIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 6V4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v2"></path>
    <path d="M4 18v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"></path>
    <rect x="2" y="6" width="20" height="12" rx="2"></rect>
    <circle cx="12" cy="12" r="2"></circle>
  </svg>
);

const HardwareIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="5" width="20" height="14" rx="2"></rect>
    <line x1="2" y1="10" x2="22" y2="10"></line>
  </svg>
);

const ResultsIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 20v-6M6 20V10M18 20V4"></path>
  </svg>
);

const SuccessIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
    <polyline points="22 4 12 14.01 9 11.01"></polyline>
  </svg>
);

const WarningIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
    <line x1="12" y1="9" x2="12" y2="13"></line>
    <line x1="12" y1="17" x2="12.01" y2="17"></line>
  </svg>
);

const BrainIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.54Z"></path>
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.54Z"></path>
  </svg>
);

function App() {
  // -----------------------------------
  // 1. STATE
  // -----------------------------------

  // UI state
  const [showCalculator, setShowCalculator] = useState<boolean>(false);
  const [backgroundGradient, setBackgroundGradient] = useState<string>('gradient-tiny');

  // Model config
  const [params, setParams] = useState<number>(65); // Billions of parameters
  const [modelQuant, setModelQuant] = useState<ModelQuantization>('Q4');

  // KV Cache
  const [useKvCache, setUseKvCache] = useState<boolean>(true);
  const [kvCacheQuant, setKvCacheQuant] = useState<KvCacheQuantization>('F16');

  // Misc
  const [contextLength, setContextLength] = useState<number>(4096);
  const [memoryMode, setMemoryMode] = useState<MemoryMode>('DISCRETE_GPU');
  const [systemMemory, setSystemMemory] = useState<number>(128); // in GB
  const [gpuVram, setGpuVram] = useState<number>(24); // in GB, default 24GB

  // -----------------------------------
  // 2. HELPER FUNCTIONS
  // -----------------------------------

  // (A) Bits-based multiplier for the main model
  const getModelQuantFactor = (q: ModelQuantization): number => {
    switch (q) {
      case 'F32': return 4.0;
      case 'F16': return 2.0;
      case 'Q8': return 1.0;
      case 'Q6': return 0.75;
      case 'Q5': return 0.625;
      case 'Q4': return 0.5;
      case 'Q3': return 0.375;
      case 'Q2': return 0.25;
      case 'GPTQ': return 0.4;
      case 'AWQ': return 0.35;
      default: return 1.0;   // fallback
    }
  };

  // (B) Bits-based multiplier for KV cache
  const getKvCacheQuantFactor = (k: KvCacheQuantization): number => {
    switch (k) {
      case 'F32': return 4.0;
      case 'F16': return 2.0;
      case 'Q8': return 1.0;
      case 'Q5': return 0.625;
      case 'Q4': return 0.5;
      default: return 1.0;   // fallback
    }
  };

  /**
   * (C) Calculate VRAM for single-user inference.
   * Split into Model Memory + KV Cache Memory.
   */
  const calculateRequiredVram = (): number => {
    // 1) Model memory
    const modelFactor = getModelQuantFactor(modelQuant);
    const baseModelMem = params * modelFactor; // GB if 1B params

    // 2) Context scaling (just as before)
    let contextScale = contextLength / 2048;
    if (contextScale < 1) contextScale = 1;
    const modelMem = baseModelMem * contextScale;

    // 3) KV cache memory (if enabled)
    let kvCacheMem = 0;
    if (useKvCache) {
      const kvFactor = getKvCacheQuantFactor(kvCacheQuant);
      const alpha = 0.2; // fraction representing typical KV overhead
      kvCacheMem = params * kvFactor * contextScale * alpha;
    }

    // 4) total
    return modelMem + kvCacheMem;
  };

  // For unified memory, up to 75% of system RAM can be used as VRAM
  const getMaxUnifiedVram = (memGB: number): number => memGB * 0.75;

  // Decide discrete GPU vs. unified memory usage
  const calculateHardwareRecommendation = (): Recommendation => {
    const requiredVram = calculateRequiredVram();
    const recSystemMemory = systemMemory;

    if (memoryMode === 'UNIFIED_MEMORY') {
      const unifiedLimit = getMaxUnifiedVram(recSystemMemory);
      if (requiredVram <= unifiedLimit) {
        return {
          gpuType: 'Unified memory (ex: Apple silicon, AMD Ryzen™ Al Max+ 395)',
          vramNeeded: requiredVram.toFixed(1),
          fitsUnified: true,
          systemRamNeeded: recSystemMemory,
          gpusRequired: 1,
        };
      } else {
        return {
          gpuType: 'Unified memory (insufficient)',
          vramNeeded: requiredVram.toFixed(1),
          fitsUnified: false,
          systemRamNeeded: recSystemMemory,
          gpusRequired: 0,
        };
      }
    }

    // Discrete GPU
    const singleGpuVram = gpuVram;
    if (requiredVram <= singleGpuVram) {
      return {
        gpuType: `Single ${singleGpuVram}GB GPU`,
        vramNeeded: requiredVram.toFixed(1),
        fitsUnified: false,
        systemRamNeeded: Math.max(recSystemMemory, requiredVram),
        gpusRequired: 1,
      };
    } else {
      // multiple GPUs
      const count = Math.ceil(requiredVram / singleGpuVram);
      return {
        gpuType: `Discrete GPUs (${singleGpuVram}GB each)`,
        vramNeeded: requiredVram.toFixed(1),
        fitsUnified: false,
        systemRamNeeded: Math.max(recSystemMemory, requiredVram),
        gpusRequired: count,
      };
    }
  };

  /** Estimate on-disk model size (GB). We do NOT factor in KV here. */
  const calculateOnDiskSize = (): number => {
    let bitsPerParam: number;
    switch (modelQuant) {
      case 'F32': bitsPerParam = 32; break;
      case 'F16': bitsPerParam = 16; break;
      case 'Q8': bitsPerParam = 8; break;
      case 'Q6': bitsPerParam = 6; break;
      case 'Q5': bitsPerParam = 5; break;
      case 'Q4': bitsPerParam = 4; break;
      case 'Q3': bitsPerParam = 3; break;
      case 'Q2': bitsPerParam = 2; break;
      case 'GPTQ': bitsPerParam = 4; break;
      case 'AWQ': bitsPerParam = 4; break;
      default: bitsPerParam = 8; break;
    }

    const totalBits = params * 1e9 * bitsPerParam;
    const bytes = totalBits / 8;
    const gigabytes = bytes / 1e9;
    const overheadFactor = 1.1; // ~10% overhead
    return gigabytes * overheadFactor;
  };

  const handleInputChange = (
    event: React.ChangeEvent<HTMLInputElement>,
    setter: React.Dispatch<React.SetStateAction<number>>
  ) => {
    const newValue = Number(event.target.value);
    if (!isNaN(newValue)) {
      setter(newValue);
    }
  };

  // Determine the utilization class for the progress bar
  const getUtilizationClass = (utilizationPercentage: number): string => {
    if (utilizationPercentage < 50) return 'low';
    if (utilizationPercentage < 75) return 'medium';
    if (utilizationPercentage < 90) return 'high';
    return 'extreme';
  };

  // Update background gradient based on total complexity
  // Calculate a complexity score based on all factors
  const calculateComplexity = useMemo(() => {
    // Base score from parameter count (heaviest weight)
    let baseScore = params;
    
    // Adjust for quantization (better quantization reduces complexity)
    const quantMultiplier = getModelQuantFactor(modelQuant) / 4.0; // Normalize to 0-1 scale (F32 = 1)
    baseScore *= quantMultiplier;
    
    // Adjust for context length
    const contextFactor = contextLength / 2048;
    baseScore *= Math.max(1, contextFactor);
    
    // Add KV cache complexity if enabled
    if (useKvCache) {
      const kvFactor = getKvCacheQuantFactor(kvCacheQuant);
      baseScore *= (1 + 0.2 * kvFactor); // 20% additional complexity from KV cache
    }
    
    return baseScore;
  }, [params, modelQuant, contextLength, useKvCache, kvCacheQuant]);

  // Update the background gradient based on the complexity
  useEffect(() => {
    // Define thresholds for different gradients based on complexity
    if (calculateComplexity <= 15) {
      setBackgroundGradient('gradient-tiny');       // 1-15B effective complexity
    } else if (calculateComplexity <= 32) {
      setBackgroundGradient('gradient-small');      // 16-32B effective complexity
    } else if (calculateComplexity <= 70) {
      setBackgroundGradient('gradient-medium');     // 33-70B effective complexity
    } else if (calculateComplexity <= 200) {
      setBackgroundGradient('gradient-large');      // 71-200B effective complexity
    } else if (calculateComplexity <= 500) {
      setBackgroundGradient('gradient-xlarge');     // 201-500B effective complexity
    } else {
      setBackgroundGradient('gradient-massive');    // 501B+ effective complexity
    }
  }, [calculateComplexity]);

  /**
   * Custom number input with styled up/down controls
   */
  const NumberInput = ({ 
    value, 
    onChange, 
    min, 
    max, 
    step = 1
  }: { 
    value: number; 
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void; 
    min: number; 
    max: number; 
    step?: number;
  }) => {
    const handleIncrement = () => {
      const newValue = Math.min(max, value + step);
      const event = {
        target: { value: String(newValue) }
      } as React.ChangeEvent<HTMLInputElement>;
      onChange(event);
    };

    const handleDecrement = () => {
      const newValue = Math.max(min, value - step);
      const event = {
        target: { value: String(newValue) }
      } as React.ChangeEvent<HTMLInputElement>;
      onChange(event);
    };

    return (
      <div className="number-input-container">
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={onChange}
          className="text-input"
        />
        <div className="number-controls">
          <div className="number-control-up" onClick={handleIncrement}>+</div>
          <div className="number-control-down" onClick={handleDecrement}>−</div>
        </div>
      </div>
    );
  };

  // -----------------------------------
  // 3. CALCULATE & RENDER
  // -----------------------------------
  const recommendation = calculateHardwareRecommendation();
  const onDiskSize = calculateOnDiskSize();
  const vramNeeded = parseFloat(recommendation.vramNeeded);
  const utilizationPercentage = (vramNeeded / gpuVram) * 100;
  const utilizationClass = getUtilizationClass(utilizationPercentage);

  return (
    <div className="App">
      {/* Dynamic background that changes based on requirements */}
      <div className={`app-background ${backgroundGradient}`}></div>

      {!showCalculator ? (
        <section className="welcome-section">
          <div className="orb orb-1"></div>
          <div className="orb orb-2"></div>
          <div className="orb orb-3"></div>
          <div className="demo-cycler"></div>
          
          <div className="welcome-content">
            <h1 className="welcome-title">
              <BrainIcon /> LLM Inference Calculator
            </h1>
            <p className="welcome-description">
              Instantly determine the hardware requirements for running Large Language Models locally.
              Get personalized GPU and memory recommendations based on model parameters, quantization level, and context length.
            </p>
            <button className="cta-button" onClick={() => setShowCalculator(true)}>
              Calculate Your Requirements
            </button>
          </div>
        </section>
      ) : (
        <div className="main-content">
          <div className="layout">
            {/* Left Panel: Inputs */}
            <div className="card input-panel">
              <div className="section">
                <h2 className="section-title">
                  <ModelIcon /> Model Configuration
                </h2>

                <div className="form-group">
                  <label className="form-label">Number of Parameters (Billions)</label>
                  <div className="slider-group">
                    <div className="slider-header">
                      <span>Model Size</span>
                      <span className="slider-value">{params}B</span>
                    </div>
                    <div className="slider-container">
                      <input
                        type="range"
                        min={1}
                        max={1000}
                        value={params}
                        onChange={(e) => setParams(Number(e.target.value))}
                      />
                      <div className="slider-marks">
                        <span>1B</span>
                        <span>250B</span>
                        <span>500B</span>
                        <span>750B</span>
                        <span>1000B</span>
                      </div>
                    </div>
                    <NumberInput 
                      min={1}
                      max={1000}
                      value={params}
                      onChange={(e) => handleInputChange(e, setParams)}
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Model Quantization</label>
                  <select
                    value={modelQuant}
                    onChange={(e) => setModelQuant(e.target.value as ModelQuantization)}
                  >
                    <option value="F32">F32 - Full Precision (32-bit)</option>
                    <option value="F16">F16 - Half Precision (16-bit)</option>
                    <option value="Q8">Q8 - 8-bit Quantization</option>
                    <option value="Q6">Q6 - 6-bit Quantization</option>
                    <option value="Q5">Q5 - 5-bit Quantization</option>
                    <option value="Q4">Q4 - 4-bit Quantization</option>
                    <option value="Q3">Q3 - 3-bit Quantization</option>
                    <option value="Q2">Q2 - 2-bit Quantization</option>
                    <option value="GPTQ">GPTQ - ~4-bit optimized</option>
                    <option value="AWQ">AWQ - ~3.5-bit optimized</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Context Length (Tokens)</label>
                  <div className="slider-group">
                    <div className="slider-header">
                      <span>Context Window</span>
                      <span className="slider-value">{contextLength}</span>
                    </div>
                    <input
                      type="range"
                      min={128}
                      max={32768}
                      step={128}
                      value={contextLength}
                      onChange={(e) => setContextLength(Number(e.target.value))}
                    />
                    <NumberInput 
                      min={128}
                      max={32768}
                      step={128}
                      value={contextLength}
                      onChange={(e) => handleInputChange(e, setContextLength)}
                    />
                  </div>
                </div>

                <div className="form-group">
                  <div className="checkbox-row">
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={useKvCache}
                        onChange={() => setUseKvCache(!useKvCache)}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                    <label>Enable KV Cache</label>
                  </div>

                  <div className={`kvCacheAnimate ${useKvCache ? "open" : "closed"}`}>
                    <label className="form-label">KV Cache Quantization</label>
                    <select
                      value={kvCacheQuant}
                      onChange={(e) => setKvCacheQuant(e.target.value as KvCacheQuantization)}
                    >
                      <option value="F32">F32 - Full Precision</option>
                      <option value="F16">F16 - Half Precision</option>
                      <option value="Q8">Q8 - 8-bit Quantization</option>
                      <option value="Q5">Q5 - 5-bit Quantization</option>
                      <option value="Q4">Q4 - 4-bit Quantization</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="section">
                <h2 className="section-title">
                  <HardwareIcon /> System Configuration
                </h2>

                <div className="form-group">
                  <label className="form-label">System Type</label>
                  <select
                    value={memoryMode}
                    onChange={(e) => setMemoryMode(e.target.value as MemoryMode)}
                  >
                    <option value="DISCRETE_GPU">Discrete GPU</option>
                    <option value="UNIFIED_MEMORY">
                      Unified memory (e.g., Apple Silicon, AMD Ryzen™ Al Max+ 395)
                    </option>
                  </select>
                </div>

                {memoryMode === 'DISCRETE_GPU' && (
                  <div className="form-group">
                    <label className="form-label">GPU VRAM (GB)</label>
                    <select
                      value={gpuVram}
                      onChange={(e) => setGpuVram(Number(e.target.value))}
                    >
                      <option value={8}>8 GB (e.g., RTX 3070 Mobile)</option>
                      <option value={12}>12 GB (e.g., RTX 3060)</option>
                      <option value={16}>16 GB (e.g., RTX 4060 Ti)</option>
                      <option value={24}>24 GB (e.g., RTX 4090, RTX A5000)</option>
                      <option value={32}>32 GB (e.g., H100 PCIe)</option>
                      <option value={40}>40 GB (e.g., A100)</option>
                      <option value={48}>48 GB (e.g., RTX A6000)</option>
                      <option value={80}>80 GB (e.g., H100 SXM)</option>
                    </select>
                  </div>
                )}

                <div className="form-group">
                  <label className="form-label">System Memory (GB)</label>
                  <div className="slider-group">
                    <div className="slider-header">
                      <span>RAM</span>
                      <span className="slider-value">{systemMemory} GB</span>
                    </div>
                    <input
                      type="range"
                      min={8}
                      max={512}
                      step={8}
                      value={systemMemory}
                      onChange={(e) => setSystemMemory(Number(e.target.value))}
                    />
                    <NumberInput 
                      min={8}
                      max={512}
                      step={8}
                      value={systemMemory}
                      onChange={(e) => handleInputChange(e, setSystemMemory)}
                    />
                  </div>
                </div>

                <div className="form-group" style={{ textAlign: 'center', marginTop: '1rem' }}>
                  <button 
                    className="cta-button" 
                    onClick={() => setShowCalculator(false)}
                    style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}
                  >
                    Back to Welcome Screen
                  </button>
                </div>
              </div>
            </div>

            {/* Right Panel: Results */}
            <div className="card results-panel">
              <h2 className="section-title">
                <ResultsIcon /> Hardware Requirements
              </h2>

              <div className="result-row">
                <span className="result-label">VRAM Needed</span>
                <span className="result-highlight">{recommendation.vramNeeded} GB</span>
              </div>

              <div className="result-row">
                <span className="result-label">On-Disk Size</span>
                <span className="result-value">{onDiskSize.toFixed(2)} GB</span>
              </div>

              <div className="result-row">
                <span className="result-label">GPU Configuration</span>
                <span className="result-value">{recommendation.gpuType}</span>
              </div>

              <div className="result-row">
                <span className="result-label">GPUs Required</span>
                <span className="result-value">
                  {recommendation.gpusRequired === 0 
                    ? "Insufficient GPU memory" 
                    : recommendation.gpusRequired === 1 
                      ? "1 (Single GPU)" 
                      : `${recommendation.gpusRequired}`}
                </span>
              </div>

              <div className="result-row">
                <span className="result-label">System RAM</span>
                <span className="result-value">{recommendation.systemRamNeeded.toFixed(1)} GB</span>
              </div>

              {memoryMode === 'UNIFIED_MEMORY' && (
                <div className={`result-status ${recommendation.fitsUnified ? 'success' : 'warning'}`}>
                  {recommendation.fitsUnified ? (
                    <>
                      <SuccessIcon /> Fits in unified memory!
                    </>
                  ) : (
                    <>
                      <WarningIcon /> Exceeds unified memory. Increase system RAM or reduce model size.
                    </>
                  )}
                </div>
              )}
              
              {/* GPU load visual */}
              {recommendation.gpusRequired > 0 && recommendation.gpusRequired <= 4 && (
                <div className="utilization-container">
                  <label className="form-label">GPU Utilization</label>
                  <div className="utilization-bar">
                    <div 
                      className={`utilization-fill ${utilizationClass}`}
                      style={{ width: `${Math.min(100, utilizationPercentage)}%` }}
                    ></div>
                  </div>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    marginTop: '0.4rem',
                    fontSize: '0.75rem', 
                    color: 'var(--text-muted)' 
                  }}>
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                  </div>
                </div>
              )}
              
              {/* Complexity indicator */}
              <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                  Complexity Level
                </div>
                <div style={{ 
                  padding: '0.4rem 0.8rem',
                  background: `linear-gradient(90deg, var(--${utilizationClass === 'low' ? 'success' : utilizationClass === 'medium' ? 'primary' : utilizationClass === 'high' ? 'warning' : 'danger'}), transparent)`,
                  borderRadius: 'var(--radius)',
                  fontWeight: '600',
                  color: `var(--${utilizationClass === 'low' ? 'success' : utilizationClass === 'medium' ? 'primary' : utilizationClass === 'high' ? 'warning' : 'danger'})`,
                  fontSize: '0.85rem'
                }}>
                  {backgroundGradient === 'gradient-tiny' && 'Tiny (1-15B)'}
                  {backgroundGradient === 'gradient-small' && 'Small (16-32B)'}
                  {backgroundGradient === 'gradient-medium' && 'Medium (33-70B)'}
                  {backgroundGradient === 'gradient-large' && 'Large (71-200B)'}
                  {backgroundGradient === 'gradient-xlarge' && 'XLarge (201-500B)'}
                  {backgroundGradient === 'gradient-massive' && 'Massive (500B+)'}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;