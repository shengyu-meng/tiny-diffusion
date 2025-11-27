// viz-frontend/src/components/ControlPanel.tsx

import React, { useState, useCallback } from 'react';
import type { GenerationParams } from '../services/api';

interface ControlPanelProps {
  onStartGeneration: (params: GenerationParams) => void;
  isConnected: boolean;
  isGenerating: boolean;
  onStopGeneration: () => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({ onStartGeneration, isConnected, isGenerating, onStopGeneration }) => {
  const [prompt, setPrompt] = useState<string>('');
  const [seqLen, setSeqLen] = useState<number>(256);
  const [numSteps, setNumSteps] = useState<number>(128);
  const [temperature, setTemperature] = useState<number>(1.0);
  const [method, setMethod] = useState<'confidence' | 'topk'>('confidence');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.95);
  const [k, setK] = useState<number>(25); // Corresponds to seq_len / 10 from backend, let's start with 25 for 256/10

  const handleStart = useCallback(() => {
    const params: GenerationParams = {
      prompt,
      seq_len: seqLen,
      num_steps: numSteps,
      temperature,
      method,
    };

    if (method === 'confidence') {
      params.confidence_threshold = confidenceThreshold;
    } else {
      params.k = k;
    }

    onStartGeneration(params);
  }, [prompt, seqLen, numSteps, temperature, method, confidenceThreshold, k, onStartGeneration]);

  return (
    <div className="Control-panel">
      <h3>Generation Controls</h3>
      <div>
        <label title="Initial text to guide the generation (context).">
          Prompt:
          <input type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)} disabled={isGenerating} />
        </label>
      </div>
      <div>
        <label title="The total length of the sequence to generate.">
          Sequence Length:
          <input type="number" value={seqLen} onChange={(e) => setSeqLen(Number(e.target.value))} disabled={isGenerating} />
        </label>
      </div>
      <div>
        <label title="Number of diffusion steps (iterations) to perform.">
          Number of Steps:
          <input type="number" value={numSteps} onChange={(e) => setNumSteps(Number(e.target.value))} disabled={isGenerating} />
        </label>
      </div>
      <div>
        <label title="Controls the randomness of token predictions. Higher values make output more diverse.">
          Temperature:
          <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} disabled={isGenerating} />
        </label>
      </div>
      <div>
        <label title="Method to select tokens for decoding at each step. 'Confidence' decodes tokens above a threshold, 'Top K' decodes the K most confident tokens.">
          Method:
          <select value={method} onChange={(e) => setMethod(e.target.value as 'confidence' | 'topk')} disabled={isGenerating}>
            <option value="confidence">Confidence</option>
            <option value="topk">Top K</option>
          </select>
        </label>
      </div>
      {method === 'confidence' ? (
        <div>
          <label title="Tokens with confidence scores above this threshold will be decoded.">
            Confidence Threshold:
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
              disabled={isGenerating}
            />
          </label>
        </div>
      ) : (
        <div>
          <label title="The K most confident tokens will be decoded at each step.">
            K (Top tokens to decode):
            <input type="number" value={k} onChange={(e) => setK(Number(e.target.value))} disabled={isGenerating} />
          </label>
        </div>
      )}
      <div style={{ marginTop: '20px' }}>
        <button onClick={handleStart} disabled={!isConnected || isGenerating} title={isConnected ? (isGenerating ? 'Generation in progress' : 'Start text generation') : 'Connect to WebSocket backend first'}>
          {isGenerating ? 'Generating...' : 'Start Generation'}
        </button>
        {isGenerating && (
          <button onClick={onStopGeneration} style={{ marginLeft: '10px' }} title="Stop the current text generation process.">
            Stop Generation
          </button>
        )}
      </div>
      {!isConnected && <p style={{ color: 'red' }}>Not connected to WebSocket backend.</p>}
    </div>
  );
};

export default ControlPanel;
