// viz-frontend/src/components/TextDisplay.tsx

import React from 'react';
import type { TokenData } from '../services/api';

interface TextDisplayProps {
  tokens: TokenData[];
  currentStep: number;
  maxSteps: number;
}

const TextDisplay: React.FC<TextDisplayProps> = ({ tokens, currentStep, maxSteps }) => {
  return (
    <div style={{
      fontFamily: 'monospace',
      whiteSpace: 'pre-wrap',
      backgroundColor: '#222',
      color: '#eee',
      padding: '10px',
      borderRadius: '8px',
      margin: '20px 0',
      minHeight: '100px',
      maxHeight: '300px',
      overflowY: 'auto'
    }}>
      <h4>Text Generation Progress: {currentStep}/{maxSteps} steps</h4>
      <p>
        {tokens.map((token, index) => (
          <span
            key={index}
            style={{
              color: token.is_masked ? 'red' :
                     (token.conf > 0.9 ? 'lightgreen' : 'yellow'),
              fontWeight: token.is_masked ? 'bold' : 'normal',
            }}
            title={`Confidence: ${token.conf.toFixed(3)}`}
          >
            {token.char === '\n' ? '↵\n' : token.char}
          </span>
        ))}
      </p>
    </div>
  );
};

export default TextDisplay;
