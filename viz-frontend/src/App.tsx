// viz-frontend/src/App.tsx

import React, { useState, useCallback, useEffect } from 'react'; // Import useEffect
import './App.css';
import { useWebSocket } from './services/api';
import type { WebSocketMessage, TokenData, GenerationParams } from './services/api';
import ControlPanel from './components/ControlPanel';
import Vis3D from './components/Vis3D';
import TextDisplay from './components/TextDisplay';

const WS_URL = 'ws://localhost:50000/ws/visualize';

function App() {
  const [tokens, setTokens] = useState<TokenData[]>([]);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [maxSteps, setMaxSteps] = useState<number>(0);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    // Initialize theme from localStorage or default to 'dark'
    const storedTheme = localStorage.getItem('theme');
    return storedTheme === 'light' ? 'light' : 'dark';
  });

  // Effect to apply theme class to documentElement
  useEffect(() => {
    document.documentElement.className = theme;
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
  }, []);

  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    if (message.type === 'step_update') {
      setTokens(message.data.tokens);
      setCurrentStep(message.data.step);
      setMaxSteps(message.data.max_steps);
      setIsGenerating(true); // Still generating
    } else if (message.type === 'generation_complete') {
      setTokens(message.data.tokens);
      setIsGenerating(false); // Generation finished
      console.log('Generation complete:', message.data.final_text);
    }
  }, []);

  const { ws, isConnected, startGeneration } = useWebSocket(WS_URL, {
    onMessage: handleWebSocketMessage,
    onOpen: () => console.log('App: WebSocket connected'),
    onClose: () => {
      console.log('App: WebSocket disconnected');
      setIsGenerating(false); // Ensure generation state is reset if connection drops
    },
    onError: (event) => console.error('App: WebSocket error:', event),
  });


  const handleStartGeneration = useCallback((params: GenerationParams) => {
    setTokens([]); // Clear previous tokens
    setCurrentStep(0);
    setMaxSteps(0);
    setIsGenerating(true);
    startGeneration(params);
  }, [startGeneration]);

  const handleStopGeneration = useCallback(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
      console.log("WebSocket connection closed to stop generation.");
      setIsGenerating(false);
    }
  }, [ws]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Diffusion Text Visualizer</h1>
        <div className="theme-toggle">
          <button onClick={toggleTheme}>
            Switch to {theme === 'light' ? 'Dark' : 'Light'} Mode
          </button>
        </div>
      </header>
      <div className="App-content">
        <ControlPanel
          onStartGeneration={handleStartGeneration}
          isConnected={isConnected}
          isGenerating={isGenerating}
          onStopGeneration={handleStopGeneration}
        />
        <div className="Visualization-area">
          <TextDisplay tokens={tokens} currentStep={currentStep} maxSteps={maxSteps} />
          <div style={{ width: '100%', height: '500px', border: '1px solid var(--panel-border)', marginTop: '20px' }}> {/* Use CSS var */}
            <Vis3D tokens={tokens} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
