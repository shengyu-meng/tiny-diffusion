// viz-frontend/src/services/api.ts

import { useRef, useEffect, useState, useCallback } from 'react';

// --- Type Definitions ---
export interface TokenData {
  id: number;
  char: string;
  conf: number;
  pos: [number, number, number]; // 3D coordinates
  is_masked: boolean;
}

export interface StepUpdateMessage {
  type: 'step_update';
  data: {
    step: number;
    max_steps: number;
    tokens: TokenData[];
  };
}

export interface GenerationCompleteMessage {
  type: 'generation_complete';
  data: {
    final_text: string;
    tokens: TokenData[]; // Final state of tokens
  };
}

export type WebSocketMessage = StepUpdateMessage | GenerationCompleteMessage;

export interface GenerationParams {
  prompt: string;
  seq_len: number;
  num_steps: number;
  temperature: number;
  method: 'confidence' | 'topk';
  confidence_threshold?: number;
  k?: number;
}

// --- WebSocket Hook ---
interface WebSocketHookOptions {
  onMessage: (message: WebSocketMessage) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (event: Event) => void;
}

export const useWebSocket = (url: string, options: WebSocketHookOptions) => {
  const { onMessage, onOpen, onClose, onError } = options;
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Use refs to store the latest callbacks to avoid re-connecting when they change
  const onMessageRef = useRef(onMessage);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onMessageRef.current = onMessage;
    onOpenRef.current = onOpen;
    onCloseRef.current = onClose;
    onErrorRef.current = onError;
  }, [onMessage, onOpen, onClose, onError]);

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      if (onOpenRef.current) onOpenRef.current();
    };

    ws.current.onmessage = (event) => {
      const message: WebSocketMessage = JSON.parse(event.data);
      if (onMessageRef.current) onMessageRef.current(message);
    };

    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      if (onCloseRef.current) onCloseRef.current();
    };

    ws.current.onerror = (event) => {
      console.error('WebSocket error:', event);
      if (onErrorRef.current) onErrorRef.current(event);
    };

    return () => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.close();
      }
    };
  }, [url]);

  const sendMessage = useCallback((message: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not open. Message not sent:', message);
    }
  }, []); // No dependencies needed as ws is a ref

  const startGeneration = useCallback((params: GenerationParams) => {
    sendMessage({
      action: 'start_generation',
      ...params,
    });
  }, [sendMessage]);

  return { ws: ws.current, isConnected, sendMessage, startGeneration };
};
