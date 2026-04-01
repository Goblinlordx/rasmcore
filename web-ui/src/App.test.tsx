import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AppProvider } from './context/AppContext';
import App from './App';

// Mock Worker
vi.stubGlobal(
  'Worker',
  class {
    postMessage = vi.fn();
    terminate = vi.fn();
    onmessage: ((e: MessageEvent) => void) | null = null;
    addEventListener = vi.fn();
    removeEventListener = vi.fn();
  },
);

describe('App', () => {
  it('renders loading state initially', () => {
    render(
      <AppProvider>
        <App />
      </AppProvider>,
    );
    expect(screen.getByText(/Loading SDK/i)).toBeInTheDocument();
  });
});
