import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import type { Operation } from '../types';
import { loadManifest, type ManifestGroupMeta } from '../utils/manifest';

interface AppContextValue {
  operations: Operation[];
  groups: Record<string, ManifestGroupMeta>;
  writeFormats: string[];
  loading: boolean;
}

const AppContext = createContext<AppContextValue>({
  operations: [],
  groups: {},
  writeFormats: ['jpeg', 'png', 'webp'],
  loading: true,
});

export function AppProvider({ children }: { children: ReactNode }) {
  const [value, setValue] = useState<AppContextValue>({
    operations: [],
    groups: {},
    writeFormats: ['jpeg', 'png', 'webp'],
    loading: true,
  });

  useEffect(() => {
    loadManifest().then((manifest) => {
      setValue({
        operations: manifest.operations,
        groups: manifest.groups,
        writeFormats: manifest.writeFormats,
        loading: false,
      });
    });
  }, []);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppContext() {
  return useContext(AppContext);
}
