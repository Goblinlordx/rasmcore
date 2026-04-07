/**
 * Shared types for the ML provider package.
 * Re-exports from the SDK's ml-provider for consistency.
 */
export type {
  MlCapabilityInfo,
  MlInputSpec,
  MlOutputKind,
  MlTileMode,
  MlPaddingMode,
  TensorDesc,
  TensorDtype,
  MlParamDescriptor,
  MlOp,
  MlProvider,
} from '../../sdk/v2/lib/ml-provider.js';

export { MlError } from '../../sdk/v2/lib/ml-provider.js';
