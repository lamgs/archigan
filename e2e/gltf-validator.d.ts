// Minimal typings for the parts of the Khronos glTF-Validator we use.
declare module "gltf-validator" {
  export interface ValidationMessage { code: string; message: string; severity: 0 | 1 | 2 | 3; pointer?: string }
  export interface ValidationReport { issues: { numErrors: number; numWarnings: number; numInfos: number; numHints: number; messages: ValidationMessage[] } }
  export function validateBytes(data: Uint8Array): Promise<ValidationReport>;
}
