export const STUDIO_PHYSICS_CHANNELS = [
  'force-magnitude',
  'kinetic-temp',
  'charge-potential',
  'bond-order',
  'atomic-stress',
  'ternary-state',
] as const;

export type PhysicsChannelName = (typeof STUDIO_PHYSICS_CHANNELS)[number];

export type SimulationMode =
  | 'metal_verlet'
  | 'cnt_tersoff'
  | 'custom_spline'
  | 'nanopore_gcmc_nemd';

export type PaletteName =
  | 'plasma'
  | 'thermal-div'
  | 'electrostatic-bwr'
  | 'bond-viridis'
  | 'stress-inferno'
  | 'ternary-triad';

export interface HardwareProfile {
  has_gpu: boolean;
  backend: string;
  device_name: string;
  unified_memory?: boolean;
  max_threads_per_tg?: number;
  simd_width?: number;
  vram_bytes?: number;
}

export interface StudioCapabilities {
  studio_version: string;
  hardware: HardwareProfile;
  channels: readonly PhysicsChannelName[];
  palettes: readonly PaletteName[];
  presets_count: number;
  max_interactive_atoms: number;
  supports_mvsx: boolean;
}

export interface SimulationMetrics {
  step: number;
  time_ps: number;
  n_atoms: number;
  capacity: number;
  potential_energy: number;
  kinetic_energy: number;
  total_energy: number;
  temperature_K: number;
  max_force: number;
  ms_per_step: number;
  ns_per_day_2fs: number;
  box_lengths_nm: [number, number, number];
  volume_nm3: number;
  stress_zz: number;
  mode: string;
  backend: string;
  history_steps: number[];
  gcmc_inserted_slots?: number[];
  gcmc_deleted_slots?: number[];
  session_id?: string;
  dropped_frames?: number;
}

export interface AtomInspection {
  index: number;
  active: boolean;
  element: string;
  atomic_number: number;
  position_nm: [number, number, number];
  position_ang: [number, number, number];
  velocity_nm_ps: [number, number, number];
  force: [number, number, number];
  force_magnitude: number;
  kinetic_energy: number;
  local_temp_K: number;
  charge: number;
  bond_order: number;
  atomic_stress: number;
  ternary_state: number;
}

export interface PresetInfo {
  id: string;
  title: string;
  category: string;
  description: string;
  n_atoms: number;
  capacity?: number;
  mode: SimulationMode;
  units: 'md' | 'reduced';
  default_channel: PhysicsChannelName;
  default_palette: PaletteName;
  temperature: number;
  efield_z?: number;
  gcmc_mu?: number;
  code_template: string;
  tags: string[];
}

export interface CustomPhysicsSpec {
  potential_expr: string;
  r_min: number;
  r_max: number;
  n_points?: number;
  temperature: number;
  efield_z: number;
  gcmc_mu: number;
  strain_rate: number;
  dt?: number;
  seed_preset_id?: string;
}

export interface SteerRequest {
  temperature?: number;
  efield_z?: number;
  gcmc_mu?: number;
  strain_rate?: number;
  active_channel?: PhysicsChannelName;
  palette?: PaletteName;
}

export interface CodeRunResult {
  ok: boolean;
  session_id?: string;
  stdout: string;
  stderr: string;
  error?: string;
  elapsed_ms: number;
  metrics?: SimulationMetrics;
}

export interface FrameStats {
  fps: number;
  frameId: number;
  simTimePs: number;
  nAtoms: number;
  scalarMin: number;
  scalarMax: number;
  activeChannel: PhysicsChannelName;
  activePalette: PaletteName;
  reversed: boolean;
  wsConnected: boolean;
}
