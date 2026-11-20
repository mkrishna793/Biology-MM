/**
 * Mol* v6 particle ColorTheme providers for Biology-MM physics channels.
 */

import type { Column } from '@molstar/core/data/db';
import { Color } from '@molstar/core/util/color';
import { ScaleLegend, TableLegend } from '@molstar/core/util/legend';
import { ParamDefinition as PD } from '@molstar/core/util/param-definition';
import type { ColorTheme, LocationColor } from '@molstar/graphics/theme/color';
import { ColorThemeCategory } from '@molstar/graphics/theme/color/categories';
import { fillByLocation, getPagedPainting, setParticleRgba } from '@molstar/graphics/theme/color/particle-pages';
import type { ThemeDataContext } from '@molstar/graphics/theme/theme';
import type { Location } from '@molstar/model/model/location';
import { Particle, type ParticleList } from '@molstar/model/model/particles/particle-list';
import { Bond, StructureElement, Unit, type ElementIndex } from '@molstar/model/model/structure';
import {
  DEFAULT_FALLBACK_COLOR,
  fillRgbaFromScalars,
  getPaletteStopsColor,
  PALETTE_NAMES,
  type PaletteName,
  samplePaletteColor,
} from './color-palettes.js';
import type { PhysicsChannelName } from './protocol.js';

export type PhysicsThemeName =
  | 'biology-mm-force-magnitude'
  | 'biology-mm-kinetic-temp'
  | 'biology-mm-charge-potential'
  | 'biology-mm-bond-order'
  | 'biology-mm-atomic-stress'
  | 'biology-mm-ternary-state'
  | 'biology-mm-multiscale-zone';

const PALETTE_LABELS: Record<PaletteName, string> = {
  plasma: 'Plasma',
  'thermal-div': 'Thermal Diverging',
  'electrostatic-bwr': 'Electrostatic BWR',
  'bond-viridis': 'Bond Viridis',
  'stress-inferno': 'Stress Inferno',
  'ternary-triad': 'Ternary Triad',
};

const PALETTE_OPTIONS: [PaletteName, string][] = PALETTE_NAMES.map((name) => [name, PALETTE_LABELS[name]]);

export function createPhysicsColorThemeParams(defaultPalette: PaletteName, defaultDomain: [number, number] = [0, 1]) {
  return {
    palette: PD.Select<PaletteName>(defaultPalette, PALETTE_OPTIONS),
    domain: PD.MappedStatic('auto', {
      auto: PD.Group({}),
      custom: PD.Interval(defaultDomain, { step: 0.001 }),
    }),
  };
}

export type PhysicsColorThemeParams = ReturnType<typeof createPhysicsColorThemeParams>;

function getParticleList(ctx: ThemeDataContext): ParticleList | undefined {
  return ctx.particles;
}

function getAttributes(particles: ParticleList | undefined) {
  return particles?.attributes ?? particles?.pages?.pages[0]?.list.attributes;
}

function formatDomainLabel(value: number): string {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs === 0) return '0';
  if (abs >= 1e4 || abs < 1e-3) return value.toExponential(2);
  return Number(value.toPrecision(4)).toString();
}

function sampleScalarValue(value: number, min: number, max: number, palette: PaletteName): Color {
  if (Number.isNaN(value)) return DEFAULT_FALLBACK_COLOR;
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    if (value === -Infinity) return samplePaletteColor(palette, 0);
    if (value === Infinity) return samplePaletteColor(palette, 1);
    return samplePaletteColor(palette, 0.5);
  }
  const t = (value - min) / (max - min);
  return samplePaletteColor(palette, t);
}

function pickFromColumn(
  column: Column<any> | undefined,
  index: number,
  min: number,
  max: number,
  palette: PaletteName,
): Color {
  if (!column || index < 0 || index >= column.rowCount) return DEFAULT_FALLBACK_COLOR;
  const val = Number(column.value(index));
  return sampleScalarValue(val, min, max, palette);
}

function fillColumnRgba(
  column: Column<any> | undefined,
  min: number,
  max: number,
  palette: PaletteName,
  count: number,
  out: Uint8Array,
  offset: number,
): void {
  if (!column) {
    for (let i = 0; i < count; ++i) setParticleRgba(out, offset + i * 4, DEFAULT_FALLBACK_COLOR);
    return;
  }

  const storage = column.storage?.();
  if (storage && storage.kind === 'array' && storage.array.length >= count) {
    const outView = out.subarray(offset, offset + count * 4);
    fillRgbaFromScalars(storage.array as ArrayLike<number>, min, max, palette, outView);
    return;
  }

  for (let i = 0; i < count; ++i) {
    setParticleRgba(out, offset + i * 4, pickFromColumn(column, i, min, max, palette));
  }
}

interface PhysicsThemeSpec<Id extends PhysicsThemeName> {
  readonly name: Id;
  readonly label: string;
  readonly description: string;
  readonly channel: PhysicsChannelName;
  readonly defaultPalette: PaletteName;
  readonly defaultDomain?: [number, number];
  readonly isTernary?: boolean;
  readonly isMultiscaleZone?: boolean;
}

function createPhysicsThemeProvider<Id extends PhysicsThemeName>(
  spec: PhysicsThemeSpec<Id>,
): ColorTheme.Provider<PhysicsColorThemeParams, Id> {
  const Params = createPhysicsColorThemeParams(spec.defaultPalette, spec.defaultDomain ?? [0, 1]);

  function factory(
    ctx: ThemeDataContext,
    props: PD.Values<PhysicsColorThemeParams>,
  ): ColorTheme<PhysicsColorThemeParams> {
    let color: LocationColor;
    let legend: ScaleLegend | TableLegend | undefined;
    let fillParticleColors: ColorTheme<PhysicsColorThemeParams>['fillParticleColors'];

    const particles = getParticleList(ctx);
    const attributes = getAttributes(particles);
    const attr = attributes?.get(spec.channel);
    const palette = props.palette;

    if (attr) {
      let domain: [number, number];
      if (props.domain.name === 'custom') {
        domain = props.domain.params as [number, number];
      } else if (spec.isTernary && attr.min >= -1 && attr.max <= 1) {
        domain = [-1, 1];
      } else if (spec.isMultiscaleZone) {
        domain = [0, 2];
      } else {
        domain = [attr.min, attr.max];
      }

      const [min, max] = domain;

      if (spec.isTernary) {
        legend = TableLegend([
          ['-1 (State -1)', samplePaletteColor(palette, 0)],
          ['0 (Neutral)', samplePaletteColor(palette, 0.5)],
          ['+1 (State +1)', samplePaletteColor(palette, 1)],
        ]);
      } else if (spec.isMultiscaleZone) {
        legend = TableLegend([
          ['0 (All-Atom)', samplePaletteColor(palette, 0)],
          ['1 (Coarse-Grained)', samplePaletteColor(palette, 0.5)],
          ['2 (Rigid Scaffold)', samplePaletteColor(palette, 1)],
        ]);
      } else {
        legend = ScaleLegend(formatDomainLabel(min), formatDomainLabel(max), [...getPaletteStopsColor(palette)]);
      }

      if (particles!.pages) {
        const painting = getPagedPainting((page) => {
          const column = page.attributes?.get(spec.channel)?.column;
          return {
            color: (i: number) => pickFromColumn(column, i, min, max, palette),
            fill: (out, offset) => fillColumnRgba(column, min, max, palette, page.count, out, offset),
          };
        }, DEFAULT_FALLBACK_COLOR);
        color = painting.color;
        fillParticleColors = painting.fill;
      } else {
        color = (location: Location): Color => {
          if (Particle.isLocation(location)) {
            return pickFromColumn(attr.column, location.index, min, max, palette);
          }
          return DEFAULT_FALLBACK_COLOR;
        };
        fillParticleColors = (list, out, offset) => {
          if (list !== particles) return fillByLocation(color, list, out, offset);
          fillColumnRgba(attr.column, min, max, palette, list.count, out, offset);
        };
      }
    } else {
      color = () => DEFAULT_FALLBACK_COLOR;
    }

    return {
      factory,
      granularity: 'instance',
      color,
      props,
      description: spec.description,
      legend,
      fillParticleColors,
    };
  }

  return {
    name: spec.name,
    label: spec.label,
    category: ColorThemeCategory.Particle,
    factory,
    getParams: () => PD.clone(Params),
    defaultValues: PD.getDefaultValues(Params),
    isApplicable: (ctx: ThemeDataContext) => {
      const attributes = getAttributes(getParticleList(ctx));
      return !!attributes && attributes.has(spec.channel);
    },
  };
}

export const ForceMagnitudeThemeProvider = createPhysicsThemeProvider({
  name: 'biology-mm-force-magnitude',
  label: 'Force Magnitude',
  description: 'Colors particles by per-atom force magnitude |F|.',
  channel: 'force-magnitude',
  defaultPalette: 'plasma',
});

export const KineticTempThemeProvider = createPhysicsThemeProvider({
  name: 'biology-mm-kinetic-temp',
  label: 'Kinetic Temperature',
  description: 'Colors particles by instantaneous per-atom kinetic temperature.',
  channel: 'kinetic-temp',
  defaultPalette: 'thermal-div',
});

export const ChargePotentialThemeProvider = createPhysicsThemeProvider({
  name: 'biology-mm-charge-potential',
  label: 'Charge / Potential',
  description: 'Colors particles by partial charge or local electrostatic potential.',
  channel: 'charge-potential',
  defaultPalette: 'electrostatic-bwr',
  defaultDomain: [-1, 1],
});

export const BondOrderThemeProvider = createPhysicsThemeProvider({
  name: 'biology-mm-bond-order',
  label: 'Bond Order',
  description: 'Colors particles by local Tersoff/reactive bond order.',
  channel: 'bond-order',
  defaultPalette: 'bond-viridis',
});

export const AtomicStressThemeProvider = createPhysicsThemeProvider({
  name: 'biology-mm-atomic-stress',
  label: 'Atomic Stress',
  description: 'Colors particles by per-atom virial stress magnitude.',
  channel: 'atomic-stress',
  defaultPalette: 'stress-inferno',
});

export const TernaryStateThemeProvider = createPhysicsThemeProvider({
  name: 'biology-mm-ternary-state',
  label: 'Ternary State',
  description: 'Colors particles by discrete ternary state (-1, 0, +1).',
  channel: 'ternary-state',
  defaultPalette: 'ternary-triad',
  defaultDomain: [-1, 1],
  isTernary: true,
});

export const MultiscaleZoneThemeProvider = createPhysicsThemeProvider({
  name: 'biology-mm-multiscale-zone',
  label: 'Multiscale Zone',
  description: 'Colors particles by multiscale resolution zone (0=All-Atom, 1=Coarse-Grained, 2=Rigid Scaffold).',
  channel: 'multiscale-zone',
  defaultPalette: 'ternary-triad',
  defaultDomain: [0, 2],
  isMultiscaleZone: true,
});

export const PHYSICS_THEME_PROVIDERS = [
  ForceMagnitudeThemeProvider,
  KineticTempThemeProvider,
  ChargePotentialThemeProvider,
  BondOrderThemeProvider,
  AtomicStressThemeProvider,
  TernaryStateThemeProvider,
  MultiscaleZoneThemeProvider,
] as const;

export const PHYSICS_CHANNEL_TO_THEME: Readonly<Record<PhysicsChannelName, PhysicsThemeName>> = {
  'force-magnitude': 'biology-mm-force-magnitude',
  'kinetic-temp': 'biology-mm-kinetic-temp',
  'charge-potential': 'biology-mm-charge-potential',
  'bond-order': 'biology-mm-bond-order',
  'atomic-stress': 'biology-mm-atomic-stress',
  'ternary-state': 'biology-mm-ternary-state',
  'multiscale-zone': 'biology-mm-multiscale-zone',
};

export const BIOLOGY_MM_SCALARS_KEY = '__biologyMmScalars';

export interface BiologyMmStructureScalarEntry {
  readonly values: Float32Array;
  readonly min: number;
  readonly max: number;
}

export type BiologyMmStructureScalarMap = ReadonlyMap<PhysicsChannelName, BiologyMmStructureScalarEntry>;

export type StructurePhysicsThemeName =
  | 'biology-mm-struct-force-magnitude'
  | 'biology-mm-struct-kinetic-temp'
  | 'biology-mm-struct-charge-potential'
  | 'biology-mm-struct-bond-order'
  | 'biology-mm-struct-atomic-stress'
  | 'biology-mm-struct-ternary-state'
  | 'biology-mm-struct-multiscale-zone';

interface StructurePhysicsThemeSpec<Id extends StructurePhysicsThemeName> {
  readonly name: Id;
  readonly label: string;
  readonly description: string;
  readonly channel: PhysicsChannelName;
  readonly defaultPalette: PaletteName;
  readonly defaultDomain?: [number, number];
  readonly isTernary?: boolean;
  readonly isMultiscaleZone?: boolean;
}

function getModelScalarMap(model: any): BiologyMmStructureScalarMap | undefined {
  return (model?._dynamicPropertyData?.[BIOLOGY_MM_SCALARS_KEY] ??
    model?._staticPropertyData?.[BIOLOGY_MM_SCALARS_KEY]) as BiologyMmStructureScalarMap | undefined;
}

function createStructurePhysicsThemeProvider<Id extends StructurePhysicsThemeName>(
  spec: StructurePhysicsThemeSpec<Id>,
): ColorTheme.Provider<PhysicsColorThemeParams, Id> {
  const Params = createPhysicsColorThemeParams(spec.defaultPalette, spec.defaultDomain ?? [0, 1]);

  function factory(
    ctx: ThemeDataContext,
    props: PD.Values<PhysicsColorThemeParams>,
  ): ColorTheme<PhysicsColorThemeParams> {
    const palette = props.palette;
    const structure = ctx.structure;

    let entry: BiologyMmStructureScalarEntry | undefined;
    if (structure) {
      for (const m of structure.models) {
        const candidate = getModelScalarMap(m)?.get(spec.channel);
        if (candidate) {
          entry = candidate;
          break;
        }
      }
    }

    let domain: [number, number];
    if (props.domain.name === 'custom') {
      domain = props.domain.params as [number, number];
    } else if (entry) {
      if (spec.isTernary && entry.min >= -1 && entry.max <= 1) {
        domain = [-1, 1];
      } else if (spec.isMultiscaleZone) {
        domain = [0, 2];
      } else {
        domain = [entry.min, entry.max];
      }
    } else if (structure && structure.models.some((m) => m.atomicConformation.B_iso_or_equiv.isDefined)) {
      let bMin = Infinity;
      let bMax = -Infinity;
      for (const m of structure.models) {
        const bCol = m.atomicConformation.B_iso_or_equiv;
        if (!bCol.isDefined) continue;
        for (let i = 0, il = bCol.rowCount; i < il; i++) {
          const v = bCol.value(i);
          if (Number.isFinite(v)) {
            if (v < bMin) bMin = v;
            if (v > bMax) bMax = v;
          }
        }
      }
      domain = Number.isFinite(bMin) && Number.isFinite(bMax) ? [bMin, bMax] : (spec.defaultDomain ?? [0, 1]);
    } else {
      domain = spec.defaultDomain ?? [0, 1];
    }

    const [min, max] = domain;

    const legend: ScaleLegend | TableLegend = spec.isTernary
      ? TableLegend([
          ['-1 (State -1)', samplePaletteColor(palette, 0)],
          ['0 (Neutral)', samplePaletteColor(palette, 0.5)],
          ['+1 (State +1)', samplePaletteColor(palette, 1)],
        ])
      : spec.isMultiscaleZone
        ? TableLegend([
            ['0 (All-Atom)', samplePaletteColor(palette, 0)],
            ['1 (Coarse-Grained)', samplePaletteColor(palette, 0.5)],
            ['2 (Rigid Scaffold)', samplePaletteColor(palette, 1)],
          ])
        : ScaleLegend(formatDomainLabel(min), formatDomainLabel(max), [...getPaletteStopsColor(palette)]);

    function colorAt(unit: Unit, element: ElementIndex): Color {
      if (!Unit.isAtomic(unit)) return DEFAULT_FALLBACK_COLOR;
      const modelEntry = getModelScalarMap(unit.model)?.get(spec.channel);
      if (modelEntry) {
        const srcIdx = unit.model.atomicHierarchy.atomSourceIndex.value(element);
        const idx = srcIdx >= 0 && srcIdx < modelEntry.values.length ? srcIdx : element;
        if (idx >= 0 && idx < modelEntry.values.length) {
          return sampleScalarValue(modelEntry.values[idx], min, max, palette);
        }
      }
      const bCol = unit.model.atomicConformation.B_iso_or_equiv;
      if (bCol.isDefined) {
        return sampleScalarValue(bCol.value(element), min, max, palette);
      }
      return DEFAULT_FALLBACK_COLOR;
    }

    const color: LocationColor = (location: Location): Color => {
      if (StructureElement.Location.is(location)) {
        return colorAt(location.unit, location.element);
      }
      if (Bond.isLocation(location)) {
        return colorAt(location.aUnit, location.aUnit.elements[location.aIndex]);
      }
      return DEFAULT_FALLBACK_COLOR;
    };

    return {
      factory,
      granularity: 'group',
      preferSmoothing: true,
      color,
      props,
      description: spec.description,
      legend,
    };
  }

  return {
    name: spec.name,
    label: spec.label,
    category: ColorThemeCategory.Atom,
    factory,
    getParams: () => PD.clone(Params),
    defaultValues: PD.getDefaultValues(Params),
    isApplicable: (ctx: ThemeDataContext) => !!ctx.structure,
  };
}

export const StructureForceMagnitudeThemeProvider = createStructurePhysicsThemeProvider({
  name: 'biology-mm-struct-force-magnitude',
  label: 'Biology-MM: Force Magnitude',
  description: 'Colors structure atoms by per-atom live Biology-MM force magnitude |F|.',
  channel: 'force-magnitude',
  defaultPalette: 'plasma',
});

export const StructureKineticTempThemeProvider = createStructurePhysicsThemeProvider({
  name: 'biology-mm-struct-kinetic-temp',
  label: 'Biology-MM: Kinetic Temperature',
  description: 'Colors structure atoms by instantaneous per-atom kinetic temperature.',
  channel: 'kinetic-temp',
  defaultPalette: 'thermal-div',
});

export const StructureChargePotentialThemeProvider = createStructurePhysicsThemeProvider({
  name: 'biology-mm-struct-charge-potential',
  label: 'Biology-MM: Charge / Potential',
  description: 'Colors structure atoms by partial charge or electrostatic potential.',
  channel: 'charge-potential',
  defaultPalette: 'electrostatic-bwr',
  defaultDomain: [-1, 1],
});

export const StructureBondOrderThemeProvider = createStructurePhysicsThemeProvider({
  name: 'biology-mm-struct-bond-order',
  label: 'Biology-MM: Bond Order',
  description: 'Colors structure atoms by local Tersoff/reactive bond order.',
  channel: 'bond-order',
  defaultPalette: 'bond-viridis',
});

export const StructureAtomicStressThemeProvider = createStructurePhysicsThemeProvider({
  name: 'biology-mm-struct-atomic-stress',
  label: 'Biology-MM: Atomic Stress',
  description: 'Colors structure atoms by per-atom virial stress magnitude.',
  channel: 'atomic-stress',
  defaultPalette: 'stress-inferno',
});

export const StructureTernaryStateThemeProvider = createStructurePhysicsThemeProvider({
  name: 'biology-mm-struct-ternary-state',
  label: 'Biology-MM: Ternary State',
  description: 'Colors structure atoms by discrete ternary state (-1, 0, +1).',
  channel: 'ternary-state',
  defaultPalette: 'ternary-triad',
  defaultDomain: [-1, 1],
  isTernary: true,
});

export const StructureMultiscaleZoneThemeProvider = createStructurePhysicsThemeProvider({
  name: 'biology-mm-struct-multiscale-zone',
  label: 'Biology-MM: Multiscale Zone',
  description: 'Colors structure atoms by multiscale resolution zone (0=All-Atom, 1=Coarse-Grained, 2=Rigid Scaffold).',
  channel: 'multiscale-zone',
  defaultPalette: 'ternary-triad',
  defaultDomain: [0, 2],
  isMultiscaleZone: true,
});

export const STRUCTURE_PHYSICS_THEME_PROVIDERS = [
  StructureForceMagnitudeThemeProvider,
  StructureKineticTempThemeProvider,
  StructureChargePotentialThemeProvider,
  StructureBondOrderThemeProvider,
  StructureAtomicStressThemeProvider,
  StructureTernaryStateThemeProvider,
  StructureMultiscaleZoneThemeProvider,
] as const;

export const PHYSICS_CHANNEL_TO_STRUCT_THEME: Readonly<Record<PhysicsChannelName, StructurePhysicsThemeName>> = {
  'force-magnitude': 'biology-mm-struct-force-magnitude',
  'kinetic-temp': 'biology-mm-struct-kinetic-temp',
  'charge-potential': 'biology-mm-struct-charge-potential',
  'bond-order': 'biology-mm-struct-bond-order',
  'atomic-stress': 'biology-mm-struct-atomic-stress',
  'ternary-state': 'biology-mm-struct-ternary-state',
  'multiscale-zone': 'biology-mm-struct-multiscale-zone',
};

export interface PhysicsThemePluginTarget {
  readonly representation: {
    readonly particles: {
      readonly themes: {
        readonly colorThemeRegistry: ColorTheme.Registry;
      };
    };
    readonly structure?: {
      readonly themes: {
        readonly colorThemeRegistry: ColorTheme.Registry;
      };
    };
  };
}

/**
 * Register all Biology-MM physics ColorTheme providers on a Mol* PluginContext's
 * particle and structure theme registries. Idempotent; returns cleanup function.
 */
export function registerPhysicsColorThemes(plugin: PhysicsThemePluginTarget): () => void {
  const particleRegistry = plugin.representation.particles.themes.colorThemeRegistry;
  const structureRegistry = plugin.representation.structure?.themes.colorThemeRegistry;
  const addedParticles: ColorTheme.Provider<any, any>[] = [];
  const addedStructure: ColorTheme.Provider<any, any>[] = [];

  for (const provider of PHYSICS_THEME_PROVIDERS) {
    if (!particleRegistry.has(provider)) {
      particleRegistry.add(provider);
      addedParticles.push(provider);
    }
  }

  if (structureRegistry) {
    for (const provider of STRUCTURE_PHYSICS_THEME_PROVIDERS) {
      if (!structureRegistry.has(provider)) {
        structureRegistry.add(provider);
        addedStructure.push(provider);
      }
    }
  }

  let unregistered = false;
  return () => {
    if (unregistered) return;
    unregistered = true;
    for (const provider of addedParticles) {
      if (particleRegistry.has(provider)) {
        particleRegistry.remove(provider);
      }
    }
    if (structureRegistry) {
      for (const provider of addedStructure) {
        if (structureRegistry.has(provider)) {
          structureRegistry.remove(provider);
        }
      }
    }
  };
}
