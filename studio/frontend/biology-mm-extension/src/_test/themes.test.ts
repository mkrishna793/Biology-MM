import { describe, expect, it } from 'vitest';
import { Column } from '@molstar/core/data/db';
import { Color } from '@molstar/core/util/color';
import { ColorTheme } from '@molstar/graphics/theme/color';
import type { ThemeDataContext } from '@molstar/graphics/theme/theme';
import { Particle, type ParticleAttribute, type ParticleList } from '@molstar/model/model/particles/particle-list';
import {
  DEFAULT_FALLBACK_COLOR,
  fillRgbaFromScalars,
  getPaletteLutRgba,
  getPaletteStopsHex,
  PALETTE_NAMES,
  samplePaletteColor,
} from '../color-palettes.js';
import {
  AtomicStressThemeProvider,
  BIOLOGY_MM_SCALARS_KEY,
  BondOrderThemeProvider,
  ChargePotentialThemeProvider,
  ForceMagnitudeThemeProvider,
  KineticTempThemeProvider,
  PHYSICS_CHANNEL_TO_STRUCT_THEME,
  PHYSICS_CHANNEL_TO_THEME,
  PHYSICS_THEME_PROVIDERS,
  registerPhysicsColorThemes,
  STRUCTURE_PHYSICS_THEME_PROVIDERS,
  StructureForceMagnitudeThemeProvider,
  StructureTernaryStateThemeProvider,
  TernaryStateThemeProvider,
} from '../physics-color-themes.js';
import { PHYSICS_CHANNELS, type PhysicsChannelName } from '../protocol.js';

function makeParticleList(channel: PhysicsChannelName, values: Float32Array, min: number, max: number): ParticleList {
  const count = values.length;
  const attr: ParticleAttribute = {
    column: Column.ofFloatArray(values),
    label: channel,
    min,
    max,
  };
  const attributes = new Map<string, ParticleAttribute>([[channel, attr]]);

  return {
    count,
    targets: new Int32Array(count),
    targetInfo: new Map([[0, { name: 'system' }]]),
    coordinates: new Float32Array(count * 3),
    attributes,
    getParticleLabel: (i: number) => `Particle ${i + 1}`,
    sourceData: { kind: 'custom', name: 'test', data: null } as unknown as ParticleList['sourceData'],
  };
}

describe('color-palettes', () => {
  it('exposes all 6 palettes with 256-entry LUTs and valid CSS hex stops', () => {
    expect(PALETTE_NAMES).toEqual([
      'plasma',
      'thermal-div',
      'electrostatic-bwr',
      'bond-viridis',
      'stress-inferno',
      'ternary-triad',
    ]);

    for (const palette of PALETTE_NAMES) {
      const lut = getPaletteLutRgba(palette);
      expect(lut).toBeInstanceOf(Uint8Array);
      expect(lut.length).toBe(256 * 4);

      const stops = getPaletteStopsHex(palette);
      expect(stops.length).toBeGreaterThanOrEqual(3);
      for (const hex of stops) {
        expect(hex).toMatch(/^#[0-9a-f]{6}$/i);
      }

      const c0 = samplePaletteColor(palette, 0);
      const cMid = samplePaletteColor(palette, 0.5);
      const c1 = samplePaletteColor(palette, 1);
      expect(typeof c0).toBe('number');
      expect(typeof cMid).toBe('number');
      expect(typeof c1).toBe('number');
      expect(samplePaletteColor(palette, -0.5)).toBe(c0);
      expect(samplePaletteColor(palette, 1.5)).toBe(c1);
    }
  });

  it('calibrates ternary-triad strictly to -1=#dc2626, 0=#94a3b8, +1=#06b6d4', () => {
    expect(samplePaletteColor('ternary-triad', 0)).toBe(Color(0xdc2626));
    expect(samplePaletteColor('ternary-triad', 0.5)).toBe(Color(0x94a3b8));
    expect(samplePaletteColor('ternary-triad', 1)).toBe(Color(0x06b6d4));
    expect(getPaletteStopsHex('ternary-triad')).toEqual(['#dc2626', '#94a3b8', '#06b6d4']);

    const scalars = new Float32Array([-1, 0, 1]);
    const out = new Uint8Array(12);
    fillRgbaFromScalars(scalars, -1, 1, 'ternary-triad', out);

    // -1 -> #dc2626 (220, 38, 38)
    expect(Array.from(out.subarray(0, 4))).toEqual([0xdc, 0x26, 0x26, 255]);
    // 0 -> #94a3b8 (148, 163, 184)
    expect(Array.from(out.subarray(4, 8))).toEqual([0x94, 0xa3, 0xb8, 255]);
    // +1 -> #06b6d4 (6, 182, 212)
    expect(Array.from(out.subarray(8, 12))).toEqual([0x06, 0xb6, 0xd4, 255]);
  });

  it('guards NaN, Infinity, min === max, and activeMask in fillRgbaFromScalars', () => {
    expect(samplePaletteColor('plasma', Number.NaN)).toBe(DEFAULT_FALLBACK_COLOR);
    expect(samplePaletteColor('plasma', -Infinity)).toBe(samplePaletteColor('plasma', 0));
    expect(samplePaletteColor('plasma', Infinity)).toBe(samplePaletteColor('plasma', 1));

    // min === max -> maps finite values to midpoint (0.5)
    const flat = new Float32Array([5, 5, Number.NaN]);
    const outFlat = new Uint8Array(12);
    fillRgbaFromScalars(flat, 5, 5, 'thermal-div', outFlat);
    const mid = Color.toRgb(samplePaletteColor('thermal-div', 0.5));
    expect(Array.from(outFlat.subarray(0, 4))).toEqual([...mid, 255]);
    expect(Array.from(outFlat.subarray(4, 8))).toEqual([...mid, 255]);
    expect(Array.from(outFlat.subarray(8, 12))).toEqual([204, 204, 204, 255]);

    // activeMask hides inactive slots as [0, 0, 0, 0]
    const vals = new Float32Array([0, 0.5, 1]);
    const mask = new Uint8Array([1, 0, 1]);
    const outMasked = new Uint8Array(12).fill(99);
    fillRgbaFromScalars(vals, 0, 1, 'bond-viridis', outMasked, mask);
    expect(outMasked[3]).toBe(255);
    expect(Array.from(outMasked.subarray(4, 8))).toEqual([0, 0, 0, 0]);
    expect(outMasked[11]).toBe(255);
  });

  it('matches samplePaletteColor accurately across all 6 palettes in fillRgbaFromScalars', () => {
    const tValues = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
    const out = new Uint8Array(tValues.length * 4);

    for (const palette of PALETTE_NAMES) {
      fillRgbaFromScalars(tValues, 0, 1, palette, out);
      for (let i = 0; i < tValues.length; i++) {
        const expected = Color.toRgb(samplePaletteColor(palette, tValues[i]));
        expect(Array.from(out.subarray(i * 4, i * 4 + 4))).toEqual([...expected, 255]);
      }
    }
  });
});

describe('physics-color-themes', () => {
  it('exports all 6 providers and maps every PHYSICS_CHANNELS entry', () => {
    expect(PHYSICS_THEME_PROVIDERS).toHaveLength(6);
    expect(ForceMagnitudeThemeProvider.name).toBe('biology-mm-force-magnitude');
    expect(KineticTempThemeProvider.name).toBe('biology-mm-kinetic-temp');
    expect(ChargePotentialThemeProvider.name).toBe('biology-mm-charge-potential');
    expect(BondOrderThemeProvider.name).toBe('biology-mm-bond-order');
    expect(AtomicStressThemeProvider.name).toBe('biology-mm-atomic-stress');
    expect(TernaryStateThemeProvider.name).toBe('biology-mm-ternary-state');

    for (const ch of PHYSICS_CHANNELS) {
      const themeName = PHYSICS_CHANNEL_TO_THEME[ch];
      expect(PHYSICS_THEME_PROVIDERS.some((p) => p.name === themeName)).toBe(true);
    }
  });

  it('checks isApplicable, factory colors, legends, and fillParticleColors for all 6 themes', () => {
    const emptyCtx: ThemeDataContext = {};

    const cases: Array<{
      channel: PhysicsChannelName;
      provider: (typeof PHYSICS_THEME_PROVIDERS)[number];
      values: Float32Array;
      min: number;
      max: number;
      legendKind: 'scale-legend' | 'table-legend';
    }> = [
      {
        channel: 'force-magnitude',
        provider: ForceMagnitudeThemeProvider,
        values: new Float32Array([0, 5, 10]),
        min: 0,
        max: 10,
        legendKind: 'scale-legend',
      },
      {
        channel: 'kinetic-temp',
        provider: KineticTempThemeProvider,
        values: new Float32Array([100, 300, 500]),
        min: 100,
        max: 500,
        legendKind: 'scale-legend',
      },
      {
        channel: 'charge-potential',
        provider: ChargePotentialThemeProvider,
        values: new Float32Array([-1, 0, 1]),
        min: -1,
        max: 1,
        legendKind: 'scale-legend',
      },
      {
        channel: 'bond-order',
        provider: BondOrderThemeProvider,
        values: new Float32Array([0.5, 1.0, 2.0]),
        min: 0.5,
        max: 2.0,
        legendKind: 'scale-legend',
      },
      {
        channel: 'atomic-stress',
        provider: AtomicStressThemeProvider,
        values: new Float32Array([0, 50, 100]),
        min: 0,
        max: 100,
        legendKind: 'scale-legend',
      },
      {
        channel: 'ternary-state',
        provider: TernaryStateThemeProvider,
        values: new Float32Array([-1, 0, 1]),
        min: -1,
        max: 1,
        legendKind: 'table-legend',
      },
    ];

    for (const c of cases) {
      expect(c.provider.isApplicable(emptyCtx)).toBe(false);

      const particles = makeParticleList(c.channel, c.values, c.min, c.max);
      const ctx: ThemeDataContext = { particles };
      expect(c.provider.isApplicable(ctx)).toBe(true);

      const theme = c.provider.factory(ctx, c.provider.defaultValues);
      expect(theme.granularity).toBe('instance');
      expect(theme.legend?.kind).toBe(c.legendKind);
      expect(theme.fillParticleColors).toBeDefined();

      // Verify fillParticleColors at non-zero byte offset matches per-location color()
      const offset = 8;
      const out = new Uint8Array(offset + particles.count * 4);
      theme.fillParticleColors!(particles, out, offset);

      for (let i = 0; i < particles.count; i++) {
        const locColor = theme.color(Particle.Location(particles, i), false);
        const [r, g, b] = Color.toRgb(locColor);
        const off = offset + i * 4;
        expect([out[off], out[off + 1], out[off + 2], out[off + 3]]).toEqual([r, g, b, 255]);
      }

      if (c.channel === 'ternary-state') {
        expect(theme.color(Particle.Location(particles, 0), false)).toBe(Color(0xdc2626));
        expect(theme.color(Particle.Location(particles, 1), false)).toBe(Color(0x94a3b8));
        expect(theme.color(Particle.Location(particles, 2), false)).toBe(Color(0x06b6d4));
      }
    }
  });

  it('registers idempotently and unregisters cleanly', () => {
    const registry = ColorTheme.createRegistry();
    const mockPlugin = {
      representation: {
        particles: {
          themes: {
            colorThemeRegistry: registry,
          },
        },
      },
    };

    const unregister1 = registerPhysicsColorThemes(mockPlugin);
    const unregister2 = registerPhysicsColorThemes(mockPlugin);

    for (const provider of PHYSICS_THEME_PROVIDERS) {
      expect(registry.has(provider)).toBe(true);
      expect(registry.has(provider.name)).toBe(true);
    }
    expect(registry.list.length).toBe(6);

    // Calling unregister1 removes all 6 even after a second idempotent register call
    unregister1();
    expect(registry.list.length).toBe(0);

    // Repeated unregister calls are safe no-ops
    unregister1();
    unregister2();
    expect(registry.list.length).toBe(0);
  });

  it('registers structure physics themes and evaluates dynamic scalars', () => {
    const particleRegistry = ColorTheme.createRegistry();
    const structureRegistry = ColorTheme.createRegistry();
    const mockPlugin = {
      representation: {
        particles: { themes: { colorThemeRegistry: particleRegistry } },
        structure: { themes: { colorThemeRegistry: structureRegistry } },
      },
    };

    const unregister = registerPhysicsColorThemes(mockPlugin);
    expect(particleRegistry.list.length).toBe(6);
    expect(structureRegistry.list.length).toBe(6);

    for (const provider of STRUCTURE_PHYSICS_THEME_PROVIDERS) {
      expect(structureRegistry.has(provider)).toBe(true);
      expect(structureRegistry.has(provider.name)).toBe(true);
    }
    expect(PHYSICS_CHANNEL_TO_STRUCT_THEME['force-magnitude']).toBe('biology-mm-struct-force-magnitude');
    expect(StructureForceMagnitudeThemeProvider.isApplicable({})).toBe(false);

    const fakeModel: any = {
      _dynamicPropertyData: {
        [BIOLOGY_MM_SCALARS_KEY]: new Map([
          ['ternary-state', { values: new Float32Array([-1, 0, 1]), min: -1, max: 1 }],
        ]),
      },
      _staticPropertyData: {},
      atomicHierarchy: {
        atomSourceIndex: Column.ofIntArray(new Int32Array([0, 1, 2])),
      },
      atomicConformation: {
        B_iso_or_equiv: Column.Undefined(3, Column.Schema.float),
      },
    };
    const fakeStructure: any = { models: [fakeModel] };
    expect(StructureTernaryStateThemeProvider.isApplicable({ structure: fakeStructure })).toBe(true);
    const theme = StructureTernaryStateThemeProvider.factory(
      { structure: fakeStructure },
      StructureTernaryStateThemeProvider.defaultValues,
    );
    expect(theme.legend?.kind).toBe('table-legend');
    expect(theme.granularity).toBe('group');

    unregister();
    expect(particleRegistry.list.length).toBe(0);
    expect(structureRegistry.list.length).toBe(0);
  });
});
