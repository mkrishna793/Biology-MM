/**
 * Pre-allocated GCMC (Grand Canonical Monte Carlo) slot pool with zero-radius tombstoning.
 *
 * Avoids topology reallocation during particle insertion/deletion by keeping a fixed
 * capacity buffer and setting `radii[slot] = 0` for deleted slots.
 */

const DEFAULT_VDW_RADII: Record<number, number> = {
  1: 1.2, // H
  6: 1.7, // C
  7: 1.55, // N
  8: 1.52, // O
  9: 1.47, // F
  11: 2.27, // Na
  14: 2.1, // Si
  15: 1.8, // P
  16: 1.8, // S
  17: 1.75, // Cl
  18: 1.88, // Ar
  19: 2.75, // K
  29: 1.4, // Cu
  79: 1.66, // Au
};

export function defaultRadiusForAtomicNumber(atomicNumber: number): number {
  return DEFAULT_VDW_RADII[atomicNumber] ?? 1.65;
}

export interface GcmcDeltaOptions {
  readonly atomicNumbers?: ArrayLike<number>;
  readonly radii?: ArrayLike<number>;
  readonly coords?: Float32Array;
  readonly insertedCoords?: ArrayLike<number>;
}

export interface GcmcDeltaResult {
  readonly insertedCount: number;
  readonly deletedCount: number;
  readonly maxActiveSlot: number;
}

export class GcmcSlotPool {
  readonly capacity: number;
  readonly activeMask: Uint8Array;
  readonly atomicNumbers: Uint8Array;
  readonly radii: Float32Array;

  private _activeCount = 0;
  private _highWaterMark = 0;

  constructor(capacity: number) {
    if (!Number.isInteger(capacity) || capacity < 0) {
      throw new RangeError(`GcmcSlotPool capacity must be a non-negative integer, got ${capacity}`);
    }
    this.capacity = capacity;
    this.activeMask = new Uint8Array(capacity);
    this.atomicNumbers = new Uint8Array(capacity);
    this.radii = new Float32Array(capacity);
  }

  get activeCount(): number {
    return this._activeCount;
  }

  /** Highest slot index + 1 that has ever been activated since last init/reset. */
  get highWaterMark(): number {
    return this._highWaterMark;
  }

  isActive(slot: number): boolean {
    return slot >= 0 && slot < this.capacity && this.activeMask[slot] === 1;
  }

  reset(): void {
    this.activeMask.fill(0);
    this.atomicNumbers.fill(0);
    this.radii.fill(0);
    this._activeCount = 0;
    this._highWaterMark = 0;
  }

  initFromTopology(nActive: number, atomicNumbers: ArrayLike<number>, radii: ArrayLike<number>): void {
    this.reset();
    const count = Math.min(nActive, this.capacity, atomicNumbers.length, radii.length);
    for (let i = 0; i < count; i++) {
      this.activeMask[i] = 1;
      const z = atomicNumbers[i];
      this.atomicNumbers[i] = z;
      const r = radii[i];
      this.radii[i] = r > 0 ? r : defaultRadiusForAtomicNumber(z);
    }
    this._activeCount = count;
    this._highWaterMark = count;
  }

  /**
   * Apply GCMC insertions and deletions in-place without reallocating buffers.
   * Deleted slots are tombstoned with `radii[slot] = 0` and `activeMask[slot] = 0`.
   */
  applyDelta(
    inserted: ArrayLike<number>,
    deleted: ArrayLike<number>,
    options?: GcmcDeltaOptions,
  ): GcmcDeltaResult {
    let deletedCount = 0;
    let insertedCount = 0;

    for (let i = 0; i < deleted.length; i++) {
      const slot = deleted[i];
      if (slot < 0 || slot >= this.capacity) continue;
      if (this.activeMask[slot] === 1) {
        this._activeCount--;
        deletedCount++;
      }
      this.activeMask[slot] = 0;
      this.atomicNumbers[slot] = 0;
      this.radii[slot] = 0;
    }

    const atomicNumbers = options?.atomicNumbers;
    const radii = options?.radii;
    const coords = options?.coords;
    const insertedCoords = options?.insertedCoords;

    for (let i = 0; i < inserted.length; i++) {
      const slot = inserted[i];
      if (slot < 0 || slot >= this.capacity) continue;
      if (this.activeMask[slot] === 0) {
        this._activeCount++;
        insertedCount++;
      }
      this.activeMask[slot] = 1;
      const z = atomicNumbers ? atomicNumbers[i] : this.atomicNumbers[slot] || 6;
      this.atomicNumbers[slot] = z;
      const r = radii ? radii[i] : defaultRadiusForAtomicNumber(z);
      this.radii[slot] = r > 0 ? r : defaultRadiusForAtomicNumber(z);

      if (coords && insertedCoords) {
        const s3 = i * 3;
        const d3 = slot * 3;
        if (d3 + 2 < coords.length && s3 + 2 < insertedCoords.length) {
          coords[d3] = insertedCoords[s3];
          coords[d3 + 1] = insertedCoords[s3 + 1];
          coords[d3 + 2] = insertedCoords[s3 + 2];
        }
      }

      if (slot + 1 > this._highWaterMark) {
        this._highWaterMark = slot + 1;
      }
    }

    return {
      insertedCount,
      deletedCount,
      maxActiveSlot: this._highWaterMark,
    };
  }
}
