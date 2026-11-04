//! File-backed append-only store for frozen (inactive) state tiles.
//!
//! Records are fixed-size (one tile's four arrays back to back, little
//! endian), keyed by `(realization, world_row, world_col)` of the tile's
//! top-left cell, so keys survive domain growth. Records are immutable
//! and append-only within a session: re-freezing a thawed tile appends a
//! new record, so the `{key -> offset}` index a checkpoint captures stays
//! valid for incremental checkpointing.
//!
//! Interior mutability (`Mutex`) lets checkpoints hold an `Arc<TileStore>`
//! and read records later without borrowing the propagator.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::error::Result;
use crate::tiles::{Tile, TILE_CELLS};

/// (realization, world_row, world_col) of the tile's top-left cell.
pub type TileKey = (u32, i64, i64);
pub type TileIndex = std::collections::HashMap<TileKey, u64>;

const FLAGS_BYTES: usize = TILE_CELLS;
const ARRIVAL_BYTES: usize = TILE_CELLS * 4;
const ROS_BYTES: usize = TILE_CELLS * 4;
const FLI_BYTES: usize = TILE_CELLS * 4;
pub const RECORD_SIZE: usize = FLAGS_BYTES + ARRIVAL_BYTES + ROS_BYTES + FLI_BYTES;

/// Serialize a tile to a fixed-size little-endian record: the flags byte
/// array followed by the arrival, ros, and fli arrays (see [`RECORD_SIZE`]).
pub fn tile_to_record(tile: &Tile) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(RECORD_SIZE);
    buffer.extend_from_slice(&tile.flags);
    for value in &tile.arrival {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    for value in &tile.ros {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    for value in &tile.fli {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    buffer
}

/// Inverse of [`tile_to_record`]; panics if `buffer` is not `RECORD_SIZE`.
pub fn record_to_tile(buffer: &[u8]) -> Box<Tile> {
    assert_eq!(buffer.len(), RECORD_SIZE, "invalid tile record size");
    let mut tile = Tile::zeroed();
    tile.flags.copy_from_slice(&buffer[..FLAGS_BYTES]);
    let mut offset = FLAGS_BYTES;
    for value in &mut tile.arrival {
        *value = i32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap());
        offset += 4;
    }
    for value in &mut tile.ros {
        *value = f32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap());
        offset += 4;
    }
    for value in &mut tile.fli {
        *value = f32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap());
        offset += 4;
    }
    tile
}

struct StoreInner {
    file: File,
    /// keys currently frozen -> record offset; older offsets stay
    /// readable for checkpoints (records are never overwritten)
    index: TileIndex,
    end: u64,
}

pub struct TileStore {
    pub path: PathBuf,
    inner: Mutex<StoreInner>,
}

impl TileStore {
    /// Create (truncating any existing file) `frozen_tiles.bin` under
    /// `directory`, making the directory tree if needed.
    pub fn open(directory: impl AsRef<Path>) -> Result<TileStore> {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory)?;
        let path = directory.join("frozen_tiles.bin");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        Ok(TileStore {
            path,
            inner: Mutex::new(StoreInner {
                file,
                index: TileIndex::new(),
                end: 0,
            }),
        })
    }

    /// Number of tiles currently frozen.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().index.len()
    }

    /// `true` when no tiles are frozen.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `key` is currently frozen.
    pub fn contains(&self, key: &TileKey) -> bool {
        self.inner.lock().unwrap().index.contains_key(key)
    }

    /// All currently frozen keys.
    pub fn keys(&self) -> Vec<TileKey> {
        self.inner.lock().unwrap().index.keys().copied().collect()
    }

    /// Currently frozen keys belonging to one realization.
    pub fn keys_for_realization(&self, realization: u32) -> Vec<TileKey> {
        self.inner
            .lock()
            .unwrap()
            .index
            .keys()
            .filter(|key| key.0 == realization)
            .copied()
            .collect()
    }

    /// Append `payload` at the end of the file and point `key` at its
    /// offset. Never overwrites existing records, so offsets a checkpoint
    /// captured earlier stay valid.
    fn append_locked(inner: &mut StoreInner, key: TileKey, payload: &[u8]) -> Result<()> {
        debug_assert_eq!(payload.len(), RECORD_SIZE);
        let offset = inner.end;
        inner.file.seek(SeekFrom::Start(offset))?;
        inner.file.write_all(payload)?;
        inner.end = offset + RECORD_SIZE as u64;
        inner.index.insert(key, offset);
        Ok(())
    }

    /// Append one tile's state and mark it frozen.
    pub fn freeze(&self, key: TileKey, tile: &Tile) -> Result<()> {
        let payload = tile_to_record(tile);
        let mut inner = self.inner.lock().unwrap();
        Self::append_locked(&mut inner, key, &payload)
    }

    /// Append a raw record (imported from another store or a checkpoint).
    pub fn write_record(&self, key: TileKey, payload: &[u8]) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        Self::append_locked(&mut inner, key, payload)
    }

    /// Read one raw record at a known offset (for checkpoints holding
    /// stale-but-valid offsets).
    pub fn read_bytes(&self, offset: u64) -> Result<Vec<u8>> {
        let mut inner = self.inner.lock().unwrap();
        inner.file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; RECORD_SIZE];
        inner.file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// Read a frozen tile without unfreezing it.
    pub fn read(&self, key: &TileKey) -> Result<Box<Tile>> {
        let offset = {
            let inner = self.inner.lock().unwrap();
            *inner.index.get(key).expect("tile not frozen")
        };
        Ok(record_to_tile(&self.read_bytes(offset)?))
    }

    /// Read a frozen tile and unfreeze it. The record stays in the file
    /// so checkpoint indices that reference it remain valid.
    pub fn thaw(&self, key: &TileKey) -> Result<Box<Tile>> {
        let offset = {
            let mut inner = self.inner.lock().unwrap();
            inner.index.remove(key).expect("tile not frozen")
        };
        Ok(record_to_tile(&self.read_bytes(offset)?))
    }

    /// Copy of the current {key -> offset} index (for checkpoints).
    pub fn snapshot_index(&self) -> TileIndex {
        self.inner.lock().unwrap().index.clone()
    }

    /// Replace the frozen index (rollback to a checkpoint's view).
    pub fn restore_index(&self, index: TileIndex) {
        self.inner.lock().unwrap().index = index;
    }

    /// Drop all frozen tiles and reclaim the file. Invalidates any
    /// incremental checkpoint that still references this store.
    pub fn clear(&self) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.index.clear();
        inner.end = 0;
        inner.file.set_len(0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles::FLAG_FIRE;

    #[test]
    fn record_roundtrip() {
        let mut tile = Tile::zeroed();
        tile.flags[0] = FLAG_FIRE;
        tile.arrival[1] = 1234;
        tile.ros[2] = 3.5;
        tile.fli[3] = -7.25;
        let record = tile_to_record(&tile);
        assert_eq!(record.len(), RECORD_SIZE);
        let restored = record_to_tile(&record);
        assert_eq!(restored.flags[0], FLAG_FIRE);
        assert_eq!(restored.arrival[1], 1234);
        assert_eq!(restored.ros[2], 3.5);
        assert_eq!(restored.fli[3], -7.25);
    }

    #[test]
    fn freeze_thaw_and_append_only_offsets() {
        let dir = std::env::temp_dir().join(format!(
            "prop-core-store-test-{}",
            std::process::id()
        ));
        let store = TileStore::open(&dir).unwrap();
        let key = (0u32, 32i64, 64i64);

        let mut tile = Tile::zeroed();
        tile.arrival[5] = 42;
        store.freeze(key, &tile).unwrap();
        assert_eq!(store.len(), 1);
        let snapshot = store.snapshot_index();

        let thawed = store.thaw(&key).unwrap();
        assert_eq!(thawed.arrival[5], 42);
        assert_eq!(store.len(), 0);

        // re-freeze appends; the old snapshot offset stays readable
        tile.arrival[5] = 99;
        store.freeze(key, &tile).unwrap();
        let old_offset = snapshot[&key];
        let old = record_to_tile(&store.read_bytes(old_offset).unwrap());
        assert_eq!(old.arrival[5], 42);
        let new = store.read(&key).unwrap();
        assert_eq!(new.arrival[5], 99);

        std::fs::remove_dir_all(&dir).ok();
    }
}
