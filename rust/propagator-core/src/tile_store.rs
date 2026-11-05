//! File-backed append-only store for frozen (inactive) state tiles.
//!
//! Records are fixed-size (one tile's four arrays back to back, little
//! endian), keyed by `(realization, world_row, world_col)` of the tile's
//! top-left cell, so keys survive domain growth. Records are immutable
//! and append-only within a session: re-freezing a thawed tile appends a
//! new record, so any `{key -> offset}` index captured earlier stays
//! valid. The store itself keeps no index: each realization owns the
//! offsets of its frozen tiles, so a cloned propagator or a checkpoint
//! shares the file but never the index.
//!
//! Interior mutability (`Mutex`) lets clones and checkpoints hold an
//! `Arc<TileStore>` and read or append records concurrently.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
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
    end: u64,
}

/// The append-only record file. It holds no notion of which tiles are
/// frozen: every propagator (and every clone or checkpoint of one) keeps
/// its own `{tile -> offset}` index, so several of them can share one file
/// without seeing each other's freezes and thaws.
pub(crate) struct TileStore {
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
            inner: Mutex::new(StoreInner { file, end: 0 }),
        })
    }

    /// Bytes written so far (records are never overwritten or reclaimed).
    #[cfg(test)]
    pub fn len_bytes(&self) -> u64 {
        self.inner.lock().unwrap().end
    }

    /// Append a raw record and return its offset. Never overwrites, so
    /// offsets handed out earlier stay valid.
    pub fn append(&self, payload: &[u8]) -> Result<u64> {
        debug_assert_eq!(payload.len(), RECORD_SIZE);
        let mut inner = self.inner.lock().unwrap();
        let offset = inner.end;
        inner.file.seek(SeekFrom::Start(offset))?;
        inner.file.write_all(payload)?;
        inner.end = offset + RECORD_SIZE as u64;
        Ok(offset)
    }

    /// Append one tile's state and return the record offset.
    pub fn freeze(&self, tile: &Tile) -> Result<u64> {
        self.append(&tile_to_record(tile))
    }

    /// Read one raw record.
    pub fn read_bytes(&self, offset: u64) -> Result<Vec<u8>> {
        let mut inner = self.inner.lock().unwrap();
        inner.file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; RECORD_SIZE];
        inner.file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// Read one tile.
    pub fn read(&self, offset: u64) -> Result<Box<Tile>> {
        Ok(record_to_tile(&self.read_bytes(offset)?))
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
    fn append_only_offsets_stay_readable() {
        let dir = std::env::temp_dir().join(format!(
            "prop-core-store-test-{}",
            std::process::id()
        ));
        let store = TileStore::open(&dir).unwrap();

        let mut tile = Tile::zeroed();
        tile.arrival[5] = 42;
        let first = store.freeze(&tile).unwrap();
        // re-freezing the same tile appends; the old offset stays readable
        tile.arrival[5] = 99;
        let second = store.freeze(&tile).unwrap();
        assert_ne!(first, second);
        assert_eq!(store.len_bytes(), 2 * RECORD_SIZE as u64);
        assert_eq!(store.read(first).unwrap().arrival[5], 42);
        assert_eq!(store.read(second).unwrap().arrival[5], 99);

        std::fs::remove_dir_all(&dir).ok();
    }
}
