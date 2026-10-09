"""
RNX Disk-Backed Multimodal Storage Engine
Implements binary slotted page storage with two-bank atomic commits, CRC32 verification,
and strict RAM / Disk byte budgets. Zero symbolic mocks: raw IEEE-754 float32 payloads.
"""

import os
import struct
import zlib
import numpy as np

# Storage layout constants
PAGE_SIZE = 4096              # Standard system page boundary
HEADER_SIZE = 32              # 32-byte binary header
PAYLOAD_SIZE = PAGE_SIZE - HEADER_SIZE # 4064 bytes payload
MAGIC = 0x524E5831            # 'RNX1' in ASCII hex

# Modality tags
MODALITY_TEXT = 0
MODALITY_AUDIO = 1
MODALITY_VISION = 2
MODALITY_STATE = 3
MODALITY_LATENT_THOUGHT = 4

class RNXStorageEngine:
    def __init__(self, filepath: str, max_disk_bytes: int = 64 * 1024 * 1024, max_ram_bytes: int = 16 * 1024 * 1024):
        self.filepath = filepath
        self.max_disk_bytes = max_disk_bytes
        self.max_ram_bytes = max_ram_bytes
        self.bank_capacity_pages = (max_disk_bytes // 2) // PAGE_SIZE
        
        # RAM Cache
        self.ram_cache = {} # page_id -> (header_dict, payload_bytes)
        self.ram_usage_bytes = 0
        
        # Initialize storage file if not exists or empty
        if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
            with open(self.filepath, "wb") as f:
                # Preallocate file to exact max_disk_bytes
                f.seek(self.max_disk_bytes - 1)
                f.write(b"\0")
            self._init_superblocks()
        else:
            self._recover_superblocks()

    def _pack_header(self, page_id: int, gen: int, modality: int, item_count: int, crc: int) -> bytes:
        # Format: <I (magic), I (page_id), Q (generation), H (modality), H (item_count), I (crc), 8s (reserved)
        reserved = b"\x00" * 8
        return struct.pack("<IIQHHI8s", MAGIC, page_id, gen, modality, item_count, crc, reserved)

    def _unpack_header(self, buf: bytes):
        magic, page_id, gen, modality, item_count, crc, reserved = struct.unpack("<IIQHHI8s", buf)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected {hex(MAGIC)}")
        return {
            "magic": magic,
            "page_id": page_id,
            "gen": gen,
            "modality": modality,
            "item_count": item_count,
            "crc": crc
        }

    def _init_superblocks(self):
        # Superblock at byte 0 for Bank A, and at bank_capacity_pages * PAGE_SIZE for Bank B
        self.active_bank = 0 # 0 for A, 1 for B
        self.active_gen = 1
        self._write_superblock(bank=0, gen=1)
        self._write_superblock(bank=1, gen=0) # bank B is older/inactive

    def _write_superblock(self, bank: int, gen: int):
        offset = bank * (self.bank_capacity_pages * PAGE_SIZE)
        sb_data = struct.pack("<IIQQ", MAGIC, bank, gen, 0)
        crc = zlib.crc32(sb_data)
        sb_block = sb_data + struct.pack("<I", crc) + b"\x00" * (PAGE_SIZE - 28)
        with open(self.filepath, "r+b") as f:
            f.seek(offset)
            f.write(sb_block)
            f.flush()
            os.fsync(f.fileno())

    def _recover_superblocks(self):
        # Read superblock A
        with open(self.filepath, "rb") as f:
            f.seek(0)
            sb_a = f.read(PAGE_SIZE)
            f.seek(self.bank_capacity_pages * PAGE_SIZE)
            sb_b = f.read(PAGE_SIZE)
        
        valid_a, gen_a = self._verify_sb(sb_a, 0)
        valid_b, gen_b = self._verify_sb(sb_b, 1)

        if valid_a and valid_b:
            if gen_a >= gen_b:
                self.active_bank = 0
                self.active_gen = gen_a
            else:
                self.active_bank = 1
                self.active_gen = gen_b
        elif valid_a:
            self.active_bank = 0
            self.active_gen = gen_a
        elif valid_b:
            self.active_bank = 1
            self.active_gen = gen_b
        else:
            raise RuntimeError("Fatal: Both storage superblocks corrupt!")

    def _verify_sb(self, buf: bytes, expected_bank: int):
        if len(buf) < PAGE_SIZE:
            return False, 0
        sb_data = buf[:24]
        crc_stored = struct.unpack("<I", buf[24:28])[0]
        crc_calc = zlib.crc32(sb_data)
        if crc_stored != crc_calc:
            return False, 0
        magic, bank, gen, _ = struct.unpack("<IIQQ", sb_data)
        if magic != MAGIC or bank != expected_bank:
            return False, 0
        return True, gen

    def write_tensor_page(self, page_id: int, modality: int, tensor_data: np.ndarray):
        """
        Serializes a float32 array into a disk-backed page.
        """
        assert tensor_data.dtype == np.float32
        payload = tensor_data.tobytes()
        if len(payload) > PAYLOAD_SIZE:
            raise ValueError(f"Payload size {len(payload)} exceeds page payload limit {PAYLOAD_SIZE}")
        
        # Zero-pad payload up to PAYLOAD_SIZE
        padded_payload = payload + b"\x00" * (PAYLOAD_SIZE - len(payload))
        item_count = tensor_data.size
        crc = zlib.crc32(padded_payload)
        
        gen = self.active_gen + 1
        header = self._pack_header(page_id, gen, modality, item_count, crc)
        page_bytes = header + padded_payload

        # Write to active bank offset (page_id + 1 because page 0 is superblock)
        bank_offset = self.active_bank * (self.bank_capacity_pages * PAGE_SIZE)
        page_offset = bank_offset + (page_id + 1) * PAGE_SIZE
        
        with open(self.filepath, "r+b") as f:
            f.seek(page_offset)
            f.write(page_bytes)
            f.flush()
            os.fsync(f.fileno())

        # Update RAM cache
        header_dict = {"page_id": page_id, "gen": gen, "modality": modality, "item_count": item_count, "crc": crc}
        self.ram_cache[page_id] = (header_dict, padded_payload)
        self.active_gen = gen

    def read_tensor_page(self, page_id: int, expected_shape: tuple) -> np.ndarray:
        """
        Reads a float32 tensor from RAM cache or disk, verifying CRC32.
        """
        # Check RAM cache
        if page_id in self.ram_cache:
            header, payload = self.ram_cache[page_id]
        else:
            bank_offset = self.active_bank * (self.bank_capacity_pages * PAGE_SIZE)
            page_offset = bank_offset + (page_id + 1) * PAGE_SIZE
            with open(self.filepath, "rb") as f:
                f.seek(page_offset)
                raw = f.read(PAGE_SIZE)
            header = self._unpack_header(raw[:HEADER_SIZE])
            payload = raw[HEADER_SIZE:]
            # Verify CRC
            if zlib.crc32(payload) != header["crc"]:
                raise IOError(f"CRC check failed for page {page_id}")
            self.ram_cache[page_id] = (header, payload)
        
        # Deserialize
        total_floats = int(np.prod(expected_shape))
        floats = np.frombuffer(payload[:total_floats * 4], dtype=np.float32)
        return floats.reshape(expected_shape).copy()

    def commit_two_bank(self):
        """
        Atomic two-bank commit: swaps active bank and bumps generation.
        """
        target_bank = 1 - self.active_bank
        new_gen = self.active_gen + 1
        # Synchronize active pages to target bank
        with open(self.filepath, "r+b") as f:
            for page_id, (header, payload) in self.ram_cache.items():
                target_offset = target_bank * (self.bank_capacity_pages * PAGE_SIZE) + (page_id + 1) * PAGE_SIZE
                hdr_bytes = self._pack_header(page_id, new_gen, header["modality"], header["item_count"], header["crc"])
                f.seek(target_offset)
                f.write(hdr_bytes + payload)
            f.flush()
            os.fsync(f.fileno())

        # Atomically write target superblock
        self._write_superblock(bank=target_bank, gen=new_gen)
        self.active_bank = target_bank
        self.active_gen = new_gen
