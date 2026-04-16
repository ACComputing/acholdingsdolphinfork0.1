#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  MewCoreDolphin 1.x  —  GameCube Emulator (single-file Python + Cython)
#  (C) 1999-2026 A.C Holdings / Team Flames
#  Brand: catsanzsh / realflameselite / @ItsJustaCat00
#  License: MIT — vibe-engineered, production-style single-file
# =============================================================================
#
#  Features:
#    - Tkinter GUI with black bg, blue (#4da6ff) text/buttons
#    - Embedded Cython core "mewcoredolphin" (auto-compiled, pure-python fallback)
#    - PowerPC 750CL (Gekko) CPU interpreter (subset + dispatch skeleton)
#    - BAT/MMU stubs, interrupt dispatch, DSP mailboxes, GPU CP FIFO stubs
#    - DOL / GCM loaders
#    - MD5 hash-cached Cython builds in ~/.mewcoredolphin/cache/
#
#  Drop in a .dol or .gcm/.iso and hit RUN.
# =============================================================================

import os
import sys
import io
import time
import struct
import hashlib
import tempfile
import threading
import traceback
import importlib
import importlib.util
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# -----------------------------------------------------------------------------
# Theme — black / blue signature look
# -----------------------------------------------------------------------------
BG          = "#000000"
FG          = "#4da6ff"
FG_DIM      = "#2a6fb0"
ACCENT      = "#4488FF"
BTN_BG      = "#050a14"
BTN_ACTIVE  = "#0a1a33"
BTN_FG      = "#4da6ff"
DISABLED_FG = "#1a3a66"
FONT_MONO   = ("Consolas", 10)
FONT_UI     = ("Segoe UI", 10)
FONT_TITLE  = ("Segoe UI", 14, "bold")

APP_NAME    = "MewCoreDolphin"
APP_VER     = "1.0"
COPYRIGHT   = "(C) 1999-2026 A.C Holdings / Team Flames"

# =============================================================================
#  CYTHON SOURCE — mewcoredolphin core
#  Compiled on demand; falls back to pure-Python twin if Cython unavailable.
# =============================================================================
CYTHON_SOURCE = r'''
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# mewcoredolphin 1.x — Gekko (PowerPC 750CL) interpreter core
# (C) 1999-2026 A.C Holdings / Team Flames

import cython
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t

# ---- Memory map constants (simplified GameCube map) ------------------------
# MEM1:   0x80000000 .. 0x81800000  (24 MiB main RAM, cached)
# MEM1u:  0xC0000000 .. 0xC1800000  (same, uncached mirror)
# MEM2 etc. omitted for the base 1.x core.

cdef uint32_t MEM1_BASE   = 0x80000000
cdef uint32_t MEM1_SIZE   = 24 * 1024 * 1024

cdef class GekkoCore:
    """PowerPC 750CL (Gekko) interpreter — mewcoredolphin 1.x."""

    cdef public uint32_t[32] gpr          # general-purpose registers r0..r31
    cdef public double[32]   fpr          # floating-point registers
    cdef public uint32_t pc               # program counter
    cdef public uint32_t lr               # link register
    cdef public uint32_t ctr              # count register
    cdef public uint32_t cr               # condition register (packed CR0..CR7)
    cdef public uint32_t xer              # fixed-point exception register
    cdef public uint32_t msr              # machine state register
    cdef public uint64_t cycles           # total cycles executed
    cdef public bint     running
    cdef bytearray       _ram             # MEM1

    def __cinit__(self):
        cdef int i
        for i in range(32):
            self.gpr[i] = 0
            self.fpr[i] = 0.0
        self.pc   = 0
        self.lr   = 0
        self.ctr  = 0
        self.cr   = 0
        self.xer  = 0
        self.msr  = 0x00002000   # FP enabled
        self.cycles  = 0
        self.running = False
        self._ram = bytearray(MEM1_SIZE)

    # -- RAM access ----------------------------------------------------------
    cpdef bytes ram_view(self, uint32_t off, uint32_t n):
        return bytes(self._ram[off:off+n])

    cpdef void ram_write(self, uint32_t off, bytes data):
        cdef Py_ssize_t n = len(data)
        self._ram[off:off+n] = data

    cdef inline uint32_t _translate(self, uint32_t ea):
        # Super-simplified BAT: MEM1 cached or uncached mirror -> physical.
        if MEM1_BASE <= ea < MEM1_BASE + MEM1_SIZE:
            return ea - MEM1_BASE
        if 0xC0000000 <= ea < 0xC0000000 + MEM1_SIZE:
            return ea - 0xC0000000
        # Fall-through: clamp into RAM to keep the interpreter alive for stubs.
        return ea & (MEM1_SIZE - 1)

    cdef inline uint32_t _read32(self, uint32_t ea):
        cdef uint32_t pa = self._translate(ea)
        # PowerPC is big-endian
        return ((self._ram[pa]     << 24) |
                (self._ram[pa + 1] << 16) |
                (self._ram[pa + 2] <<  8) |
                 self._ram[pa + 3])

    cdef inline void _write32(self, uint32_t ea, uint32_t v):
        cdef uint32_t pa = self._translate(ea)
        self._ram[pa]     = (v >> 24) & 0xFF
        self._ram[pa + 1] = (v >> 16) & 0xFF
        self._ram[pa + 2] = (v >>  8) & 0xFF
        self._ram[pa + 3] =  v        & 0xFF

    # Python-visible wrappers
    cpdef uint32_t read32(self, uint32_t ea):
        return self._read32(ea)

    cpdef void write32(self, uint32_t ea, uint32_t v):
        self._write32(ea, v)

    # -- CR0 helpers ---------------------------------------------------------
    cdef inline void _set_cr0(self, int32_t r):
        cdef uint32_t f
        if r < 0:
            f = 0x8
        elif r > 0:
            f = 0x4
        else:
            f = 0x2
        # Clear top nibble (CR0) and insert
        self.cr = (self.cr & 0x0FFFFFFF) | (f << 28)

    # -- Single-step dispatcher ---------------------------------------------
    cpdef int step(self):
        """Execute one Gekko instruction. Returns cycles consumed."""
        cdef uint32_t op   = self._read32(self.pc)
        cdef uint32_t prim = (op >> 26) & 0x3F
        cdef uint32_t rD, rA, rB, rS
        cdef int32_t  simm
        cdef uint32_t uimm
        cdef int32_t  bd
        cdef uint32_t ext
        cdef uint32_t ea

        # Default next PC
        cdef uint32_t npc = self.pc + 4

        if prim == 14:                       # addi   rD, rA, SIMM
            rD   = (op >> 21) & 0x1F
            rA   = (op >> 16) & 0x1F
            simm = <int32_t><int16_t>(op & 0xFFFF)
            self.gpr[rD] = (self.gpr[rA] if rA != 0 else 0) + <uint32_t>simm

        elif prim == 15:                     # addis  rD, rA, SIMM
            rD   = (op >> 21) & 0x1F
            rA   = (op >> 16) & 0x1F
            uimm = (op & 0xFFFF) << 16
            self.gpr[rD] = (self.gpr[rA] if rA != 0 else 0) + uimm

        elif prim == 24:                     # ori    rA, rS, UIMM
            rS   = (op >> 21) & 0x1F
            rA   = (op >> 16) & 0x1F
            uimm = op & 0xFFFF
            self.gpr[rA] = self.gpr[rS] | uimm

        elif prim == 25:                     # oris   rA, rS, UIMM
            rS   = (op >> 21) & 0x1F
            rA   = (op >> 16) & 0x1F
            uimm = (op & 0xFFFF) << 16
            self.gpr[rA] = self.gpr[rS] | uimm

        elif prim == 32:                     # lwz    rD, d(rA)
            rD   = (op >> 21) & 0x1F
            rA   = (op >> 16) & 0x1F
            simm = <int32_t><int16_t>(op & 0xFFFF)
            ea   = (self.gpr[rA] if rA != 0 else 0) + <uint32_t>simm
            self.gpr[rD] = self._read32(ea)

        elif prim == 36:                     # stw    rS, d(rA)
            rS   = (op >> 21) & 0x1F
            rA   = (op >> 16) & 0x1F
            simm = <int32_t><int16_t>(op & 0xFFFF)
            ea   = (self.gpr[rA] if rA != 0 else 0) + <uint32_t>simm
            self._write32(ea, self.gpr[rS])

        elif prim == 18:                     # b / bl / ba / bla
            bd = <int32_t>(op & 0x03FFFFFC)
            if bd & 0x02000000:              # sign extend 26-bit
                bd |= <int32_t>0xFC000000
            if (op & 1) != 0:                # LK
                self.lr = self.pc + 4
            if (op & 2) != 0:                # AA
                npc = <uint32_t>bd
            else:
                npc = self.pc + <uint32_t>bd

        elif prim == 19 and ((op >> 1) & 0x3FF) == 16:  # bclr / blr
            npc = self.lr & 0xFFFFFFFC
            if (op & 1) != 0:
                self.lr = self.pc + 4

        elif prim == 31:                     # X-form ops — small subset
            ext = (op >> 1) & 0x3FF
            rD  = (op >> 21) & 0x1F
            rA  = (op >> 16) & 0x1F
            rB  = (op >> 11) & 0x1F
            if ext == 266:                   # add rD, rA, rB
                self.gpr[rD] = self.gpr[rA] + self.gpr[rB]
                if (op & 1) != 0:
                    self._set_cr0(<int32_t>self.gpr[rD])
            elif ext == 40:                  # subf rD, rA, rB
                self.gpr[rD] = self.gpr[rB] - self.gpr[rA]
                if (op & 1) != 0:
                    self._set_cr0(<int32_t>self.gpr[rD])
            elif ext == 444:                 # or rA, rS, rB  (mr = or rA,rS,rS)
                rS = (op >> 21) & 0x1F
                self.gpr[rA] = self.gpr[rS] | self.gpr[rB]
                if (op & 1) != 0:
                    self._set_cr0(<int32_t>self.gpr[rA])
            else:
                # Unimplemented — treat as NOP and keep chugging.
                pass
        else:
            # Unimplemented primary — NOP for the stub dispatcher.
            pass

        self.pc = npc
        self.cycles += 1
        return 1

    cpdef uint64_t run(self, uint64_t max_cycles):
        """Run up to max_cycles. Returns cycles actually executed."""
        cdef uint64_t start = self.cycles
        self.running = True
        while self.running and (self.cycles - start) < max_cycles:
            self.step()
        return self.cycles - start

    cpdef void stop(self):
        self.running = False


# ---- DSP mailbox stubs -----------------------------------------------------
cdef class DSP:
    cdef public uint32_t cpu_mbox_hi, cpu_mbox_lo
    cdef public uint32_t dsp_mbox_hi, dsp_mbox_lo
    def __cinit__(self):
        self.cpu_mbox_hi = 0
        self.cpu_mbox_lo = 0
        self.dsp_mbox_hi = 0x80000000  # "mail present" sentinel for stubs
        self.dsp_mbox_lo = 0

# ---- GPU Command Processor FIFO stub ---------------------------------------
cdef class GPU_CP:
    cdef public uint32_t fifo_base, fifo_end, fifo_wr, fifo_rd
    cdef public bint     gp_link_enable
    def __cinit__(self):
        self.fifo_base = 0
        self.fifo_end  = 0
        self.fifo_wr   = 0
        self.fifo_rd   = 0
        self.gp_link_enable = False


CORE_NAME    = "mewcoredolphin"
CORE_VERSION = "1.0-cython"

def banner():
    return "mewcoredolphin 1.x (Cython) — Gekko interpreter online."
'''

# =============================================================================
#  CORE LOADER — Cython build w/ MD5 cache, fall back to pure Python twin
# =============================================================================
def _core_cache_dir() -> Path:
    d = Path.home() / ".mewcoredolphin" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_cython_core(log):
    """Compile CYTHON_SOURCE into an importable module. Cache by MD5 hash."""
    try:
        import Cython  # noqa: F401
        from Cython.Build import cythonize
        from setuptools import setup, Extension
    except Exception as e:
        log(f"[core] Cython unavailable ({e}) — using pure-Python core.")
        return None

    h = hashlib.md5(CYTHON_SOURCE.encode("utf-8")).hexdigest()[:12]
    mod_name = f"mewcoredolphin_{h}"
    cache = _core_cache_dir()

    # Is it already built?
    for p in cache.iterdir():
        if p.name.startswith(mod_name) and p.suffix in (".pyd", ".so"):
            log(f"[core] Cached Cython core found: {p.name}")
            spec = importlib.util.spec_from_file_location(mod_name, p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

    log("[core] Building Cython core (mewcoredolphin 1.x) ...")
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        src = tmp / f"{mod_name}.pyx"
        src.write_text(CYTHON_SOURCE, encoding="utf-8")

        setup_py = tmp / "setup.py"
        setup_py.write_text(
            "from setuptools import setup\n"
            "from Cython.Build import cythonize\n"
            f"setup(ext_modules=cythonize([r'{src.as_posix()}'], "
            "language_level=3))\n",
            encoding="utf-8"
        )
        try:
            proc = subprocess.run(
                [sys.executable, str(setup_py), "build_ext", "--inplace"],
                cwd=tmp, capture_output=True, text=True, timeout=180
            )
            if proc.returncode != 0:
                log("[core] Cython build FAILED:")
                log(proc.stdout[-1200:])
                log(proc.stderr[-1200:])
                return None

            built = None
            for p in tmp.iterdir():
                if p.name.startswith(mod_name) and p.suffix in (".pyd", ".so"):
                    built = p
                    break
            if built is None:
                log("[core] Build succeeded but artifact not found.")
                return None

            dst = cache / built.name
            dst.write_bytes(built.read_bytes())
            log(f"[core] Cython core built and cached: {dst.name}")

            spec = importlib.util.spec_from_file_location(mod_name, dst)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            log(f"[core] Cython build exception: {e}")
            return None


# --------- Pure-Python twin (correctness mirror of the Cython class) --------
class _PyGekkoCore:
    """Pure-python Gekko interpreter — slower, but always available."""
    MEM1_BASE = 0x80000000
    MEM1_SIZE = 24 * 1024 * 1024

    def __init__(self):
        self.gpr = [0] * 32
        self.fpr = [0.0] * 32
        self.pc = 0
        self.lr = 0
        self.ctr = 0
        self.cr = 0
        self.xer = 0
        self.msr = 0x00002000
        self.cycles = 0
        self.running = False
        self._ram = bytearray(self.MEM1_SIZE)

    def ram_view(self, off, n):
        return bytes(self._ram[off:off + n])

    def ram_write(self, off, data):
        self._ram[off:off + len(data)] = data

    def _translate(self, ea):
        if self.MEM1_BASE <= ea < self.MEM1_BASE + self.MEM1_SIZE:
            return ea - self.MEM1_BASE
        if 0xC0000000 <= ea < 0xC0000000 + self.MEM1_SIZE:
            return ea - 0xC0000000
        return ea & (self.MEM1_SIZE - 1)

    def read32(self, ea):
        pa = self._translate(ea)
        return struct.unpack(">I", bytes(self._ram[pa:pa + 4]))[0]

    def write32(self, ea, v):
        pa = self._translate(ea)
        self._ram[pa:pa + 4] = struct.pack(">I", v & 0xFFFFFFFF)

    def _sx16(self, v):
        return v - 0x10000 if v & 0x8000 else v

    def _set_cr0(self, r):
        r = r - 0x100000000 if r & 0x80000000 else r
        if r < 0:
            f = 0x8
        elif r > 0:
            f = 0x4
        else:
            f = 0x2
        self.cr = (self.cr & 0x0FFFFFFF) | (f << 28)

    def step(self):
        op = self.read32(self.pc)
        prim = (op >> 26) & 0x3F
        npc = (self.pc + 4) & 0xFFFFFFFF

        if prim == 14:                      # addi
            rD = (op >> 21) & 0x1F
            rA = (op >> 16) & 0x1F
            simm = self._sx16(op & 0xFFFF)
            base = self.gpr[rA] if rA != 0 else 0
            self.gpr[rD] = (base + simm) & 0xFFFFFFFF
        elif prim == 15:                    # addis
            rD = (op >> 21) & 0x1F
            rA = (op >> 16) & 0x1F
            base = self.gpr[rA] if rA != 0 else 0
            self.gpr[rD] = (base + ((op & 0xFFFF) << 16)) & 0xFFFFFFFF
        elif prim == 24:                    # ori
            rS = (op >> 21) & 0x1F
            rA = (op >> 16) & 0x1F
            self.gpr[rA] = (self.gpr[rS] | (op & 0xFFFF)) & 0xFFFFFFFF
        elif prim == 25:                    # oris
            rS = (op >> 21) & 0x1F
            rA = (op >> 16) & 0x1F
            self.gpr[rA] = (self.gpr[rS] | ((op & 0xFFFF) << 16)) & 0xFFFFFFFF
        elif prim == 32:                    # lwz
            rD = (op >> 21) & 0x1F
            rA = (op >> 16) & 0x1F
            simm = self._sx16(op & 0xFFFF)
            base = self.gpr[rA] if rA != 0 else 0
            self.gpr[rD] = self.read32((base + simm) & 0xFFFFFFFF)
        elif prim == 36:                    # stw
            rS = (op >> 21) & 0x1F
            rA = (op >> 16) & 0x1F
            simm = self._sx16(op & 0xFFFF)
            base = self.gpr[rA] if rA != 0 else 0
            self.write32((base + simm) & 0xFFFFFFFF, self.gpr[rS])
        elif prim == 18:                    # b / bl / ba / bla
            bd = op & 0x03FFFFFC
            if bd & 0x02000000:
                bd -= 0x04000000
            if op & 1:
                self.lr = (self.pc + 4) & 0xFFFFFFFF
            if op & 2:
                npc = bd & 0xFFFFFFFF
            else:
                npc = (self.pc + bd) & 0xFFFFFFFF
        elif prim == 19 and ((op >> 1) & 0x3FF) == 16:  # blr
            npc = self.lr & 0xFFFFFFFC
            if op & 1:
                self.lr = (self.pc + 4) & 0xFFFFFFFF
        elif prim == 31:
            ext = (op >> 1) & 0x3FF
            rD = (op >> 21) & 0x1F
            rA = (op >> 16) & 0x1F
            rB = (op >> 11) & 0x1F
            if ext == 266:                  # add
                self.gpr[rD] = (self.gpr[rA] + self.gpr[rB]) & 0xFFFFFFFF
                if op & 1:
                    self._set_cr0(self.gpr[rD])
            elif ext == 40:                 # subf
                self.gpr[rD] = (self.gpr[rB] - self.gpr[rA]) & 0xFFFFFFFF
                if op & 1:
                    self._set_cr0(self.gpr[rD])
            elif ext == 444:                # or (mr)
                rS = (op >> 21) & 0x1F
                self.gpr[rA] = (self.gpr[rS] | self.gpr[rB]) & 0xFFFFFFFF
                if op & 1:
                    self._set_cr0(self.gpr[rA])
            # else: NOP
        # else: NOP

        self.pc = npc
        self.cycles += 1
        return 1

    def run(self, max_cycles):
        start = self.cycles
        self.running = True
        while self.running and (self.cycles - start) < max_cycles:
            self.step()
        return self.cycles - start

    def stop(self):
        self.running = False


class _PyDSP:
    def __init__(self):
        self.cpu_mbox_hi = 0
        self.cpu_mbox_lo = 0
        self.dsp_mbox_hi = 0x80000000
        self.dsp_mbox_lo = 0


class _PyGPU_CP:
    def __init__(self):
        self.fifo_base = 0
        self.fifo_end = 0
        self.fifo_wr = 0
        self.fifo_rd = 0
        self.gp_link_enable = False


class _PyCoreModule:
    CORE_NAME = "mewcoredolphin"
    CORE_VERSION = "1.0-python"
    GekkoCore = _PyGekkoCore
    DSP = _PyDSP
    GPU_CP = _PyGPU_CP

    @staticmethod
    def banner():
        return "mewcoredolphin 1.x (pure Python) — Gekko interpreter online."


def load_core(log):
    mod = _build_cython_core(log)
    if mod is None:
        log("[core] Falling back to pure-Python mewcoredolphin core.")
        return _PyCoreModule()
    log(f"[core] Loaded: {mod.banner()}")
    return mod


# =============================================================================
#  ROM LOADERS — DOL and GCM/ISO
# =============================================================================
class DOLLoader:
    """Load a Nintendo .dol executable into MEM1 and set entry point."""

    @staticmethod
    def load(core_cpu, data: bytes, log) -> int:
        if len(data) < 0x100:
            raise ValueError("DOL too small")

        text_off  = struct.unpack(">7I", data[0x00:0x1C])
        data_off  = struct.unpack(">11I", data[0x1C:0x48])
        text_addr = struct.unpack(">7I", data[0x48:0x64])
        data_addr = struct.unpack(">11I", data[0x64:0x90])
        text_sz   = struct.unpack(">7I", data[0x90:0xAC])
        data_sz   = struct.unpack(">11I", data[0xAC:0xD8])
        bss_addr, bss_size, entry = struct.unpack(">III", data[0xD8:0xE4])

        for i, (off, addr, sz) in enumerate(zip(text_off, text_addr, text_sz)):
            if off and sz:
                chunk = data[off:off + sz]
                pa = core_cpu._translate(addr) if hasattr(core_cpu, "_translate") else \
                     (addr - 0x80000000) & 0xFFFFFFFF
                core_cpu.ram_write(pa, chunk)
                log(f"[dol] TEXT{i}: {sz:>8} bytes -> {addr:#010x}")

        for i, (off, addr, sz) in enumerate(zip(data_off, data_addr, data_sz)):
            if off and sz:
                chunk = data[off:off + sz]
                pa = core_cpu._translate(addr) if hasattr(core_cpu, "_translate") else \
                     (addr - 0x80000000) & 0xFFFFFFFF
                core_cpu.ram_write(pa, chunk)
                log(f"[dol] DATA{i}: {sz:>8} bytes -> {addr:#010x}")

        if bss_size:
            log(f"[dol] BSS : {bss_size:>8} bytes -> {bss_addr:#010x}")

        core_cpu.pc = entry
        log(f"[dol] Entry: {entry:#010x}")
        return entry


class GCMLoader:
    """Peek a GCM/ISO header and extract the bootable DOL."""

    @staticmethod
    def load(core_cpu, data: bytes, log) -> int:
        if len(data) < 0x440:
            raise ValueError("GCM too small")
        game_code = data[0x00:0x06].decode("ascii", errors="replace").rstrip("\x00")
        title     = data[0x20:0x20 + 0x60].split(b"\x00", 1)[0]
        try:
            title_s = title.decode("utf-8", errors="replace")
        except Exception:
            title_s = "<unreadable>"
        log(f"[gcm] Game code: {game_code}   Title: {title_s}")

        dol_off  = struct.unpack(">I", data[0x420:0x424])[0]
        fst_off  = struct.unpack(">I", data[0x424:0x428])[0]
        log(f"[gcm] DOL @ {dol_off:#010x}   FST @ {fst_off:#010x}")

        if dol_off == 0 or dol_off >= len(data):
            raise ValueError("GCM has no valid DOL offset")

        dol_end = fst_off if fst_off > dol_off else min(len(data), dol_off + 0x800000)
        dol_blob = data[dol_off:dol_end]
        return DOLLoader.load(core_cpu, dol_blob, log)


# =============================================================================
#  EMULATOR HOST — owns CPU, DSP, GPU, and thread lifecycle
# =============================================================================
class EmulatorHost:
    def __init__(self, log_cb):
        self.log = log_cb
        self.core_mod = load_core(self.log)
        self.cpu = self.core_mod.GekkoCore()
        self.dsp = self.core_mod.DSP()
        self.gpu = self.core_mod.GPU_CP()
        self._thread = None
        self._stop_flag = threading.Event()
        self.loaded_path = None
        self.fps = 0.0
        self.mips = 0.0

    def reset(self):
        self.cpu = self.core_mod.GekkoCore()
        self.dsp = self.core_mod.DSP()
        self.gpu = self.core_mod.GPU_CP()
        self.fps = 0.0
        self.mips = 0.0
        self.log("[host] System reset.")

    def load_file(self, path: str):
        self.reset()
        data = Path(path).read_bytes()
        low = path.lower()
        if low.endswith(".dol"):
            DOLLoader.load(self.cpu, data, self.log)
        elif low.endswith((".gcm", ".iso")):
            GCMLoader.load(self.cpu, data, self.log)
        else:
            # Try to sniff: DOL has text[0] offset at 0x00
            if len(data) >= 0x440 and data[0x1C:0x1E] != b"\x00\x00":
                DOLLoader.load(self.cpu, data, self.log)
            else:
                GCMLoader.load(self.cpu, data, self.log)
        self.loaded_path = path
        self.log(f"[host] Loaded: {os.path.basename(path)}")

    def _run_loop(self):
        SLICE = 100_000     # cycles per slice
        last = time.perf_counter()
        slice_cycles = 0
        try:
            while not self._stop_flag.is_set():
                ran = self.cpu.run(SLICE)
                slice_cycles += ran
                now = time.perf_counter()
                if now - last >= 0.5:
                    dt = now - last
                    self.mips = (slice_cycles / 1_000_000.0) / dt
                    self.fps = min(60.0, self.mips * 2.0)  # cosmetic
                    slice_cycles = 0
                    last = now
        except Exception:
            self.log("[host] CPU thread crashed:")
            self.log(traceback.format_exc())
        finally:
            self.cpu.stop()
            self.log("[host] CPU thread exited.")

    def start(self):
        if self._thread and self._thread.is_alive():
            self.log("[host] Already running.")
            return
        if self.loaded_path is None:
            self.log("[host] No ROM loaded.")
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run_loop,
                                        name="MewCoreGekko",
                                        daemon=True)
        self._thread.start()
        self.log("[host] Emulation started.")

    def stop(self):
        self._stop_flag.set()
        try:
            self.cpu.stop()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=1.0)
        self.log("[host] Emulation stopped.")


# =============================================================================
#  TKINTER GUI
# =============================================================================
class MewCoreDolphinGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} {APP_VER}  —  GameCube Emulator")
        self.geometry("980x640")
        self.configure(bg=BG)
        self.minsize(880, 560)

        self._apply_ttk_style()
        self._build_menubar()
        self._build_toolbar()
        self._build_main()
        self._build_statusbar()

        self.host = EmulatorHost(self._log)
        self._log(f"{APP_NAME} {APP_VER}  {COPYRIGHT}")
        self._log(f"Python {sys.version.split()[0]}  on  {sys.platform}")
        self._log("Ready. Load a .dol / .gcm / .iso to begin.")

        self.after(250, self._tick_status)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- styling ----------------------------------------------------------
    def _apply_ttk_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG, font=FONT_UI)
        style.configure("Title.TLabel", background=BG, foreground=FG,
                        font=FONT_TITLE)
        style.configure("Dim.TLabel", background=BG, foreground=FG_DIM,
                        font=FONT_UI)

        style.configure("Blue.TButton",
                        background=BTN_BG,
                        foreground=BTN_FG,
                        bordercolor=FG,
                        lightcolor=FG,
                        darkcolor=FG_DIM,
                        focusthickness=1,
                        focuscolor=FG,
                        font=FONT_UI,
                        padding=(14, 6))
        style.map("Blue.TButton",
                  background=[("active", BTN_ACTIVE),
                              ("disabled", BTN_BG)],
                  foreground=[("disabled", DISABLED_FG),
                              ("active", FG)])

        style.configure("TSeparator", background=FG_DIM)
        style.configure("Horizontal.TProgressbar",
                        troughcolor=BG, background=FG, bordercolor=FG_DIM,
                        lightcolor=FG, darkcolor=FG_DIM)

    # ---- menubar ----------------------------------------------------------
    def _build_menubar(self):
        mb = tk.Menu(self, bg=BG, fg=FG, activebackground=BTN_ACTIVE,
                     activeforeground=FG, tearoff=False, bd=0)

        m_file = tk.Menu(mb, bg=BG, fg=FG, activebackground=BTN_ACTIVE,
                         activeforeground=FG, tearoff=False)
        m_file.add_command(label="Open ROM...", accelerator="Ctrl+O",
                           command=self._action_open)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self._on_close)
        mb.add_cascade(label="File", menu=m_file)

        m_emu = tk.Menu(mb, bg=BG, fg=FG, activebackground=BTN_ACTIVE,
                        activeforeground=FG, tearoff=False)
        m_emu.add_command(label="Run",   accelerator="F5", command=self._action_run)
        m_emu.add_command(label="Stop",  accelerator="Shift+F5", command=self._action_stop)
        m_emu.add_command(label="Reset", accelerator="Ctrl+R", command=self._action_reset)
        mb.add_cascade(label="Emulation", menu=m_emu)

        m_dbg = tk.Menu(mb, bg=BG, fg=FG, activebackground=BTN_ACTIVE,
                        activeforeground=FG, tearoff=False)
        m_dbg.add_command(label="Dump CPU State", command=self._action_dump_cpu)
        m_dbg.add_command(label="Clear Log", command=self._action_clear_log)
        mb.add_cascade(label="Debug", menu=m_dbg)

        m_help = tk.Menu(mb, bg=BG, fg=FG, activebackground=BTN_ACTIVE,
                         activeforeground=FG, tearoff=False)
        m_help.add_command(label="About", command=self._action_about)
        mb.add_cascade(label="Help", menu=m_help)

        self.config(menu=mb)

        self.bind_all("<Control-o>",       lambda e: self._action_open())
        self.bind_all("<F5>",              lambda e: self._action_run())
        self.bind_all("<Shift-F5>",        lambda e: self._action_stop())
        self.bind_all("<Control-r>",       lambda e: self._action_reset())

    # ---- toolbar ----------------------------------------------------------
    def _build_toolbar(self):
        bar = tk.Frame(self, bg=BG, highlightbackground=FG_DIM,
                       highlightthickness=1)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 0))

        btn_open  = ttk.Button(bar, text="OPEN ROM", style="Blue.TButton",
                               command=self._action_open)
        btn_run   = ttk.Button(bar, text="RUN",      style="Blue.TButton",
                               command=self._action_run)
        btn_stop  = ttk.Button(bar, text="STOP",     style="Blue.TButton",
                               command=self._action_stop)
        btn_reset = ttk.Button(bar, text="RESET",    style="Blue.TButton",
                               command=self._action_reset)
        btn_dump  = ttk.Button(bar, text="DUMP CPU", style="Blue.TButton",
                               command=self._action_dump_cpu)

        for b in (btn_open, btn_run, btn_stop, btn_reset, btn_dump):
            b.pack(side=tk.LEFT, padx=4, pady=6)

        title = ttk.Label(bar,
                          text=f"{APP_NAME} {APP_VER}",
                          style="Title.TLabel")
        title.pack(side=tk.RIGHT, padx=10)

    # ---- main area --------------------------------------------------------
    def _build_main(self):
        main = tk.Frame(self, bg=BG)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Left: "screen" panel (placeholder canvas with blue border)
        left = tk.Frame(main, bg=BG, highlightbackground=FG,
                        highlightthickness=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.screen = tk.Canvas(left, bg=BG, bd=0, highlightthickness=0)
        self.screen.pack(fill=tk.BOTH, expand=True)
        self._draw_screen_placeholder()
        self.screen.bind("<Configure>", lambda e: self._draw_screen_placeholder())

        # Right: info + log column
        right = tk.Frame(main, bg=BG, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # CPU state panel
        info_box = tk.LabelFrame(right, text=" Gekko CPU ", bg=BG, fg=FG,
                                 bd=1, relief=tk.SOLID,
                                 highlightbackground=FG_DIM,
                                 font=FONT_UI, labelanchor="nw")
        info_box.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        self.lbl_pc     = tk.Label(info_box, text="PC  : 0x00000000", bg=BG,
                                   fg=FG, font=FONT_MONO, anchor="w")
        self.lbl_lr     = tk.Label(info_box, text="LR  : 0x00000000", bg=BG,
                                   fg=FG, font=FONT_MONO, anchor="w")
        self.lbl_ctr    = tk.Label(info_box, text="CTR : 0x00000000", bg=BG,
                                   fg=FG, font=FONT_MONO, anchor="w")
        self.lbl_cyc    = tk.Label(info_box, text="CYC : 0",          bg=BG,
                                   fg=FG, font=FONT_MONO, anchor="w")
        self.lbl_mips   = tk.Label(info_box, text="MIPS: 0.00",       bg=BG,
                                   fg=FG, font=FONT_MONO, anchor="w")
        for w in (self.lbl_pc, self.lbl_lr, self.lbl_ctr,
                  self.lbl_cyc, self.lbl_mips):
            w.pack(fill=tk.X, padx=8, pady=1)
        tk.Label(info_box, text="", bg=BG).pack()

        # Log
        log_box = tk.LabelFrame(right, text=" Log ", bg=BG, fg=FG, bd=1,
                                relief=tk.SOLID, font=FONT_UI, labelanchor="nw")
        log_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.log_widget = scrolledtext.ScrolledText(
            log_box, bg=BG, fg=FG, insertbackground=FG,
            font=FONT_MONO, wrap=tk.WORD, height=18,
            bd=0, highlightthickness=0
        )
        self.log_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.log_widget.configure(state=tk.DISABLED)

    def _draw_screen_placeholder(self):
        c = self.screen
        c.delete("all")
        w = max(c.winfo_width(), 10)
        h = max(c.winfo_height(), 10)
        c.create_rectangle(4, 4, w - 4, h - 4, outline=FG, width=1)
        c.create_text(w // 2, h // 2 - 22,
                      text=f"{APP_NAME} {APP_VER}",
                      fill=FG, font=("Segoe UI", 22, "bold"))
        c.create_text(w // 2, h // 2 + 8,
                      text="MewCoreDolphin — Gekko (PowerPC 750CL) / DSP / CP FIFO",
                      fill=FG_DIM, font=("Segoe UI", 10))
        c.create_text(w // 2, h // 2 + 30,
                      text=COPYRIGHT,
                      fill=FG_DIM, font=("Segoe UI", 9))

    # ---- statusbar --------------------------------------------------------
    def _build_statusbar(self):
        bar = tk.Frame(self, bg=BG, highlightbackground=FG_DIM,
                       highlightthickness=1)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.lbl_status = tk.Label(bar, text="Idle.", bg=BG, fg=FG,
                                   font=FONT_MONO, anchor="w")
        self.lbl_status.pack(side=tk.LEFT, padx=8, pady=3)
        self.lbl_rom    = tk.Label(bar, text="No ROM", bg=BG, fg=FG_DIM,
                                   font=FONT_MONO, anchor="e")
        self.lbl_rom.pack(side=tk.RIGHT, padx=8, pady=3)

    # ---- log --------------------------------------------------------------
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            self.log_widget.configure(state=tk.NORMAL)
            self.log_widget.insert(tk.END, line)
            self.log_widget.see(tk.END)
            self.log_widget.configure(state=tk.DISABLED)
        except Exception:
            # Log widget not built yet
            sys.stdout.write(line)

    # ---- actions ----------------------------------------------------------
    def _action_open(self):
        path = filedialog.askopenfilename(
            title="Open GameCube ROM",
            filetypes=[
                ("GameCube ROMs", "*.dol *.gcm *.iso"),
                ("DOL executable", "*.dol"),
                ("GCM / ISO image", "*.gcm *.iso"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return
        try:
            self.host.stop()
            self.host.load_file(path)
            self.lbl_rom.configure(text=os.path.basename(path), fg=FG)
        except Exception as e:
            self._log(f"[err] load failed: {e}")
            messagebox.showerror("Load failed", str(e))

    def _action_run(self):
        self.host.start()

    def _action_stop(self):
        self.host.stop()

    def _action_reset(self):
        self.host.stop()
        self.host.reset()

    def _action_dump_cpu(self):
        cpu = self.host.cpu
        lines = [f"PC  = {cpu.pc:#010x}",
                 f"LR  = {cpu.lr:#010x}",
                 f"CTR = {cpu.ctr:#010x}",
                 f"CR  = {cpu.cr:#010x}",
                 f"XER = {cpu.xer:#010x}",
                 f"MSR = {cpu.msr:#010x}",
                 f"CYC = {cpu.cycles}"]
        lines.append("")
        lines.append("GPRs:")
        for row in range(8):
            chunk = "  ".join(
                f"r{row*4+c:02d}={cpu.gpr[row*4+c]:#010x}" for c in range(4)
            )
            lines.append("  " + chunk)
        self._log("\n".join(lines))

    def _action_clear_log(self):
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.delete("1.0", tk.END)
        self.log_widget.configure(state=tk.DISABLED)

    def _action_about(self):
        messagebox.showinfo(
            f"About {APP_NAME}",
            f"{APP_NAME} {APP_VER}\n"
            f"GameCube emulator\n"
            f"MewCoreDolphin 1.x (Gekko + DSP + CP FIFO)\n\n"
            f"{COPYRIGHT}"
        )

    # ---- periodic status tick --------------------------------------------
    def _tick_status(self):
        try:
            cpu = self.host.cpu
            self.lbl_pc.configure(text=f"PC  : {cpu.pc:#010x}")
            self.lbl_lr.configure(text=f"LR  : {cpu.lr:#010x}")
            self.lbl_ctr.configure(text=f"CTR : {cpu.ctr:#010x}")
            self.lbl_cyc.configure(text=f"CYC : {cpu.cycles}")
            self.lbl_mips.configure(text=f"MIPS: {self.host.mips:6.2f}")

            running = (self.host._thread is not None
                       and self.host._thread.is_alive())
            if running:
                self.lbl_status.configure(
                    text=f"Running — {self.host.mips:6.2f} MIPS"
                )
            else:
                self.lbl_status.configure(
                    text="Idle." if self.host.loaded_path is None
                    else "Stopped."
                )
        except Exception:
            pass
        self.after(250, self._tick_status)

    def _on_close(self):
        try:
            self.host.stop()
        except Exception:
            pass
        self.destroy()


# =============================================================================
#  ENTRY POINT
# =============================================================================
def main():
    app = MewCoreDolphinGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
