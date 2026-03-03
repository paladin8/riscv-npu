# Phase 10: GDB Remote Stub

## Goal

Implement the GDB Remote Serial Protocol (RSP) over TCP so that `riscv64-unknown-elf-gdb` can connect to the emulator and debug firmware using standard GDB commands: step, continue, breakpoints, register/memory inspection.

## Deliverables

- `gdb/server.py`: TCP server that speaks GDB RSP
- `gdb/protocol.py`: Packet framing, checksum, ack/nack
- `gdb/commands.py`: RSP command handlers mapped to CPU operations
- CLI `gdb` subcommand: `riscv_npu gdb <elf> [--port 1234]`
- Unit tests for protocol parsing and command handling
- Integration test connecting a mock GDB client

## Design Decisions

1. **Standard RSP over TCP.** The GDB Remote Serial Protocol is text-based with `$packet-data#checksum` framing. We implement the subset that GDB actually uses when debugging a RISC-V target. No threading — the stub is single-threaded and synchronous: wait for GDB packet, process it, send reply, repeat.

2. **Register layout matches `riscv:rv32` target.** GDB's built-in RISC-V target description expects registers in this order for rv32:
   - Registers 0–31: x0–x31 (4 bytes each, little-endian hex)
   - Register 32: pc (4 bytes)
   Total for the `g` packet: 33 × 4 = 132 bytes → 264 hex chars.

   Float registers (f0–f31) and fcsr are exposed via `qXfer:features:read` target description XML so GDB knows they exist. This lets `info float` and `print $f0` work. Without the XML, GDB won't ask for FP registers at all.

3. **Target description XML for FPU registers.** Serve a minimal target description via `qXfer:features:read:target.xml` that declares:
   - `<feature name="org.gnu.gdb.riscv.cpu">` — x0–x31 + pc (mandatory)
   - `<feature name="org.gnu.gdb.riscv.fpu">` — f0–f31 + fflags + frm + fcsr
   This uses GDB's standard RISC-V feature names, so register display and `info registers` work correctly with the stock `riscv64-unknown-elf-gdb`.

4. **Software breakpoints only.** GDB sends `Z0,addr,kind` (insert) and `z0,addr,kind` (remove) for software breakpoints. We maintain a `set[int]` of breakpoint addresses. On `continue`, the CPU runs until `cpu.pc in breakpoints` or `cpu.halted`. No hardware breakpoints (Z1) or watchpoints (Z2/Z3/Z4) — reply empty packet to signal unsupported.

5. **Stop reply protocol.** After step/continue, send `S05` (SIGTRAP) when a breakpoint is hit or step completes. Send `W00` when the program exits normally. Send `S0B` (SIGSEGV) on memory access errors.

6. **No-ack mode.** Support `QStartNoAckMode` — once enabled, skip `+`/`-` ack bytes. This is what modern GDB uses and simplifies the protocol.

7. **Ctrl+C (interrupt).** GDB sends byte `0x03` to interrupt a running target. During `continue`, we check for incoming data between steps (using `select()` with zero timeout). When `0x03` arrives, we stop and send `S02` (SIGINT).

8. **Module location: `src/riscv_npu/gdb/`.** Separate from TUI — the GDB stub is an independent debug frontend. Both the TUI debugger and GDB stub use `cpu.step()` and breakpoint sets, but they don't share code because the control flow is fundamentally different (TUI is interactive readline, GDB is packet-driven).

## Supported RSP Commands

### Required (GDB won't connect without these)

| Packet          | Description                    | Response                                   |
|-----------------|--------------------------------|--------------------------------------------|
| `?`             | Halt reason                    | `S05` (stopped, SIGTRAP)                   |
| `g`             | Read all registers             | Hex string: x0–x31 + pc (264 hex chars)    |
| `G XX...`       | Write all registers            | `OK`                                       |
| `p n`           | Read register n                | 8 hex chars (LE)                           |
| `P n=XX...`     | Write register n               | `OK`                                       |
| `m addr,length` | Read memory                    | Hex bytes                                  |
| `M addr,len:XX` | Write memory                   | `OK`                                       |
| `c`             | Continue                       | Stop reply when halted (`S05`/`W00`)       |
| `s`             | Single step                    | `S05`                                      |
| `Z0,addr,kind`  | Insert breakpoint              | `OK`                                       |
| `z0,addr,kind`  | Remove breakpoint              | `OK`                                       |
| `k`             | Kill                           | (close connection)                         |

### Query packets

| Packet                                | Response                                          |
|---------------------------------------|---------------------------------------------------|
| `qSupported`                          | `PacketSize=4096;QStartNoAckMode+;qXfer:features:read+` |
| `QStartNoAckMode`                     | `OK` (disable ack)                                |
| `qXfer:features:read:target.xml:o,l`  | Target description XML                            |
| `qAttached`                           | `1` (attached to existing process)                |
| `qC`                                  | `QC1` (thread 1)                                  |
| `Hg0` / `Hc0`                        | `OK` (set thread — single-threaded, always OK)    |

### Unsupported (reply empty)

All other packets get an empty response (`$#00`), which tells GDB the feature is not supported. This includes `Z1`–`Z4` (hardware break/watchpoints), `vCont`, `qSymbol`, etc.

## Deliverables List (ordered, dependency-aware)

1. **D1: `gdb/protocol.py`** — RSP packet framing (parse, build, checksum)
2. **D2: `gdb/commands.py`** — Command dispatch: map RSP packets to CPU operations
3. **D3: `gdb/target_xml.py`** — Target description XML for RISC-V registers
4. **D4: `gdb/server.py`** — TCP server, connection handling, main loop
5. **D5: CLI integration** — `gdb` subcommand in cli.py
6. **D6: Unit tests** — Protocol parsing, command handling, XML
7. **D7: Integration test** — Mock GDB client connects and exercises step/continue/breakpoints

## Implementation Details

### D1: `gdb/protocol.py`

```python
def parse_packet(data: bytes) -> str | None:
    """Extract packet body from '$body#xx' framing.

    Returns the body string if checksum is valid, None otherwise.
    """

def build_packet(body: str) -> bytes:
    """Build a framed RSP packet: $body#xx."""

def checksum(data: str) -> int:
    """Compute RSP checksum: sum of ASCII values mod 256."""

def hex_encode(data: bytes) -> str:
    """Encode raw bytes as hex string (e.g. b'\\x80' -> '80')."""

def hex_decode(hex_str: str) -> bytes:
    """Decode hex string to raw bytes."""

def encode_reg32(value: int) -> str:
    """Encode a 32-bit value as 8 little-endian hex chars."""

def decode_reg32(hex_str: str) -> int:
    """Decode 8 little-endian hex chars to a 32-bit integer."""
```

Register encoding is **little-endian**: value `0x80000000` becomes `"00000080"`. This matches the wire format GDB expects for little-endian RISC-V targets.

### D2: `gdb/commands.py`

```python
@dataclass
class GdbState:
    """State for the GDB debug session."""
    cpu: CPU
    breakpoints: set[int] = field(default_factory=set)
    no_ack: bool = False
    target_xml: str = ""

def handle_packet(state: GdbState, packet: str) -> str | None:
    """Dispatch an RSP packet and return the response body.

    Returns None for packets that require no immediate response
    (e.g., continue — the response comes later when the CPU stops).
    """

def handle_question(state: GdbState) -> str:
    """'?' — return halt reason. Always S05 (SIGTRAP) on attach."""

def handle_read_regs(state: GdbState) -> str:
    """'g' — read all registers as concatenated LE hex."""

def handle_write_regs(state: GdbState, data: str) -> str:
    """'G' — write all registers from concatenated LE hex."""

def handle_read_reg(state: GdbState, packet: str) -> str:
    """'p n' — read single register n."""

def handle_write_reg(state: GdbState, packet: str) -> str:
    """'P n=val' — write single register n."""

def handle_read_mem(state: GdbState, packet: str) -> str:
    """'m addr,length' — read memory as hex."""

def handle_write_mem(state: GdbState, packet: str) -> str:
    """'M addr,length:data' — write hex data to memory."""

def handle_step(state: GdbState) -> str:
    """'s' — single step, return stop reply."""

def handle_continue(state: GdbState, sock: socket.socket) -> str:
    """'c' — run until breakpoint/halt/interrupt, return stop reply.

    Checks sock for incoming 0x03 (Ctrl+C) between steps using select().
    """

def handle_insert_bp(state: GdbState, packet: str) -> str:
    """'Z0,addr,kind' — insert software breakpoint."""

def handle_remove_bp(state: GdbState, packet: str) -> str:
    """'z0,addr,kind' — remove software breakpoint."""

def handle_query(state: GdbState, packet: str) -> str:
    """Route 'q...' and 'Q...' query packets."""
```

**Register numbering** (for `p`/`P` packets):

| Index   | Register  | Source                                  |
|---------|-----------|-----------------------------------------|
| 0–31    | x0–x31    | `cpu.registers.read(n)`                 |
| 32      | pc        | `cpu.pc`                                |
| 33–64   | f0–f31    | `cpu.fpu_state.fregs.read_bits(n - 33)` |
| 65      | fflags    | `cpu.fpu_state.fflags`                  |
| 66      | frm       | `cpu.fpu_state.frm`                     |
| 67      | fcsr      | `cpu.fpu_state.fcsr`                    |

The `g` packet only includes registers 0–32 (GPR + PC). FP registers are accessed via `p`/`P` individual reads — GDB uses the target description to know they exist and queries them separately.

### D3: `gdb/target_xml.py`

```python
TARGET_XML = """\
<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target version="1.0">
  <architecture>riscv:rv32</architecture>
  <feature name="org.gnu.gdb.riscv.cpu">
    <!-- x0-x31 -->
    <reg name="zero" bitsize="32" regnum="0" type="int" />
    <reg name="ra"   bitsize="32" regnum="1" type="code_ptr" />
    ...
    <reg name="t6"   bitsize="32" regnum="31" type="int" />
    <reg name="pc"   bitsize="32" regnum="32" type="code_ptr" />
  </feature>
  <feature name="org.gnu.gdb.riscv.fpu">
    <reg name="ft0"   bitsize="32" regnum="33" type="ieee_single" />
    ...
    <reg name="ft11"  bitsize="32" regnum="64" type="ieee_single" />
    <reg name="fflags" bitsize="32" regnum="65" type="int" />
    <reg name="frm"    bitsize="32" regnum="66" type="int" />
    <reg name="fcsr"   bitsize="32" regnum="67" type="int" />
  </feature>
</target>
"""
```

Standard RISC-V ABI register names so GDB displays `ra`, `sp`, `a0`, etc. instead of `x1`, `x2`, `x10`.

### D4: `gdb/server.py`

```python
def serve(cpu: CPU, port: int = 1234) -> None:
    """Start the GDB RSP server and block until the session ends.

    Opens a TCP socket, waits for GDB to connect, then enters the
    packet-processing loop. Handles one connection at a time.

    Args:
        cpu: The CPU instance to debug.
        port: TCP port to listen on.
    """
```

**Main loop pseudocode:**

```
sock = listen on port
print "Listening on :port, waiting for GDB..."
conn = accept()
print "GDB connected."
state = GdbState(cpu=cpu, target_xml=TARGET_XML)
buffer = b""

while True:
    # Read data from GDB
    buffer += conn.recv(4096)

    # Check for interrupt (0x03)
    if 0x03 in buffer and state is running:
        handle interrupt

    # Extract and process complete packets
    while buffer contains '$..#xx':
        packet = parse_packet(...)
        if not state.no_ack:
            conn.send(b"+")  # ACK
        response = handle_packet(state, packet)
        if response is not None:
            conn.send(build_packet(response))

    # 'c' command: run CPU, checking for interrupt between steps
    if state needs to continue:
        reply = handle_continue(state, conn)
        conn.send(build_packet(reply))
```

**Socket options:** `SO_REUSEADDR` so the port is immediately reusable after a session ends (important during development/testing).

**Graceful shutdown:** On `k` (kill) or connection close, print the final CPU state (cycle count, a0/a1) to stderr and exit.

### D5: CLI integration

Add to `cli.py`:

```python
gdb_parser = sub.add_parser("gdb", help="Start GDB remote debug server")
gdb_parser.add_argument("binary", help="Path to ELF file to debug")
gdb_parser.add_argument(
    "--port", type=int, default=1234,
    help="TCP port to listen on (default: 1234)",
)
gdb_parser.add_argument(
    "--write", action="append", type=_parse_write_arg, default=[],
    metavar="SYMBOL:FILE",
    help="Write FILE contents to ELF symbol address (repeatable)",
)
```

In command dispatch:

```python
elif args.command == "gdb":
    from .gdb import serve
    # (same ELF loading / --write logic as run_binary, but call serve() instead of cpu.run())
```

### D6: Unit tests

**`tests/gdb/test_protocol.py`:**

| Test                                   | Description                                    |
|----------------------------------------|------------------------------------------------|
| `test_checksum_empty`                  | Empty string checksum is 0                     |
| `test_checksum_known`                  | Known packet body checksum matches             |
| `test_build_packet`                    | Returns `$body#xx` with correct checksum       |
| `test_parse_packet_valid`              | Extracts body from valid packet                |
| `test_parse_packet_bad_checksum`       | Returns None for wrong checksum                |
| `test_parse_packet_no_framing`         | Returns None for data without `$..#xx`         |
| `test_hex_encode_decode_roundtrip`     | Encode then decode returns original bytes      |
| `test_encode_reg32_little_endian`      | `0x80000000` encodes as `"00000080"`           |
| `test_decode_reg32_little_endian`      | `"00000080"` decodes as `0x80000000`           |
| `test_encode_decode_reg32_roundtrip`   | Encode then decode returns original value      |

**`tests/gdb/test_commands.py`:**

| Test                                    | Description                                              |
|-----------------------------------------|----------------------------------------------------------|
| `test_halt_reason`                      | `?` returns `S05`                                        |
| `test_read_regs_initial`               | `g` returns 264 hex chars, all zero except PC            |
| `test_read_reg_pc`                      | `p20` returns PC value (reg 32 = 0x20 hex)               |
| `test_read_reg_gpr`                     | `p01` returns x1 value                                   |
| `test_read_reg_fpr`                     | `p21` returns f0 value (reg 33 = 0x21 hex)               |
| `test_write_reg`                        | `P01=78563412` sets x1 to `0x12345678`                   |
| `test_write_reg_pc`                     | `P20=00000180` sets PC to `0x80010000`                   |
| `test_read_mem`                         | `m80000000,4` returns 4 bytes from RAM                   |
| `test_read_mem_unmapped`               | `m00000000,4` returns `E01` error                        |
| `test_write_mem`                        | `M80000000,4:deadbeef` writes to RAM                     |
| `test_step`                             | `s` advances PC by 4 (on a NOP), returns `S05`           |
| `test_step_halted`                      | `s` when halted returns `W00`                            |
| `test_continue_to_breakpoint`          | `c` with breakpoint set stops at breakpoint              |
| `test_continue_to_halt`               | `c` with no breakpoints runs to halt, returns `W00`      |
| `test_insert_breakpoint`              | `Z0,80000010,4` adds to breakpoint set                   |
| `test_remove_breakpoint`              | `z0,80000010,4` removes from breakpoint set              |
| `test_unsupported_returns_empty`      | `Z1,...` (hardware bp) returns empty                     |
| `test_query_supported`                | `qSupported` returns feature list                        |
| `test_no_ack_mode`                    | `QStartNoAckMode` sets `state.no_ack = True`            |
| `test_query_attached`                 | `qAttached` returns `1`                                  |
| `test_query_current_thread`           | `qC` returns `QC1`                                       |
| `test_set_thread`                     | `Hg0` returns `OK`                                       |
| `test_xfer_target_xml`               | `qXfer:features:read:target.xml:0,fff` returns XML chunk |

**`tests/gdb/test_target_xml.py`:**

| Test                               | Description                                    |
|------------------------------------|------------------------------------------------|
| `test_xml_is_well_formed`          | Parses without errors (`xml.etree.ElementTree`) |
| `test_xml_has_architecture`        | Contains `<architecture>riscv:rv32</architecture>` |
| `test_xml_has_cpu_feature`         | Has `org.gnu.gdb.riscv.cpu` feature            |
| `test_xml_has_fpu_feature`         | Has `org.gnu.gdb.riscv.fpu` feature            |
| `test_xml_gpr_count`              | CPU feature has 33 registers (x0–x31 + pc)     |
| `test_xml_fpr_count`              | FPU feature has 35 registers (f0–f31 + 3 CSRs) |
| `test_xml_pc_is_code_ptr`         | PC register has `type="code_ptr"`               |
| `test_xml_regnums_contiguous`     | Regnums go 0–67 without gaps                    |

### D7: Integration test

**`tests/gdb/test_integration.py`:**

Uses a mock GDB client (raw TCP socket) to connect to the server in a background thread and exercise a real debug session:

```python
def test_gdb_connect_and_step():
    """Start server in thread, connect, step, read regs, disconnect."""
    # Load a simple firmware (fibonacci.elf or a hand-assembled NOP sled)
    # Start serve() in a daemon thread
    # Connect via socket
    # Send: qSupported, ?, g, s, g, k
    # Verify register values change after step
    # Verify server thread exits cleanly

def test_gdb_breakpoint_hit():
    """Set breakpoint, continue, verify stop at breakpoint address."""

def test_gdb_memory_read_write():
    """Write to memory via M packet, read back via m packet, verify match."""

def test_gdb_continue_to_exit():
    """Continue a short program to completion, verify W00 reply."""
```

These tests run the server in a background thread with a short timeout to prevent hangs.

## Usage Example

Terminal 1 — start the emulator:
```bash
uv run python -m riscv_npu gdb firmware/hello/hello.elf --port 1234
# Listening on :1234, waiting for GDB...
# GDB connected.
```

Terminal 2 — connect GDB:
```bash
riscv64-unknown-elf-gdb firmware/hello/hello.elf
(gdb) target remote :1234
Remote debugging using :1234
0x80000000 in _start ()
(gdb) info registers
(gdb) break main
(gdb) continue
Breakpoint 1, main () at main.c:5
(gdb) step
(gdb) print $a0
(gdb) info float
(gdb) x/16xw 0x80000000
(gdb) kill
```

## Acceptance Criteria

1. `uv run pytest tests/gdb/ -v` — all tests pass
2. `uv run pytest` — all 960+ existing tests still pass
3. `uv run python -m riscv_npu gdb firmware/hello/hello.elf` — server starts, prints listening message
4. `riscv64-unknown-elf-gdb` connects, `info registers` shows correct values, `step`/`continue`/`break` work
5. `info float` shows f0–f31 and fcsr
6. Ctrl+C in GDB interrupts a running `continue`
7. Server exits cleanly on `kill` or GDB disconnect
8. No new external dependencies (stdlib `socket` and `select` only)
