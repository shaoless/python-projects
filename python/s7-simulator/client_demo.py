"""
S7 客户端测试 — 连接模拟 PLC 读写数据.
支持 S7 风格地址: M130.0, MD102, MB100, MW104 等.
默认端口 102, 可用 --port 指定.
"""

import argparse
import struct
import time
from snap7.client import Client
from snap7.type import Area
from s7_address import parse, S7Address


# ── 读写函数 ──────────────────────────────────────────────

def read_s7(client: Client, address: str):
    """读取 S7 地址. 返回 bool/int/float/bytes 根据类型自动判断."""
    addr = parse(address)
    return _read_addr(client, addr)


def write_s7(client: Client, address: str, value):
    """写入 S7 地址. 自动判断类型并打包."""
    addr = parse(address)
    _write_addr(client, addr, value)


def _read_addr(client: Client, addr: S7Address):
    """内部: 按解析后的地址读取."""
    if addr.area == Area.DB:
        data = client.db_read(addr.db_num, start=addr.byte_offset, size=addr.size)
    else:
        data = client.read_area(addr.area, addr.db_num, addr.byte_offset, addr.size)

    if addr.bit is not None:
        return bool(data[0] & (1 << addr.bit))

    if addr.size == 1:
        return data[0]          # byte → int
    elif addr.size == 2:
        return struct.unpack(">H", data)[0]   # word → int
    elif addr.size == 4:
        return struct.unpack(">f", data)[0]   # dword → float (默认 real)
    return data


def _write_addr(client: Client, addr: S7Address, value):
    """内部: 按解析后的地址写入."""
    if addr.bit is not None:
        # 位写入: 读-改-写
        if addr.area == Area.DB:
            old = client.db_read(addr.db_num, start=addr.byte_offset, size=1)
        else:
            old = client.read_area(addr.area, addr.db_num, addr.byte_offset, 1)

        new_byte = bytearray(old)
        if value:
            new_byte[0] |= (1 << addr.bit)
        else:
            new_byte[0] &= ~(1 << addr.bit)

        if addr.area == Area.DB:
            client.db_write(addr.db_num, start=addr.byte_offset, data=new_byte)
        else:
            client.write_area(addr.area, addr.db_num, addr.byte_offset, new_byte)
        return

    # 字节 / 字 / 双字
    if addr.size == 1:
        data = bytearray([value & 0xFF])
    elif addr.size == 2:
        data = bytearray(struct.pack(">H", value))
    elif addr.size == 4:
        data = bytearray(struct.pack(">f", float(value)))
    else:
        raise ValueError(f"不支持的大小: {addr.size}")

    if addr.area == Area.DB:
        client.db_write(addr.db_num, start=addr.byte_offset, data=data)
    else:
        client.write_area(addr.area, addr.db_num, addr.byte_offset, data)


# ── 主程序 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="S7 PLC Client Demo")
    parser.add_argument("--port", type=int, default=102, help="TCP port (default: 102)")
    args = parser.parse_args()

    client = Client()

    print(f"连接 S7-1200 Simulator (127.0.0.1:{args.port})...")
    try:
        client.connect("127.0.0.1", rack=0, slot=1, tcp_port=args.port)
    except Exception as e:
        print(f"连接失败: {e}")
        print("请先启动 server.py")
        return

    print("已连接\n")

    # ── 1. 读 DB1 ──
    print("=== 读取 DB1 ===")
    data = client.db_read(1, start=0, size=256)
    magic, ver, temp, hum, count = struct.unpack_from(">HHffi", data, 0)
    print(f"  Magic=0x{magic:04X}  Ver={ver}  Temp={temp:.1f}°C  Humidity={hum:.1f}%  Count={count}")

    # ── 2. DB1 温度 → DB2 报警联锁 ──
    print("\n=== 写入 DB1 温度 = 55.0°C (触发高温报警) ===")
    client.db_write(1, start=4, data=bytearray(struct.pack(">f", 55.0)))
    time.sleep(0.3)

    data2 = client.db_read(2, start=0, size=8)
    state, alarm, sp = struct.unpack_from(">HHf", data2, 0)
    alarm_text = {0: "正常", 1: "高温报警!", 2: "低温报警!"}
    print(f"  DB2 报警 = {alarm} ({alarm_text.get(alarm, '未知')})")

    print("\n=== 恢复 = 36.6°C ===")
    client.db_write(1, start=4, data=bytearray(struct.pack(">f", 36.6)))
    time.sleep(0.3)
    data2 = client.db_read(2, start=0, size=8)
    alarm = struct.unpack_from(">H", data2, 2)[0]
    print(f"  DB2 报警 = {alarm} ({alarm_text.get(alarm, '未知')})")

    # ── 3. S7 地址格式演示 ──
    print("\n=== S7 地址格式读写 ===")

    # M130.0 — 位写入/读取
    print("\n--- M130.0 (位) ---")
    write_s7(client, "M130.0", True)
    val = read_s7(client, "M130.0")
    print(f"  写入 True → 读取 M130.0 = {val}")

    write_s7(client, "M130.0", False)
    val = read_s7(client, "M130.0")
    print(f"  写入 False → 读取 M130.0 = {val}")

    # M130.1 — 同字节不同位
    print("\n--- M130.1 (位, 与 M130.0 同字节) ---")
    write_s7(client, "M130.1", True)
    v0 = read_s7(client, "M130.0")
    v1 = read_s7(client, "M130.1")
    print(f"  M130.0={v0}, M130.1={v1}  (互不影响)")

    # M101.0
    print("\n--- M101.0 (位) ---")
    write_s7(client, "M101.0", True)
    val = read_s7(client, "M101.0")
    print(f"  写入 True → 读取 M101.0 = {val}")

    # MD102 — 双字浮点
    print("\n--- MD102 (双字 / Real) ---")
    write_s7(client, "MD102", 3.14)
    val = read_s7(client, "MD102")
    print(f"  写入 3.14 → 读取 MD102 = {val}")

    # MW100 — 字
    print("\n--- MW100 (字) ---")
    write_s7(client, "MW100", 0x1234)
    val = read_s7(client, "MW100")
    print(f"  写入 0x1234 → 读取 MW100 = 0x{val:04X}")

    # MB100 — 字节
    print("\n--- MB104 (字节) ---")
    write_s7(client, "MB104", 0xAB)
    val = read_s7(client, "MB104")
    print(f"  写入 0xAB → 读取 MB104 = 0x{val:02X}")

    # ── 4. I/O 映射 ──
    print("\n=== I/O 映射测试 ===")
    client.write_area(Area.PE, 0, 0, bytearray([0xAA, 0x55]))
    time.sleep(0.2)
    q = client.read_area(Area.PA, 0, 0, 2)
    print(f"  I=[0xAA, 0x55] → Q={q.hex(' ')} (扫描周期应镜像)")

    client.disconnect()
    print("\n测试完成.")


if __name__ == "__main__":
    main()
