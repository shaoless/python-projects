"""
S7 地址解析工具
支持格式:
  M{byte}.{bit}  — 位 (M130.0, M101.0)
  MB{byte}       — 字节
  MW{byte}       — 字 (2 bytes)
  MD{byte}       — 双字 (4 bytes)
  DB{num}.DBX{byte}.{bit}  — DB 位
  DB{num}.DBB{byte}        — DB 字节
  DB{num}.DBW{byte}        — DB 字
  DB{num}.DBD{byte}        — DB 双字
"""

import re
from dataclasses import dataclass
from snap7.type import Area


@dataclass
class S7Address:
    area: int          # Area.MK / Area.DB
    db_num: int        # DB 编号 (M 区 = 0)
    byte_offset: int   # 起始字节
    bit: int | None    # 位偏移 (None = 非位操作)
    size: int          # 字节数 (1/2/4, 位操作为 1)
    type_name: str     # "bool" "byte" "word" "dword" "real"


# 正则: M{byte}.{bit}
_RE_M_BIT = re.compile(r"^M(\d+)\.(\d)$", re.IGNORECASE)
# 正则: MB{byte} MW{byte} MD{byte}
_RE_M_BYTE = re.compile(r"^M([BWD])(\d+)$", re.IGNORECASE)
# 正则: DB{num}.DBX{byte}.{bit}
_RE_DB_BIT = re.compile(r"^DB(\d+)\.DBX(\d+)\.(\d)$", re.IGNORECASE)
# 正则: DB{num}.DBB{byte} DBW DBD
_RE_DB_BYTE = re.compile(r"^DB(\d+)\.DB([BWD])(\d+)$", re.IGNORECASE)


def parse(address: str) -> S7Address:
    """解析 S7 风格地址字符串, 返回 S7Address."""

    # --- M 区位 ---
    m = _RE_M_BIT.match(address)
    if m:
        byte_off = int(m.group(1))
        bit = int(m.group(2))
        if bit > 7:
            raise ValueError(f"位号超出范围 (0-7): {bit}")
        return S7Address(Area.MK, 0, byte_off, bit, 1, "bool")

    m = _RE_M_BYTE.match(address)
    if m:
        suffix = m.group(1).upper()
        byte_off = int(m.group(2))
        size_map = {"B": 1, "W": 2, "D": 4}
        type_map = {"B": "byte", "W": "word", "D": "dword"}
        return S7Address(Area.MK, 0, byte_off, None,
                         size_map[suffix], type_map[suffix])

    # --- DB 区位 ---
    m = _RE_DB_BIT.match(address)
    if m:
        db_num = int(m.group(1))
        byte_off = int(m.group(2))
        bit = int(m.group(3))
        if bit > 7:
            raise ValueError(f"位号超出范围 (0-7): {bit}")
        return S7Address(Area.DB, db_num, byte_off, bit, 1, "bool")

    m = _RE_DB_BYTE.match(address)
    if m:
        db_num = int(m.group(1))
        suffix = m.group(2).upper()
        byte_off = int(m.group(3))
        size_map = {"B": 1, "W": 2, "D": 4}
        type_map = {"B": "byte", "W": "word", "D": "dword"}
        return S7Address(Area.DB, db_num, byte_off, None,
                         size_map[suffix], type_map[suffix])

    raise ValueError(f"无法识别的地址格式: {address}")
