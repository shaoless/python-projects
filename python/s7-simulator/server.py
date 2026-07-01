"""
S7-1200 PLC Simulator Server
Simulates a Siemens S7-1200 PLC via python-snap7 server.
Rack=0, Slot=1.  Default port=102, override with --port.
"""

import argparse
import logging
import struct
import sys
import time
from snap7.server import Server, ServerState
from snap7.type import SrvArea, SrvEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("s7-sim")

# Event codes
EVT_READ = 0
EVT_WRITE = 1


class S71200Simulator:
    """Simulates an S7-1200 PLC with DB1, DB2, I, Q, M areas."""

    def __init__(self):
        # 创建存储区
        self.db1 = bytearray(256)
        self.db2 = bytearray(256)
        self.input_area = bytearray(64)
        self.output_area = bytearray(64)
        self.marker_area = bytearray(256)

        self._init_defaults()

        # 创建服务器
        self.server = Server(log=False)

        # 注册各区 (SrvArea: PE=输入 PA=输出 MK=标志 DB=数据块)
        self.server.register_area(SrvArea.DB, 1, self.db1)
        self.server.register_area(SrvArea.DB, 2, self.db2)
        self.server.register_area(SrvArea.PE, 0, self.input_area)
        self.server.register_area(SrvArea.PA, 0, self.output_area)
        self.server.register_area(SrvArea.MK, 0, self.marker_area)

        # 事件回调 (仅日志记录)
        self.server.set_events_callback(self._on_event)

        self.running = True

    def _init_defaults(self):
        """初始化模拟数据."""
        struct.pack_into(">H", self.db1, 0, 0x5347)   # bytes 0-1: magic
        struct.pack_into(">H", self.db1, 2, 0x0700)   # bytes 2-3: version
        struct.pack_into(">f", self.db1, 4, 36.6)     # bytes 4-7: 温度 °C
        struct.pack_into(">f", self.db1, 8, 65.0)     # bytes 8-11: 湿度 %
        struct.pack_into(">i", self.db1, 12, 0)       # bytes 12-15: 计数器

        struct.pack_into(">H", self.db2, 0, 0)        # 运行状态
        struct.pack_into(">H", self.db2, 2, 0)        # 报警码
        struct.pack_into(">f", self.db2, 4, 0.0)      # 设定值

        self.marker_area[0] = 0x01   # M0.0 = True
        self.marker_area[1] = 0x00   # M1.0 = False

    def _on_event(self, event: SrvEvent):
        """S7 event callback (log only)."""
        code = event.EvtCode
        sender = event.EvtSender
        if code == EVT_READ:
            log.info("Read  from %s (param=%d,%d,%d)",
                     sender, event.EvtParam1, event.EvtParam2, event.EvtParam3)
        elif code == EVT_WRITE:
            log.info("Write from %s (param=%d,%d,%d)",
                     sender, event.EvtParam1, event.EvtParam2, event.EvtParam3)

    def _sync_db1_to_db2(self):
        """模拟联锁: DB1 温度 → DB2 报警码."""
        temp = struct.unpack_from(">f", self.db1, 4)[0]
        if temp > 50.0:
            struct.pack_into(">H", self.db2, 2, 1)   # 高温报警
        elif temp < 5.0:
            struct.pack_into(">H", self.db2, 2, 2)   # 低温报警
        else:
            struct.pack_into(">H", self.db2, 2, 0)   # 正常

    def run(self, port=102):
        """主循环."""
        self.server.start(tcp_port=port)

        log.info("=" * 50)
        log.info("S7-1200 Simulator 已启动")
        log.info("  地址: 127.0.0.1:%d  Rack=0  Slot=1", port)
        log.info("  DB1(256B) DB2(256B) I(64B) Q(64B) M(256B)")
        log.info("=" * 50)

        try:
            while self.running:
                self._scan_cycle()
                time.sleep(0.1)
        except KeyboardInterrupt:
            log.info("正在停止...")
        finally:
            self.server.stop()
            self.server.destroy()
            log.info("S7-1200 Simulator 已停止")

    def _scan_cycle(self):
        """PLC scan cycle (100ms)."""
        # 计数器自增
        count = struct.unpack_from(">i", self.db1, 12)[0]
        struct.pack_into(">i", self.db1, 12, count + 1)

        # DB1 温度 → DB2 报警联锁
        self._sync_db1_to_db2()

        # I → Q 镜像
        self.output_area[:] = self.input_area[:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S7-1200 PLC Simulator")
    parser.add_argument("--port", type=int, default=102, help="TCP port to listen on (default: 102)")
    args = parser.parse_args()

    sim = S71200Simulator()
    sim.run(port=args.port)
