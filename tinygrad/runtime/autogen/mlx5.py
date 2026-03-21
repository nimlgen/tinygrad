# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
__u8: TypeAlias = Annotated[int, ctypes.c_ubyte]
__be16: TypeAlias = Annotated[int, ctypes.c_uint16]
__be32: TypeAlias = Annotated[int, ctypes.c_uint32]
__be64: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class struct_mlx5_cmd_layout(c.Struct):
  SIZE = 64
  type: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  rsvd0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 1]
  inlen: Annotated[Annotated[int, ctypes.c_uint32], 4]
  in_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  _in: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 16]
  out: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 32]
  out_ptr: Annotated[Annotated[int, ctypes.c_uint64], 48]
  outlen: Annotated[Annotated[int, ctypes.c_uint32], 56]
  token: Annotated[Annotated[int, ctypes.c_ubyte], 60]
  sig: Annotated[Annotated[int, ctypes.c_ubyte], 61]
  rsvd1: Annotated[Annotated[int, ctypes.c_ubyte], 62]
  status_own: Annotated[Annotated[int, ctypes.c_ubyte], 63]
@c.record
class struct_mlx5_cmd_prot_block(c.Struct):
  SIZE = 576
  data: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[512]], 0]
  rsvd0: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[48]], 512]
  next: Annotated[Annotated[int, ctypes.c_uint64], 560]
  block_num: Annotated[Annotated[int, ctypes.c_uint32], 568]
  rsvd1: Annotated[Annotated[int, ctypes.c_ubyte], 572]
  token: Annotated[Annotated[int, ctypes.c_ubyte], 573]
  ctrl_sig: Annotated[Annotated[int, ctypes.c_ubyte], 574]
  sig: Annotated[Annotated[int, ctypes.c_ubyte], 575]
@c.record
class struct_mlx5_init_seg(c.Struct):
  SIZE = 512
  fw_rev: Annotated[Annotated[int, ctypes.c_uint32], 0]
  cmdif_rev_fw_sub: Annotated[Annotated[int, ctypes.c_uint32], 4]
  rsvd0: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 8]
  cmdq_addr_h: Annotated[Annotated[int, ctypes.c_uint32], 16]
  cmdq_addr_l_sz: Annotated[Annotated[int, ctypes.c_uint32], 20]
  cmd_dbell: Annotated[Annotated[int, ctypes.c_uint32], 24]
  rsvd1: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[120]], 28]
  initializing: Annotated[Annotated[int, ctypes.c_uint32], 508]
c.init_records()
MLX5_CMD_OP_QUERY_HCA_CAP = 0x100 # type: ignore
MLX5_CMD_OP_QUERY_ADAPTER = 0x101 # type: ignore
MLX5_CMD_OP_INIT_HCA = 0x102 # type: ignore
MLX5_CMD_OP_TEARDOWN_HCA = 0x103 # type: ignore
MLX5_CMD_OP_ENABLE_HCA = 0x104 # type: ignore
MLX5_CMD_OP_DISABLE_HCA = 0x105 # type: ignore
MLX5_CMD_OP_QUERY_PAGES = 0x107 # type: ignore
MLX5_CMD_OP_MANAGE_PAGES = 0x108 # type: ignore
MLX5_CMD_OP_SET_HCA_CAP = 0x109 # type: ignore
MLX5_CMD_OP_QUERY_ISSI = 0x10a # type: ignore
MLX5_CMD_OP_SET_ISSI = 0x10b # type: ignore
MLX5_CMD_OP_SET_DRIVER_VERSION = 0x10d # type: ignore
MLX5_CMD_OP_CREATE_MKEY = 0x200 # type: ignore
MLX5_CMD_OP_QUERY_SPECIAL_CONTEXTS = 0x203 # type: ignore
MLX5_CMD_OP_CREATE_EQ = 0x301 # type: ignore
MLX5_CMD_OP_DESTROY_EQ = 0x302 # type: ignore
MLX5_CMD_OP_CREATE_CQ = 0x400 # type: ignore
MLX5_CMD_OP_DESTROY_CQ = 0x401 # type: ignore
MLX5_CMD_OP_CREATE_QP = 0x500 # type: ignore
MLX5_CMD_OP_DESTROY_QP = 0x501 # type: ignore
MLX5_CMD_OP_RST2INIT_QP = 0x502 # type: ignore
MLX5_CMD_OP_INIT2RTR_QP = 0x503 # type: ignore
MLX5_CMD_OP_RTR2RTS_QP = 0x504 # type: ignore
MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT = 0x754 # type: ignore
MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT = 0x755 # type: ignore
MLX5_CMD_OP_ALLOC_PD = 0x800 # type: ignore
MLX5_CMD_OP_ALLOC_UAR = 0x802 # type: ignore
MLX5_CMD_OP_ACCESS_REG = 0x805 # type: ignore
MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN = 0x816 # type: ignore
MLX5_CMD_STAT_OK = 0x0 # type: ignore
MLX5_CMD_STAT_INT_ERR = 0x1 # type: ignore
MLX5_CMD_STAT_BAD_OP_ERR = 0x2 # type: ignore
MLX5_CMD_STAT_BAD_PARAM_ERR = 0x3 # type: ignore
MLX5_CMD_STAT_BAD_SYS_STATE_ERR = 0x4 # type: ignore
MLX5_CMD_STAT_BAD_RES_ERR = 0x5 # type: ignore
MLX5_CMD_STAT_RES_BUSY = 0x6 # type: ignore
MLX5_CMD_STAT_LIM_ERR = 0x8 # type: ignore
MLX5_CMD_STAT_BAD_RES_STATE_ERR = 0x9 # type: ignore
MLX5_CMD_STAT_NO_RES_ERR = 0xf # type: ignore
MLX5_CMD_STAT_BAD_INP_LEN_ERR = 0x50 # type: ignore
MLX5_CMD_STAT_BAD_OUTP_LEN_ERR = 0x51 # type: ignore
MLX5_CAP_GENERAL = 0x0 # type: ignore
MLX5_CAP_ODP = 0x2 # type: ignore
MLX5_CAP_ATOMIC = 0x3 # type: ignore
MLX5_CAP_ROCE = 0x4 # type: ignore
HCA_CAP_OPMOD_GET_MAX = 0 # type: ignore
HCA_CAP_OPMOD_GET_CUR = 1 # type: ignore
MLX5_PAGES_GIVE = 1 # type: ignore
MLX5_PAGES_TAKE = 2 # type: ignore
MLX5_BOOT_PAGES = 1 # type: ignore
MLX5_INIT_PAGES = 2 # type: ignore
MLX5_REG_HOST_ENDIANNESS = 0x7004 # type: ignore
MLX5_REG_DTOR = 0xC00E # type: ignore
MLX5_PCI_CMD_XPORT = 0x07 # type: ignore
MLX5_CMD_DATA_BLOCK_SIZE = 512 # type: ignore
CMD_OWNER_HW = 0x01 # type: ignore
CAP_GEN_ABS_NATIVE_PORT_NUM = 0x007 # type: ignore
CAP_GEN_HCA_CAP_2 = 0x020 # type: ignore
CAP_GEN_EVENT_ON_VHCA_STATE_ALLOCATED = 0x023 # type: ignore
CAP_GEN_EVENT_ON_VHCA_STATE_ACTIVE = 0x024 # type: ignore
CAP_GEN_EVENT_ON_VHCA_STATE_IN_USE = 0x025 # type: ignore
CAP_GEN_EVENT_ON_VHCA_STATE_TEARDOWN_REQUEST = 0x026 # type: ignore
CAP_GEN_LOG_MAX_QP = 0x09B # type: ignore
CAP_GEN_LOG_MAX_CQ = 0x0DB # type: ignore
CAP_GEN_RELEASE_ALL_PAGES = 0x145 # type: ignore
CAP_GEN_CACHE_LINE_128BYTE = 0x164 # type: ignore
CAP_GEN_NUM_PORTS = 0x1B8 # type: ignore
CAP_GEN_PKEY_TABLE_SIZE = 0x190 # type: ignore
CAP_GEN_PCI_SYNC_FOR_FW_UPDATE_EVENT = 0x1F1 # type: ignore
CAP_GEN_CMDIF_CHECKSUM = 0x210 # type: ignore
CAP_GEN_DCT = 0x21A # type: ignore
CAP_GEN_ROCE = 0x21D # type: ignore
CAP_GEN_ATOMIC = 0x21E # type: ignore
CAP_GEN_ODP = 0x227 # type: ignore
CAP_GEN_MKEY_BY_NAME = 0x266 # type: ignore
CAP_GEN_LOG_MAX_PD = 0x32B # type: ignore
CAP_GEN_PCIE_RESET_USING_HOTRESET = 0x335 # type: ignore
CAP_GEN_PCI_SYNC_FOR_FW_UPDATE_WITH_DRIVER_UNLOAD = 0x336 # type: ignore
CAP_GEN_VHCA_STATE = 0x3EA # type: ignore
CAP_GEN_ROCE_RW_SUPPORTED = 0x3A1 # type: ignore
CAP_GEN_LOG_MAX_CURRENT_UC_LIST = 0x3FB # type: ignore
CAP_GEN_LOG_UAR_PAGE_SZ = 0x490 # type: ignore
CAP_GEN_NUM_VHCA_PORTS = 0x610 # type: ignore
CAP_GEN_SW_OWNER_ID = 0x61E # type: ignore
CAP_GEN_NUM_TOTAL_DYNAMIC_VF_MSIX = 0x708 # type: ignore