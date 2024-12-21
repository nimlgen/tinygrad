# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))





_SCSI_SG_H = 1 # macro
__need_size_t = True # macro
SG_DXFER_NONE = -1 # macro
SG_DXFER_TO_DEV = -2 # macro
SG_DXFER_FROM_DEV = -3 # macro
SG_DXFER_TO_FROM_DEV = -4 # macro
SG_FLAG_DIRECT_IO = 1 # macro
SG_FLAG_LUN_INHIBIT = 2 # macro
SG_FLAG_NO_DXFER = 0x10000 # macro
SG_INFO_OK_MASK = 0x1 # macro
SG_INFO_OK = 0x0 # macro
SG_INFO_CHECK = 0x1 # macro
SG_INFO_DIRECT_IO_MASK = 0x6 # macro
SG_INFO_INDIRECT_IO = 0x0 # macro
SG_INFO_DIRECT_IO = 0x2 # macro
SG_INFO_MIXED_IO = 0x4 # macro
SG_EMULATED_HOST = 0x2203 # macro
SG_SET_TRANSFORM = 0x2204 # macro
SG_GET_TRANSFORM = 0x2205 # macro
SG_SET_RESERVED_SIZE = 0x2275 # macro
SG_GET_RESERVED_SIZE = 0x2272 # macro
SG_GET_SCSI_ID = 0x2276 # macro
SG_SET_FORCE_LOW_DMA = 0x2279 # macro
SG_GET_LOW_DMA = 0x227a # macro
SG_SET_FORCE_PACK_ID = 0x227b # macro
SG_GET_PACK_ID = 0x227c # macro
SG_GET_NUM_WAITING = 0x227d # macro
SG_GET_SG_TABLESIZE = 0x227F # macro
SG_GET_VERSION_NUM = 0x2282 # macro
SG_SCSI_RESET = 0x2284 # macro
SG_SCSI_RESET_NOTHING = 0 # macro
SG_SCSI_RESET_DEVICE = 1 # macro
SG_SCSI_RESET_BUS = 2 # macro
SG_SCSI_RESET_HOST = 3 # macro
SG_IO = 0x2285 # macro
SG_GET_REQUEST_TABLE = 0x2286 # macro
SG_SET_KEEP_ORPHAN = 0x2287 # macro
SG_GET_KEEP_ORPHAN = 0x2288 # macro
SG_SCATTER_SZ = (8*4096) # macro
SG_DEFAULT_RETRIES = 1 # macro
SG_DEF_FORCE_LOW_DMA = 0 # macro
SG_DEF_FORCE_PACK_ID = 0 # macro
SG_DEF_KEEP_ORPHAN = 0 # macro
SG_DEF_RESERVED_SIZE = (8*4096) # macro
SG_MAX_QUEUE = 16 # macro
SG_BIG_BUFF = (8*4096) # macro
SG_MAX_SENSE = 16 # macro
SG_SET_TIMEOUT = 0x2201 # macro
SG_GET_TIMEOUT = 0x2202 # macro
SG_GET_COMMAND_Q = 0x2270 # macro
SG_SET_COMMAND_Q = 0x2271 # macro
SG_SET_DEBUG = 0x227e # macro
SG_NEXT_CMD_LEN = 0x2283 # macro
# SG_DEFAULT_TIMEOUT = (60*HZ) # macro
SG_DEF_COMMAND_Q = 0 # macro
SG_DEF_UNDERRUN_FLAG = 0 # macro
class struct_sg_iovec(Structure):
    pass

struct_sg_iovec._pack_ = 1 # source:False
struct_sg_iovec._fields_ = [
    ('iov_base', ctypes.POINTER(None)),
    ('iov_len', ctypes.c_uint64),
]

sg_iovec_t = struct_sg_iovec
class struct_sg_io_hdr(Structure):
    pass

struct_sg_io_hdr._pack_ = 1 # source:False
struct_sg_io_hdr._fields_ = [
    ('interface_id', ctypes.c_int32),
    ('dxfer_direction', ctypes.c_int32),
    ('cmd_len', ctypes.c_ubyte),
    ('mx_sb_len', ctypes.c_ubyte),
    ('iovec_count', ctypes.c_uint16),
    ('dxfer_len', ctypes.c_uint32),
    ('dxferp', ctypes.POINTER(None)),
    ('cmdp', ctypes.POINTER(ctypes.c_ubyte)),
    ('sbp', ctypes.POINTER(ctypes.c_ubyte)),
    ('timeout', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('pack_id', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('usr_ptr', ctypes.POINTER(None)),
    ('status', ctypes.c_ubyte),
    ('masked_status', ctypes.c_ubyte),
    ('msg_status', ctypes.c_ubyte),
    ('sb_len_wr', ctypes.c_ubyte),
    ('host_status', ctypes.c_uint16),
    ('driver_status', ctypes.c_uint16),
    ('resid', ctypes.c_int32),
    ('duration', ctypes.c_uint32),
    ('info', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

sg_io_hdr_t = struct_sg_io_hdr
class struct_sg_scsi_id(Structure):
    pass

struct_sg_scsi_id._pack_ = 1 # source:False
struct_sg_scsi_id._fields_ = [
    ('host_no', ctypes.c_int32),
    ('channel', ctypes.c_int32),
    ('scsi_id', ctypes.c_int32),
    ('lun', ctypes.c_int32),
    ('scsi_type', ctypes.c_int32),
    ('h_cmd_per_lun', ctypes.c_int16),
    ('d_queue_depth', ctypes.c_int16),
    ('unused', ctypes.c_int32 * 2),
]

class struct_sg_req_info(Structure):
    pass

struct_sg_req_info._pack_ = 1 # source:False
struct_sg_req_info._fields_ = [
    ('req_state', ctypes.c_char),
    ('orphan', ctypes.c_char),
    ('sg_io_owned', ctypes.c_char),
    ('problem', ctypes.c_char),
    ('pack_id', ctypes.c_int32),
    ('usr_ptr', ctypes.POINTER(None)),
    ('duration', ctypes.c_uint32),
    ('unused', ctypes.c_int32),
]

sg_req_info_t = struct_sg_req_info
Sg_io_hdr = struct_sg_io_hdr
class struct_sg_io_vec(Structure):
    pass

Sg_io_vec = struct_sg_io_vec
Sg_scsi_id = struct_sg_scsi_id
Sg_req_info = struct_sg_req_info
class struct_sg_header(Structure):
    pass

struct_sg_header._pack_ = 1 # source:False
struct_sg_header._fields_ = [
    ('pack_len', ctypes.c_int32),
    ('reply_len', ctypes.c_int32),
    ('pack_id', ctypes.c_int32),
    ('result', ctypes.c_int32),
    ('twelve_byte', ctypes.c_uint32, 1),
    ('target_status', ctypes.c_uint32, 5),
    ('host_status', ctypes.c_uint32, 8),
    ('driver_status', ctypes.c_uint32, 8),
    ('other_flags', ctypes.c_uint32, 10),
    ('sense_buffer', ctypes.c_ubyte * 16),
]

__all__ = \
    ['SG_BIG_BUFF', 'SG_DEFAULT_RETRIES', 'SG_DEF_COMMAND_Q',
    'SG_DEF_FORCE_LOW_DMA', 'SG_DEF_FORCE_PACK_ID',
    'SG_DEF_KEEP_ORPHAN', 'SG_DEF_RESERVED_SIZE',
    'SG_DEF_UNDERRUN_FLAG', 'SG_DXFER_FROM_DEV', 'SG_DXFER_NONE',
    'SG_DXFER_TO_DEV', 'SG_DXFER_TO_FROM_DEV', 'SG_EMULATED_HOST',
    'SG_FLAG_DIRECT_IO', 'SG_FLAG_LUN_INHIBIT', 'SG_FLAG_NO_DXFER',
    'SG_GET_COMMAND_Q', 'SG_GET_KEEP_ORPHAN', 'SG_GET_LOW_DMA',
    'SG_GET_NUM_WAITING', 'SG_GET_PACK_ID', 'SG_GET_REQUEST_TABLE',
    'SG_GET_RESERVED_SIZE', 'SG_GET_SCSI_ID', 'SG_GET_SG_TABLESIZE',
    'SG_GET_TIMEOUT', 'SG_GET_TRANSFORM', 'SG_GET_VERSION_NUM',
    'SG_INFO_CHECK', 'SG_INFO_DIRECT_IO', 'SG_INFO_DIRECT_IO_MASK',
    'SG_INFO_INDIRECT_IO', 'SG_INFO_MIXED_IO', 'SG_INFO_OK',
    'SG_INFO_OK_MASK', 'SG_IO', 'SG_MAX_QUEUE', 'SG_MAX_SENSE',
    'SG_NEXT_CMD_LEN', 'SG_SCATTER_SZ', 'SG_SCSI_RESET',
    'SG_SCSI_RESET_BUS', 'SG_SCSI_RESET_DEVICE', 'SG_SCSI_RESET_HOST',
    'SG_SCSI_RESET_NOTHING', 'SG_SET_COMMAND_Q', 'SG_SET_DEBUG',
    'SG_SET_FORCE_LOW_DMA', 'SG_SET_FORCE_PACK_ID',
    'SG_SET_KEEP_ORPHAN', 'SG_SET_RESERVED_SIZE', 'SG_SET_TIMEOUT',
    'SG_SET_TRANSFORM', 'Sg_io_hdr', 'Sg_io_vec', 'Sg_req_info',
    'Sg_scsi_id', '_SCSI_SG_H', '__need_size_t', 'sg_io_hdr_t',
    'sg_iovec_t', 'sg_req_info_t', 'struct_sg_header',
    'struct_sg_io_hdr', 'struct_sg_io_vec', 'struct_sg_iovec',
    'struct_sg_req_info', 'struct_sg_scsi_id']
