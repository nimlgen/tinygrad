# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/opt/rocm/include']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

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



_libraries = {}
_libraries['libhsakmt.so'] = ctypes.CDLL('/home/nimlgen/amd/ROCT-Thunk-Interface/build/libhsakmt.so')
# _libraries['libhsakmt.so'] = ctypes.CDLL('/home/nimlgen/amd/ROCR-Runtime/src/build/libhsa-runtime64.so')
class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries['FIXME_STUB'] = FunctionFactoryStub() #  ctypes.CDLL('FIXME_STUB')


HSAuint8 = ctypes.c_ubyte
HSAint8 = ctypes.c_byte
HSAuint16 = ctypes.c_uint16
HSAint16 = ctypes.c_int16
HSAuint32 = ctypes.c_uint32
HSAint32 = ctypes.c_int32
HSAint64 = ctypes.c_int64
HSAuint64 = ctypes.c_uint64
HSA_HANDLE = ctypes.POINTER(None)
HSA_QUEUEID = ctypes.c_uint64

# values for enumeration '_HSAKMT_STATUS'
_HSAKMT_STATUS__enumvalues = {
    0: 'HSAKMT_STATUS_SUCCESS',
    1: 'HSAKMT_STATUS_ERROR',
    2: 'HSAKMT_STATUS_DRIVER_MISMATCH',
    3: 'HSAKMT_STATUS_INVALID_PARAMETER',
    4: 'HSAKMT_STATUS_INVALID_HANDLE',
    5: 'HSAKMT_STATUS_INVALID_NODE_UNIT',
    6: 'HSAKMT_STATUS_NO_MEMORY',
    7: 'HSAKMT_STATUS_BUFFER_TOO_SMALL',
    10: 'HSAKMT_STATUS_NOT_IMPLEMENTED',
    11: 'HSAKMT_STATUS_NOT_SUPPORTED',
    12: 'HSAKMT_STATUS_UNAVAILABLE',
    13: 'HSAKMT_STATUS_OUT_OF_RESOURCES',
    20: 'HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED',
    21: 'HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR',
    22: 'HSAKMT_STATUS_KERNEL_ALREADY_OPENED',
    23: 'HSAKMT_STATUS_HSAMMU_UNAVAILABLE',
    30: 'HSAKMT_STATUS_WAIT_FAILURE',
    31: 'HSAKMT_STATUS_WAIT_TIMEOUT',
    35: 'HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED',
    36: 'HSAKMT_STATUS_MEMORY_NOT_REGISTERED',
    37: 'HSAKMT_STATUS_MEMORY_ALIGNMENT',
}
HSAKMT_STATUS_SUCCESS = 0
HSAKMT_STATUS_ERROR = 1
HSAKMT_STATUS_DRIVER_MISMATCH = 2
HSAKMT_STATUS_INVALID_PARAMETER = 3
HSAKMT_STATUS_INVALID_HANDLE = 4
HSAKMT_STATUS_INVALID_NODE_UNIT = 5
HSAKMT_STATUS_NO_MEMORY = 6
HSAKMT_STATUS_BUFFER_TOO_SMALL = 7
HSAKMT_STATUS_NOT_IMPLEMENTED = 10
HSAKMT_STATUS_NOT_SUPPORTED = 11
HSAKMT_STATUS_UNAVAILABLE = 12
HSAKMT_STATUS_OUT_OF_RESOURCES = 13
HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED = 20
HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR = 21
HSAKMT_STATUS_KERNEL_ALREADY_OPENED = 22
HSAKMT_STATUS_HSAMMU_UNAVAILABLE = 23
HSAKMT_STATUS_WAIT_FAILURE = 30
HSAKMT_STATUS_WAIT_TIMEOUT = 31
HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED = 35
HSAKMT_STATUS_MEMORY_NOT_REGISTERED = 36
HSAKMT_STATUS_MEMORY_ALIGNMENT = 37
_HSAKMT_STATUS = ctypes.c_uint32 # enum
HSAKMT_STATUS = _HSAKMT_STATUS
HSAKMT_STATUS__enumvalues = _HSAKMT_STATUS__enumvalues
class struct__HsaVersionInfo(Structure):
    pass

struct__HsaVersionInfo._pack_ = 1 # source:False
struct__HsaVersionInfo._fields_ = [
    ('KernelInterfaceMajorVersion', ctypes.c_uint32),
    ('KernelInterfaceMinorVersion', ctypes.c_uint32),
]

HsaVersionInfo = struct__HsaVersionInfo
class struct__HsaSystemProperties(Structure):
    pass

struct__HsaSystemProperties._pack_ = 1 # source:False
struct__HsaSystemProperties._fields_ = [
    ('NumNodes', ctypes.c_uint32),
    ('PlatformOem', ctypes.c_uint32),
    ('PlatformId', ctypes.c_uint32),
    ('PlatformRev', ctypes.c_uint32),
]

HsaSystemProperties = struct__HsaSystemProperties
class union_HSA_ENGINE_ID(Union):
    pass

class struct_struct_hsakmttypes_h_165(Structure):
    pass

struct_struct_hsakmttypes_h_165._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_165._fields_ = [
    ('uCode', ctypes.c_uint32, 10),
    ('Major', ctypes.c_uint32, 6),
    ('Minor', ctypes.c_uint32, 8),
    ('Stepping', ctypes.c_uint32, 8),
]

union_HSA_ENGINE_ID._pack_ = 1 # source:False
union_HSA_ENGINE_ID._fields_ = [
    ('Value', ctypes.c_uint32),
    ('ui32', struct_struct_hsakmttypes_h_165),
]

HSA_ENGINE_ID = union_HSA_ENGINE_ID
class union_HSA_ENGINE_VERSION(Union):
    pass

class struct_struct_hsakmttypes_h_177(Structure):
    pass

struct_struct_hsakmttypes_h_177._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_177._fields_ = [
    ('uCodeSDMA', ctypes.c_uint32, 10),
    ('uCodeRes', ctypes.c_uint32, 10),
    ('Reserved', ctypes.c_uint32, 12),
]

union_HSA_ENGINE_VERSION._pack_ = 1 # source:False
union_HSA_ENGINE_VERSION._anonymous_ = ('_0',)
union_HSA_ENGINE_VERSION._fields_ = [
    ('Value', ctypes.c_uint32),
    ('_0', struct_struct_hsakmttypes_h_177),
]

HSA_ENGINE_VERSION = union_HSA_ENGINE_VERSION
class union_HSA_CAPABILITY(Union):
    pass

class struct_struct_hsakmttypes_h_188(Structure):
    pass

struct_struct_hsakmttypes_h_188._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_188._fields_ = [
    ('HotPluggable', ctypes.c_uint32, 1),
    ('HSAMMUPresent', ctypes.c_uint32, 1),
    ('SharedWithGraphics', ctypes.c_uint32, 1),
    ('QueueSizePowerOfTwo', ctypes.c_uint32, 1),
    ('QueueSize32bit', ctypes.c_uint32, 1),
    ('QueueIdleEvent', ctypes.c_uint32, 1),
    ('VALimit', ctypes.c_uint32, 1),
    ('WatchPointsSupported', ctypes.c_uint32, 1),
    ('WatchPointsTotalBits', ctypes.c_uint32, 4),
    ('DoorbellType', ctypes.c_uint32, 2),
    ('AQLQueueDoubleMap', ctypes.c_uint32, 1),
    ('DebugTrapSupported', ctypes.c_uint32, 1),
    ('WaveLaunchTrapOverrideSupported', ctypes.c_uint32, 1),
    ('WaveLaunchModeSupported', ctypes.c_uint32, 1),
    ('PreciseMemoryOperationsSupported', ctypes.c_uint32, 1),
    ('DEPRECATED_SRAM_EDCSupport', ctypes.c_uint32, 1),
    ('Mem_EDCSupport', ctypes.c_uint32, 1),
    ('RASEventNotify', ctypes.c_uint32, 1),
    ('ASICRevision', ctypes.c_uint32, 4),
    ('SRAM_EDCSupport', ctypes.c_uint32, 1),
    ('SVMAPISupported', ctypes.c_uint32, 1),
    ('CoherentHostAccess', ctypes.c_uint32, 1),
    ('DebugSupportedFirmware', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 2),
]

union_HSA_CAPABILITY._pack_ = 1 # source:False
union_HSA_CAPABILITY._fields_ = [
    ('Value', ctypes.c_uint32),
    ('ui32', struct_struct_hsakmttypes_h_188),
]

HSA_CAPABILITY = union_HSA_CAPABILITY
class union_HSA_DEBUG_PROPERTIES(Union):
    pass

class struct_struct_hsakmttypes_h_229(Structure):
    pass

struct_struct_hsakmttypes_h_229._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_229._fields_ = [
    ('WatchAddrMaskLoBit', ctypes.c_uint64, 4),
    ('WatchAddrMaskHiBit', ctypes.c_uint64, 6),
    ('DispatchInfoAlwaysValid', ctypes.c_uint64, 1),
    ('AddressWatchpointShareKind', ctypes.c_uint64, 1),
    ('Reserved', ctypes.c_uint64, 52),
]

union_HSA_DEBUG_PROPERTIES._pack_ = 1 # source:False
union_HSA_DEBUG_PROPERTIES._anonymous_ = ('_0',)
union_HSA_DEBUG_PROPERTIES._fields_ = [
    ('Value', ctypes.c_uint64),
    ('_0', struct_struct_hsakmttypes_h_229),
]

HSA_DEBUG_PROPERTIES = union_HSA_DEBUG_PROPERTIES
class struct__HsaNodeProperties(Structure):
    pass

struct__HsaNodeProperties._pack_ = 1 # source:False
struct__HsaNodeProperties._fields_ = [
    ('NumCPUCores', ctypes.c_uint32),
    ('NumFComputeCores', ctypes.c_uint32),
    ('NumMemoryBanks', ctypes.c_uint32),
    ('NumCaches', ctypes.c_uint32),
    ('NumIOLinks', ctypes.c_uint32),
    ('CComputeIdLo', ctypes.c_uint32),
    ('FComputeIdLo', ctypes.c_uint32),
    ('Capability', HSA_CAPABILITY),
    ('MaxWavesPerSIMD', ctypes.c_uint32),
    ('LDSSizeInKB', ctypes.c_uint32),
    ('GDSSizeInKB', ctypes.c_uint32),
    ('WaveFrontSize', ctypes.c_uint32),
    ('NumShaderBanks', ctypes.c_uint32),
    ('NumArrays', ctypes.c_uint32),
    ('NumCUPerArray', ctypes.c_uint32),
    ('NumSIMDPerCU', ctypes.c_uint32),
    ('MaxSlotsScratchCU', ctypes.c_uint32),
    ('EngineId', HSA_ENGINE_ID),
    ('VendorId', ctypes.c_uint16),
    ('DeviceId', ctypes.c_uint16),
    ('LocationId', ctypes.c_uint32),
    ('LocalMemSize', ctypes.c_uint64),
    ('MaxEngineClockMhzFCompute', ctypes.c_uint32),
    ('MaxEngineClockMhzCCompute', ctypes.c_uint32),
    ('DrmRenderMinor', ctypes.c_int32),
    ('MarketingName', ctypes.c_uint16 * 64),
    ('AMDName', ctypes.c_ubyte * 64),
    ('uCodeEngineVersions', HSA_ENGINE_VERSION),
    ('DebugProperties', HSA_DEBUG_PROPERTIES),
    ('HiveID', ctypes.c_uint64),
    ('NumSdmaEngines', ctypes.c_uint32),
    ('NumSdmaXgmiEngines', ctypes.c_uint32),
    ('NumSdmaQueuesPerEngine', ctypes.c_ubyte),
    ('NumCpQueues', ctypes.c_ubyte),
    ('NumGws', ctypes.c_ubyte),
    ('Reserved2', ctypes.c_ubyte),
    ('Domain', ctypes.c_uint32),
    ('UniqueID', ctypes.c_uint64),
    ('VGPRSizePerCU', ctypes.c_uint32),
    ('SGPRSizePerCU', ctypes.c_uint32),
    ('NumXcc', ctypes.c_uint32),
    ('KFDGpuID', ctypes.c_uint32),
    ('FamilyID', ctypes.c_uint32),
]

HsaNodeProperties = struct__HsaNodeProperties

# values for enumeration '_HSA_HEAPTYPE'
_HSA_HEAPTYPE__enumvalues = {
    0: 'HSA_HEAPTYPE_SYSTEM',
    1: 'HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC',
    2: 'HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE',
    3: 'HSA_HEAPTYPE_GPU_GDS',
    4: 'HSA_HEAPTYPE_GPU_LDS',
    5: 'HSA_HEAPTYPE_GPU_SCRATCH',
    6: 'HSA_HEAPTYPE_DEVICE_SVM',
    7: 'HSA_HEAPTYPE_MMIO_REMAP',
    8: 'HSA_HEAPTYPE_NUMHEAPTYPES',
    4294967295: 'HSA_HEAPTYPE_SIZE',
}
HSA_HEAPTYPE_SYSTEM = 0
HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC = 1
HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE = 2
HSA_HEAPTYPE_GPU_GDS = 3
HSA_HEAPTYPE_GPU_LDS = 4
HSA_HEAPTYPE_GPU_SCRATCH = 5
HSA_HEAPTYPE_DEVICE_SVM = 6
HSA_HEAPTYPE_MMIO_REMAP = 7
HSA_HEAPTYPE_NUMHEAPTYPES = 8
HSA_HEAPTYPE_SIZE = 4294967295
_HSA_HEAPTYPE = ctypes.c_uint32 # enum
HSA_HEAPTYPE = _HSA_HEAPTYPE
HSA_HEAPTYPE__enumvalues = _HSA_HEAPTYPE__enumvalues
class union_HSA_MEMORYPROPERTY(Union):
    pass

class struct_struct_hsakmttypes_h_359(Structure):
    pass

struct_struct_hsakmttypes_h_359._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_359._fields_ = [
    ('HotPluggable', ctypes.c_uint32, 1),
    ('NonVolatile', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 30),
]

union_HSA_MEMORYPROPERTY._pack_ = 1 # source:False
union_HSA_MEMORYPROPERTY._fields_ = [
    ('MemoryProperty', ctypes.c_uint32),
    ('ui32', struct_struct_hsakmttypes_h_359),
]

HSA_MEMORYPROPERTY = union_HSA_MEMORYPROPERTY
class struct__HsaMemoryProperties(Structure):
    pass

class union_union_hsakmttypes_h_377(Union):
    pass

class struct_struct_hsakmttypes_h_380(Structure):
    pass

struct_struct_hsakmttypes_h_380._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_380._fields_ = [
    ('SizeInBytesLow', ctypes.c_uint32),
    ('SizeInBytesHigh', ctypes.c_uint32),
]

union_union_hsakmttypes_h_377._pack_ = 1 # source:False
union_union_hsakmttypes_h_377._fields_ = [
    ('SizeInBytes', ctypes.c_uint64),
    ('ui32', struct_struct_hsakmttypes_h_380),
]

struct__HsaMemoryProperties._pack_ = 1 # source:False
struct__HsaMemoryProperties._anonymous_ = ('_0',)
struct__HsaMemoryProperties._fields_ = [
    ('HeapType', HSA_HEAPTYPE),
    ('_0', union_union_hsakmttypes_h_377),
    ('Flags', HSA_MEMORYPROPERTY),
    ('Width', ctypes.c_uint32),
    ('MemoryClockMax', ctypes.c_uint32),
    ('VirtualBaseAddress', ctypes.c_uint64),
]

HsaMemoryProperties = struct__HsaMemoryProperties
class union_HsaCacheType(Union):
    pass

class struct_struct_hsakmttypes_h_407(Structure):
    pass

struct_struct_hsakmttypes_h_407._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_407._fields_ = [
    ('Data', ctypes.c_uint32, 1),
    ('Instruction', ctypes.c_uint32, 1),
    ('CPU', ctypes.c_uint32, 1),
    ('HSACU', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 28),
]

union_HsaCacheType._pack_ = 1 # source:False
union_HsaCacheType._fields_ = [
    ('Value', ctypes.c_uint32),
    ('ui32', struct_struct_hsakmttypes_h_407),
]

HsaCacheType = union_HsaCacheType
class struct__HaCacheProperties(Structure):
    pass

struct__HaCacheProperties._pack_ = 1 # source:False
struct__HaCacheProperties._fields_ = [
    ('ProcessorIdLow', ctypes.c_uint32),
    ('CacheLevel', ctypes.c_uint32),
    ('CacheSize', ctypes.c_uint32),
    ('CacheLineSize', ctypes.c_uint32),
    ('CacheLinesPerTag', ctypes.c_uint32),
    ('CacheAssociativity', ctypes.c_uint32),
    ('CacheLatency', ctypes.c_uint32),
    ('CacheType', HsaCacheType),
    ('SiblingMap', ctypes.c_uint32 * 256),
]

HsaCacheProperties = struct__HaCacheProperties
class struct__HsaCComputeProperties(Structure):
    pass

struct__HsaCComputeProperties._pack_ = 1 # source:False
struct__HsaCComputeProperties._fields_ = [
    ('SiblingMap', ctypes.c_uint32 * 256),
]

HsaCComputeProperties = struct__HsaCComputeProperties

# values for enumeration '_HSA_IOLINKTYPE'
_HSA_IOLINKTYPE__enumvalues = {
    0: 'HSA_IOLINKTYPE_UNDEFINED',
    1: 'HSA_IOLINKTYPE_HYPERTRANSPORT',
    2: 'HSA_IOLINKTYPE_PCIEXPRESS',
    3: 'HSA_IOLINKTYPE_AMBA',
    4: 'HSA_IOLINKTYPE_MIPI',
    5: 'HSA_IOLINK_TYPE_QPI_1_1',
    6: 'HSA_IOLINK_TYPE_RESERVED1',
    7: 'HSA_IOLINK_TYPE_RESERVED2',
    8: 'HSA_IOLINK_TYPE_RAPID_IO',
    9: 'HSA_IOLINK_TYPE_INFINIBAND',
    10: 'HSA_IOLINK_TYPE_RESERVED3',
    11: 'HSA_IOLINK_TYPE_XGMI',
    12: 'HSA_IOLINK_TYPE_XGOP',
    13: 'HSA_IOLINK_TYPE_GZ',
    14: 'HSA_IOLINK_TYPE_ETHERNET_RDMA',
    15: 'HSA_IOLINK_TYPE_RDMA_OTHER',
    16: 'HSA_IOLINK_TYPE_OTHER',
    17: 'HSA_IOLINKTYPE_NUMIOLINKTYPES',
    4294967295: 'HSA_IOLINKTYPE_SIZE',
}
HSA_IOLINKTYPE_UNDEFINED = 0
HSA_IOLINKTYPE_HYPERTRANSPORT = 1
HSA_IOLINKTYPE_PCIEXPRESS = 2
HSA_IOLINKTYPE_AMBA = 3
HSA_IOLINKTYPE_MIPI = 4
HSA_IOLINK_TYPE_QPI_1_1 = 5
HSA_IOLINK_TYPE_RESERVED1 = 6
HSA_IOLINK_TYPE_RESERVED2 = 7
HSA_IOLINK_TYPE_RAPID_IO = 8
HSA_IOLINK_TYPE_INFINIBAND = 9
HSA_IOLINK_TYPE_RESERVED3 = 10
HSA_IOLINK_TYPE_XGMI = 11
HSA_IOLINK_TYPE_XGOP = 12
HSA_IOLINK_TYPE_GZ = 13
HSA_IOLINK_TYPE_ETHERNET_RDMA = 14
HSA_IOLINK_TYPE_RDMA_OTHER = 15
HSA_IOLINK_TYPE_OTHER = 16
HSA_IOLINKTYPE_NUMIOLINKTYPES = 17
HSA_IOLINKTYPE_SIZE = 4294967295
_HSA_IOLINKTYPE = ctypes.c_uint32 # enum
HSA_IOLINKTYPE = _HSA_IOLINKTYPE
HSA_IOLINKTYPE__enumvalues = _HSA_IOLINKTYPE__enumvalues
class union_HSA_LINKPROPERTY(Union):
    pass

class struct_struct_hsakmttypes_h_474(Structure):
    pass

struct_struct_hsakmttypes_h_474._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_474._fields_ = [
    ('Override', ctypes.c_uint32, 1),
    ('NonCoherent', ctypes.c_uint32, 1),
    ('NoAtomics32bit', ctypes.c_uint32, 1),
    ('NoAtomics64bit', ctypes.c_uint32, 1),
    ('NoPeerToPeerDMA', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 27),
]

union_HSA_LINKPROPERTY._pack_ = 1 # source:False
union_HSA_LINKPROPERTY._fields_ = [
    ('LinkProperty', ctypes.c_uint32),
    ('ui32', struct_struct_hsakmttypes_h_474),
]

HSA_LINKPROPERTY = union_HSA_LINKPROPERTY
class struct__HsaIoLinkProperties(Structure):
    pass

struct__HsaIoLinkProperties._pack_ = 1 # source:False
struct__HsaIoLinkProperties._fields_ = [
    ('IoLinkType', HSA_IOLINKTYPE),
    ('VersionMajor', ctypes.c_uint32),
    ('VersionMinor', ctypes.c_uint32),
    ('NodeFrom', ctypes.c_uint32),
    ('NodeTo', ctypes.c_uint32),
    ('Weight', ctypes.c_uint32),
    ('MinimumLatency', ctypes.c_uint32),
    ('MaximumLatency', ctypes.c_uint32),
    ('MinimumBandwidth', ctypes.c_uint32),
    ('MaximumBandwidth', ctypes.c_uint32),
    ('RecTransferSize', ctypes.c_uint32),
    ('Flags', HSA_LINKPROPERTY),
]

HsaIoLinkProperties = struct__HsaIoLinkProperties
class struct__HsaMemFlags(Structure):
    pass

class union_union_hsakmttypes_h_514(Union):
    pass

class struct_struct_hsakmttypes_h_516(Structure):
    pass

struct_struct_hsakmttypes_h_516._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_516._fields_ = [
    ('NonPaged', ctypes.c_uint32, 1),
    ('CachePolicy', ctypes.c_uint32, 2),
    ('ReadOnly', ctypes.c_uint32, 1),
    ('PageSize', ctypes.c_uint32, 2),
    ('HostAccess', ctypes.c_uint32, 1),
    ('NoSubstitute', ctypes.c_uint32, 1),
    ('GDSMemory', ctypes.c_uint32, 1),
    ('Scratch', ctypes.c_uint32, 1),
    ('AtomicAccessFull', ctypes.c_uint32, 1),
    ('AtomicAccessPartial', ctypes.c_uint32, 1),
    ('ExecuteAccess', ctypes.c_uint32, 1),
    ('CoarseGrain', ctypes.c_uint32, 1),
    ('AQLQueueMemory', ctypes.c_uint32, 1),
    ('FixedAddress', ctypes.c_uint32, 1),
    ('NoNUMABind', ctypes.c_uint32, 1),
    ('Uncached', ctypes.c_uint32, 1),
    ('NoAddress', ctypes.c_uint32, 1),
    ('OnlyAddress', ctypes.c_uint32, 1),
    ('ExtendedCoherent', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 11),
]

union_union_hsakmttypes_h_514._pack_ = 1 # source:False
union_union_hsakmttypes_h_514._fields_ = [
    ('ui32', struct_struct_hsakmttypes_h_516),
    ('Value', ctypes.c_uint32),
]

struct__HsaMemFlags._pack_ = 1 # source:False
struct__HsaMemFlags._anonymous_ = ('_0',)
struct__HsaMemFlags._fields_ = [
    ('_0', union_union_hsakmttypes_h_514),
]

HsaMemFlags = struct__HsaMemFlags
class struct__HsaMemMapFlags(Structure):
    pass

class union_union_hsakmttypes_h_579(Union):
    pass

class struct_struct_hsakmttypes_h_581(Structure):
    pass

struct_struct_hsakmttypes_h_581._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_581._fields_ = [
    ('Reserved1', ctypes.c_uint32, 1),
    ('CachePolicy', ctypes.c_uint32, 2),
    ('ReadOnly', ctypes.c_uint32, 1),
    ('PageSize', ctypes.c_uint32, 2),
    ('HostAccess', ctypes.c_uint32, 1),
    ('Migrate', ctypes.c_uint32, 1),
    ('Probe', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 23),
]

union_union_hsakmttypes_h_579._pack_ = 1 # source:False
union_union_hsakmttypes_h_579._fields_ = [
    ('ui32', struct_struct_hsakmttypes_h_581),
    ('Value', ctypes.c_uint32),
]

struct__HsaMemMapFlags._pack_ = 1 # source:False
struct__HsaMemMapFlags._anonymous_ = ('_0',)
struct__HsaMemMapFlags._fields_ = [
    ('_0', union_union_hsakmttypes_h_579),
]

HsaMemMapFlags = struct__HsaMemMapFlags
class struct__HsaGraphicsResourceInfo(Structure):
    pass

struct__HsaGraphicsResourceInfo._pack_ = 1 # source:False
struct__HsaGraphicsResourceInfo._fields_ = [
    ('MemoryAddress', ctypes.POINTER(None)),
    ('SizeInBytes', ctypes.c_uint64),
    ('Metadata', ctypes.POINTER(None)),
    ('MetadataSizeInBytes', ctypes.c_uint32),
    ('NodeId', ctypes.c_uint32),
]

HsaGraphicsResourceInfo = struct__HsaGraphicsResourceInfo

# values for enumeration '_HSA_CACHING_TYPE'
_HSA_CACHING_TYPE__enumvalues = {
    0: 'HSA_CACHING_CACHED',
    1: 'HSA_CACHING_NONCACHED',
    2: 'HSA_CACHING_WRITECOMBINED',
    3: 'HSA_CACHING_RESERVED',
    4: 'HSA_CACHING_NUM_CACHING',
    4294967295: 'HSA_CACHING_SIZE',
}
HSA_CACHING_CACHED = 0
HSA_CACHING_NONCACHED = 1
HSA_CACHING_WRITECOMBINED = 2
HSA_CACHING_RESERVED = 3
HSA_CACHING_NUM_CACHING = 4
HSA_CACHING_SIZE = 4294967295
_HSA_CACHING_TYPE = ctypes.c_uint32 # enum
HSA_CACHING_TYPE = _HSA_CACHING_TYPE
HSA_CACHING_TYPE__enumvalues = _HSA_CACHING_TYPE__enumvalues

# values for enumeration '_HSA_PAGE_SIZE'
_HSA_PAGE_SIZE__enumvalues = {
    0: 'HSA_PAGE_SIZE_4KB',
    1: 'HSA_PAGE_SIZE_64KB',
    2: 'HSA_PAGE_SIZE_2MB',
    3: 'HSA_PAGE_SIZE_1GB',
}
HSA_PAGE_SIZE_4KB = 0
HSA_PAGE_SIZE_64KB = 1
HSA_PAGE_SIZE_2MB = 2
HSA_PAGE_SIZE_1GB = 3
_HSA_PAGE_SIZE = ctypes.c_uint32 # enum
HSA_PAGE_SIZE = _HSA_PAGE_SIZE
HSA_PAGE_SIZE__enumvalues = _HSA_PAGE_SIZE__enumvalues

# values for enumeration '_HSA_DEVICE'
_HSA_DEVICE__enumvalues = {
    0: 'HSA_DEVICE_CPU',
    1: 'HSA_DEVICE_GPU',
    2: 'MAX_HSA_DEVICE',
}
HSA_DEVICE_CPU = 0
HSA_DEVICE_GPU = 1
MAX_HSA_DEVICE = 2
_HSA_DEVICE = ctypes.c_uint32 # enum
HSA_DEVICE = _HSA_DEVICE
HSA_DEVICE__enumvalues = _HSA_DEVICE__enumvalues

# values for enumeration '_HSA_QUEUE_PRIORITY'
_HSA_QUEUE_PRIORITY__enumvalues = {
    -3: 'HSA_QUEUE_PRIORITY_MINIMUM',
    -2: 'HSA_QUEUE_PRIORITY_LOW',
    -1: 'HSA_QUEUE_PRIORITY_BELOW_NORMAL',
    0: 'HSA_QUEUE_PRIORITY_NORMAL',
    1: 'HSA_QUEUE_PRIORITY_ABOVE_NORMAL',
    2: 'HSA_QUEUE_PRIORITY_HIGH',
    3: 'HSA_QUEUE_PRIORITY_MAXIMUM',
    4: 'HSA_QUEUE_PRIORITY_NUM_PRIORITY',
    4294967295: 'HSA_QUEUE_PRIORITY_SIZE',
}
HSA_QUEUE_PRIORITY_MINIMUM = -3
HSA_QUEUE_PRIORITY_LOW = -2
HSA_QUEUE_PRIORITY_BELOW_NORMAL = -1
HSA_QUEUE_PRIORITY_NORMAL = 0
HSA_QUEUE_PRIORITY_ABOVE_NORMAL = 1
HSA_QUEUE_PRIORITY_HIGH = 2
HSA_QUEUE_PRIORITY_MAXIMUM = 3
HSA_QUEUE_PRIORITY_NUM_PRIORITY = 4
HSA_QUEUE_PRIORITY_SIZE = 4294967295
_HSA_QUEUE_PRIORITY = ctypes.c_int64 # enum
HSA_QUEUE_PRIORITY = _HSA_QUEUE_PRIORITY
HSA_QUEUE_PRIORITY__enumvalues = _HSA_QUEUE_PRIORITY__enumvalues

# values for enumeration '_HSA_QUEUE_TYPE'
_HSA_QUEUE_TYPE__enumvalues = {
    1: 'HSA_QUEUE_COMPUTE',
    2: 'HSA_QUEUE_SDMA',
    3: 'HSA_QUEUE_MULTIMEDIA_DECODE',
    4: 'HSA_QUEUE_MULTIMEDIA_ENCODE',
    5: 'HSA_QUEUE_SDMA_XGMI',
    11: 'HSA_QUEUE_COMPUTE_OS',
    12: 'HSA_QUEUE_SDMA_OS',
    13: 'HSA_QUEUE_MULTIMEDIA_DECODE_OS',
    14: 'HSA_QUEUE_MULTIMEDIA_ENCODE_OS',
    21: 'HSA_QUEUE_COMPUTE_AQL',
    22: 'HSA_QUEUE_DMA_AQL',
    23: 'HSA_QUEUE_DMA_AQL_XGMI',
    4294967295: 'HSA_QUEUE_TYPE_SIZE',
}
HSA_QUEUE_COMPUTE = 1
HSA_QUEUE_SDMA = 2
HSA_QUEUE_MULTIMEDIA_DECODE = 3
HSA_QUEUE_MULTIMEDIA_ENCODE = 4
HSA_QUEUE_SDMA_XGMI = 5
HSA_QUEUE_COMPUTE_OS = 11
HSA_QUEUE_SDMA_OS = 12
HSA_QUEUE_MULTIMEDIA_DECODE_OS = 13
HSA_QUEUE_MULTIMEDIA_ENCODE_OS = 14
HSA_QUEUE_COMPUTE_AQL = 21
HSA_QUEUE_DMA_AQL = 22
HSA_QUEUE_DMA_AQL_XGMI = 23
HSA_QUEUE_TYPE_SIZE = 4294967295
_HSA_QUEUE_TYPE = ctypes.c_uint32 # enum
HSA_QUEUE_TYPE = _HSA_QUEUE_TYPE
HSA_QUEUE_TYPE__enumvalues = _HSA_QUEUE_TYPE__enumvalues
class struct_HsaUserContextSaveAreaHeader(Structure):
    pass

struct_HsaUserContextSaveAreaHeader._pack_ = 1 # source:False
struct_HsaUserContextSaveAreaHeader._fields_ = [
    ('ControlStackOffset', ctypes.c_uint32),
    ('ControlStackSize', ctypes.c_uint32),
    ('WaveStateOffset', ctypes.c_uint32),
    ('WaveStateSize', ctypes.c_uint32),
    ('DebugOffset', ctypes.c_uint32),
    ('DebugSize', ctypes.c_uint32),
    ('ErrorReason', ctypes.POINTER(ctypes.c_int64)),
    ('ErrorEventId', ctypes.c_uint32),
    ('Reserved1', ctypes.c_uint32),
]

HsaUserContextSaveAreaHeader = struct_HsaUserContextSaveAreaHeader
class struct_HsaQueueInfo(Structure):
    pass

struct_HsaQueueInfo._pack_ = 1 # source:False
struct_HsaQueueInfo._fields_ = [
    ('QueueDetailError', ctypes.c_uint32),
    ('QueueTypeExtended', ctypes.c_uint32),
    ('NumCUAssigned', ctypes.c_uint32),
    ('CUMaskInfo', ctypes.POINTER(ctypes.c_uint32)),
    ('UserContextSaveArea', ctypes.POINTER(ctypes.c_uint32)),
    ('SaveAreaSizeInBytes', ctypes.c_uint64),
    ('ControlStackTop', ctypes.POINTER(ctypes.c_uint32)),
    ('ControlStackUsedInBytes', ctypes.c_uint64),
    ('SaveAreaHeader', ctypes.POINTER(struct_HsaUserContextSaveAreaHeader)),
    ('Reserved2', ctypes.c_uint64),
]

HsaQueueInfo = struct_HsaQueueInfo
class struct__HsaQueueResource(Structure):
    pass

class union_union_hsakmttypes_h_743(Union):
    pass

union_union_hsakmttypes_h_743._pack_ = 1 # source:False
union_union_hsakmttypes_h_743._fields_ = [
    ('Queue_DoorBell', ctypes.POINTER(ctypes.c_uint32)),
    ('Queue_DoorBell_aql', ctypes.POINTER(ctypes.c_uint64)),
    ('QueueDoorBell', ctypes.c_uint64),
]

class union_union_hsakmttypes_h_751(Union):
    pass

union_union_hsakmttypes_h_751._pack_ = 1 # source:False
union_union_hsakmttypes_h_751._fields_ = [
    ('Queue_write_ptr', ctypes.POINTER(ctypes.c_uint32)),
    ('Queue_write_ptr_aql', ctypes.POINTER(ctypes.c_uint64)),
    ('QueueWptrValue', ctypes.c_uint64),
]

class union_union_hsakmttypes_h_759(Union):
    pass

union_union_hsakmttypes_h_759._pack_ = 1 # source:False
union_union_hsakmttypes_h_759._fields_ = [
    ('Queue_read_ptr', ctypes.POINTER(ctypes.c_uint32)),
    ('Queue_read_ptr_aql', ctypes.POINTER(ctypes.c_uint64)),
    ('QueueRptrValue', ctypes.c_uint64),
]

struct__HsaQueueResource._pack_ = 1 # source:False
struct__HsaQueueResource._anonymous_ = ('_0', '_1', '_2',)
struct__HsaQueueResource._fields_ = [
    ('QueueId', ctypes.c_uint64),
    ('_0', union_union_hsakmttypes_h_743),
    ('_1', union_union_hsakmttypes_h_751),
    ('_2', union_union_hsakmttypes_h_759),
    ('ErrorReason', ctypes.POINTER(ctypes.c_int64)),
]

HsaQueueResource = struct__HsaQueueResource
class struct__HsaQueueReport(Structure):
    pass

struct__HsaQueueReport._pack_ = 1 # source:False
struct__HsaQueueReport._fields_ = [
    ('VMID', ctypes.c_uint32),
    ('QueueAddress', ctypes.POINTER(None)),
    ('QueueSize', ctypes.c_uint64),
]

HsaQueueReport = struct__HsaQueueReport

# values for enumeration '_HSA_DBG_WAVEOP'
_HSA_DBG_WAVEOP__enumvalues = {
    1: 'HSA_DBG_WAVEOP_HALT',
    2: 'HSA_DBG_WAVEOP_RESUME',
    3: 'HSA_DBG_WAVEOP_KILL',
    4: 'HSA_DBG_WAVEOP_DEBUG',
    5: 'HSA_DBG_WAVEOP_TRAP',
    5: 'HSA_DBG_NUM_WAVEOP',
    4294967295: 'HSA_DBG_MAX_WAVEOP',
}
HSA_DBG_WAVEOP_HALT = 1
HSA_DBG_WAVEOP_RESUME = 2
HSA_DBG_WAVEOP_KILL = 3
HSA_DBG_WAVEOP_DEBUG = 4
HSA_DBG_WAVEOP_TRAP = 5
HSA_DBG_NUM_WAVEOP = 5
HSA_DBG_MAX_WAVEOP = 4294967295
_HSA_DBG_WAVEOP = ctypes.c_uint32 # enum
HSA_DBG_WAVEOP = _HSA_DBG_WAVEOP
HSA_DBG_WAVEOP__enumvalues = _HSA_DBG_WAVEOP__enumvalues

# values for enumeration '_HSA_DBG_WAVEMODE'
_HSA_DBG_WAVEMODE__enumvalues = {
    0: 'HSA_DBG_WAVEMODE_SINGLE',
    2: 'HSA_DBG_WAVEMODE_BROADCAST_PROCESS',
    3: 'HSA_DBG_WAVEMODE_BROADCAST_PROCESS_CU',
    3: 'HSA_DBG_NUM_WAVEMODE',
    4294967295: 'HSA_DBG_MAX_WAVEMODE',
}
HSA_DBG_WAVEMODE_SINGLE = 0
HSA_DBG_WAVEMODE_BROADCAST_PROCESS = 2
HSA_DBG_WAVEMODE_BROADCAST_PROCESS_CU = 3
HSA_DBG_NUM_WAVEMODE = 3
HSA_DBG_MAX_WAVEMODE = 4294967295
_HSA_DBG_WAVEMODE = ctypes.c_uint32 # enum
HSA_DBG_WAVEMODE = _HSA_DBG_WAVEMODE
HSA_DBG_WAVEMODE__enumvalues = _HSA_DBG_WAVEMODE__enumvalues

# values for enumeration '_HSA_DBG_WAVEMSG_TYPE'
_HSA_DBG_WAVEMSG_TYPE__enumvalues = {
    0: 'HSA_DBG_WAVEMSG_AUTO',
    1: 'HSA_DBG_WAVEMSG_USER',
    2: 'HSA_DBG_WAVEMSG_ERROR',
    3: 'HSA_DBG_NUM_WAVEMSG',
    4294967295: 'HSA_DBG_MAX_WAVEMSG',
}
HSA_DBG_WAVEMSG_AUTO = 0
HSA_DBG_WAVEMSG_USER = 1
HSA_DBG_WAVEMSG_ERROR = 2
HSA_DBG_NUM_WAVEMSG = 3
HSA_DBG_MAX_WAVEMSG = 4294967295
_HSA_DBG_WAVEMSG_TYPE = ctypes.c_uint32 # enum
HSA_DBG_WAVEMSG_TYPE = _HSA_DBG_WAVEMSG_TYPE
HSA_DBG_WAVEMSG_TYPE__enumvalues = _HSA_DBG_WAVEMSG_TYPE__enumvalues

# values for enumeration '_HSA_DBG_WATCH_MODE'
_HSA_DBG_WATCH_MODE__enumvalues = {
    0: 'HSA_DBG_WATCH_READ',
    1: 'HSA_DBG_WATCH_NONREAD',
    2: 'HSA_DBG_WATCH_ATOMIC',
    3: 'HSA_DBG_WATCH_ALL',
    4: 'HSA_DBG_WATCH_NUM',
}
HSA_DBG_WATCH_READ = 0
HSA_DBG_WATCH_NONREAD = 1
HSA_DBG_WATCH_ATOMIC = 2
HSA_DBG_WATCH_ALL = 3
HSA_DBG_WATCH_NUM = 4
_HSA_DBG_WATCH_MODE = ctypes.c_uint32 # enum
HSA_DBG_WATCH_MODE = _HSA_DBG_WATCH_MODE
HSA_DBG_WATCH_MODE__enumvalues = _HSA_DBG_WATCH_MODE__enumvalues

# values for enumeration '_HSA_DBG_TRAP_OVERRIDE'
_HSA_DBG_TRAP_OVERRIDE__enumvalues = {
    0: 'HSA_DBG_TRAP_OVERRIDE_OR',
    1: 'HSA_DBG_TRAP_OVERRIDE_REPLACE',
    2: 'HSA_DBG_TRAP_OVERRIDE_NUM',
}
HSA_DBG_TRAP_OVERRIDE_OR = 0
HSA_DBG_TRAP_OVERRIDE_REPLACE = 1
HSA_DBG_TRAP_OVERRIDE_NUM = 2
_HSA_DBG_TRAP_OVERRIDE = ctypes.c_uint32 # enum
HSA_DBG_TRAP_OVERRIDE = _HSA_DBG_TRAP_OVERRIDE
HSA_DBG_TRAP_OVERRIDE__enumvalues = _HSA_DBG_TRAP_OVERRIDE__enumvalues

# values for enumeration '_HSA_DBG_TRAP_MASK'
_HSA_DBG_TRAP_MASK__enumvalues = {
    1: 'HSA_DBG_TRAP_MASK_FP_INVALID',
    2: 'HSA_DBG_TRAP_MASK_FP_INPUT_DENOMAL',
    4: 'HSA_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO',
    8: 'HSA_DBG_TRAP_MASK_FP_OVERFLOW',
    16: 'HSA_DBG_TRAP_MASK_FP_UNDERFLOW',
    32: 'HSA_DBG_TRAP_MASK_FP_INEXACT',
    64: 'HSA_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO',
    128: 'HSA_DBG_TRAP_MASK_DBG_ADDRESS_WATCH',
    256: 'HSA_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION',
}
HSA_DBG_TRAP_MASK_FP_INVALID = 1
HSA_DBG_TRAP_MASK_FP_INPUT_DENOMAL = 2
HSA_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO = 4
HSA_DBG_TRAP_MASK_FP_OVERFLOW = 8
HSA_DBG_TRAP_MASK_FP_UNDERFLOW = 16
HSA_DBG_TRAP_MASK_FP_INEXACT = 32
HSA_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO = 64
HSA_DBG_TRAP_MASK_DBG_ADDRESS_WATCH = 128
HSA_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION = 256
_HSA_DBG_TRAP_MASK = ctypes.c_uint32 # enum
HSA_DBG_TRAP_MASK = _HSA_DBG_TRAP_MASK
HSA_DBG_TRAP_MASK__enumvalues = _HSA_DBG_TRAP_MASK__enumvalues

# values for enumeration '_HSA_DBG_TRAP_EXCEPTION_CODE'
_HSA_DBG_TRAP_EXCEPTION_CODE__enumvalues = {
    0: 'HSA_DBG_EC_NONE',
    1: 'HSA_DBG_EC_QUEUE_WAVE_ABORT',
    2: 'HSA_DBG_EC_QUEUE_WAVE_TRAP',
    3: 'HSA_DBG_EC_QUEUE_WAVE_MATH_ERROR',
    4: 'HSA_DBG_EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION',
    5: 'HSA_DBG_EC_QUEUE_WAVE_MEMORY_VIOLATION',
    6: 'HSA_DBG_EC_QUEUE_WAVE_APERTURE_VIOLATION',
    16: 'HSA_DBG_EC_QUEUE_PACKET_DISPATCH_DIM_INVALID',
    17: 'HSA_DBG_EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID',
    18: 'HSA_DBG_EC_QUEUE_PACKET_DISPATCH_CODE_INVALID',
    19: 'HSA_DBG_EC_QUEUE_PACKET_RESERVED',
    20: 'HSA_DBG_EC_QUEUE_PACKET_UNSUPPORTED',
    21: 'HSA_DBG_EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID',
    22: 'HSA_DBG_EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID',
    23: 'HSA_DBG_EC_QUEUE_PACKET_VENDOR_UNSUPPORTED',
    30: 'HSA_DBG_EC_QUEUE_PREEMPTION_ERROR',
    31: 'HSA_DBG_EC_QUEUE_NEW',
    32: 'HSA_DBG_EC_DEVICE_QUEUE_DELETE',
    33: 'HSA_DBG_EC_DEVICE_MEMORY_VIOLATION',
    34: 'HSA_DBG_EC_DEVICE_RAS_ERROR',
    35: 'HSA_DBG_EC_DEVICE_FATAL_HALT',
    36: 'HSA_DBG_EC_DEVICE_NEW',
    48: 'HSA_DBG_EC_PROCESS_RUNTIME',
    49: 'HSA_DBG_EC_PROCESS_DEVICE_REMOVE',
    50: 'HSA_DBG_EC_MAX',
}
HSA_DBG_EC_NONE = 0
HSA_DBG_EC_QUEUE_WAVE_ABORT = 1
HSA_DBG_EC_QUEUE_WAVE_TRAP = 2
HSA_DBG_EC_QUEUE_WAVE_MATH_ERROR = 3
HSA_DBG_EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION = 4
HSA_DBG_EC_QUEUE_WAVE_MEMORY_VIOLATION = 5
HSA_DBG_EC_QUEUE_WAVE_APERTURE_VIOLATION = 6
HSA_DBG_EC_QUEUE_PACKET_DISPATCH_DIM_INVALID = 16
HSA_DBG_EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID = 17
HSA_DBG_EC_QUEUE_PACKET_DISPATCH_CODE_INVALID = 18
HSA_DBG_EC_QUEUE_PACKET_RESERVED = 19
HSA_DBG_EC_QUEUE_PACKET_UNSUPPORTED = 20
HSA_DBG_EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID = 21
HSA_DBG_EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID = 22
HSA_DBG_EC_QUEUE_PACKET_VENDOR_UNSUPPORTED = 23
HSA_DBG_EC_QUEUE_PREEMPTION_ERROR = 30
HSA_DBG_EC_QUEUE_NEW = 31
HSA_DBG_EC_DEVICE_QUEUE_DELETE = 32
HSA_DBG_EC_DEVICE_MEMORY_VIOLATION = 33
HSA_DBG_EC_DEVICE_RAS_ERROR = 34
HSA_DBG_EC_DEVICE_FATAL_HALT = 35
HSA_DBG_EC_DEVICE_NEW = 36
HSA_DBG_EC_PROCESS_RUNTIME = 48
HSA_DBG_EC_PROCESS_DEVICE_REMOVE = 49
HSA_DBG_EC_MAX = 50
_HSA_DBG_TRAP_EXCEPTION_CODE = ctypes.c_uint32 # enum
HSA_DBG_TRAP_EXCEPTION_CODE = _HSA_DBG_TRAP_EXCEPTION_CODE
HSA_DBG_TRAP_EXCEPTION_CODE__enumvalues = _HSA_DBG_TRAP_EXCEPTION_CODE__enumvalues

# values for enumeration '_HSA_DBG_WAVE_LAUNCH_MODE'
_HSA_DBG_WAVE_LAUNCH_MODE__enumvalues = {
    0: 'HSA_DBG_WAVE_LAUNCH_MODE_NORMAL',
    1: 'HSA_DBG_WAVE_LAUNCH_MODE_HALT',
    2: 'HSA_DBG_WAVE_LAUNCH_MODE_KILL',
    3: 'HSA_DBG_WAVE_LAUNCH_MODE_SINGLE_STEP',
    4: 'HSA_DBG_WAVE_LAUNCH_MODE_DISABLE',
    5: 'HSA_DBG_WAVE_LAUNCH_MODE_NUM',
}
HSA_DBG_WAVE_LAUNCH_MODE_NORMAL = 0
HSA_DBG_WAVE_LAUNCH_MODE_HALT = 1
HSA_DBG_WAVE_LAUNCH_MODE_KILL = 2
HSA_DBG_WAVE_LAUNCH_MODE_SINGLE_STEP = 3
HSA_DBG_WAVE_LAUNCH_MODE_DISABLE = 4
HSA_DBG_WAVE_LAUNCH_MODE_NUM = 5
_HSA_DBG_WAVE_LAUNCH_MODE = ctypes.c_uint32 # enum
HSA_DBG_WAVE_LAUNCH_MODE = _HSA_DBG_WAVE_LAUNCH_MODE
HSA_DBG_WAVE_LAUNCH_MODE__enumvalues = _HSA_DBG_WAVE_LAUNCH_MODE__enumvalues

# values for enumeration 'HSA_DBG_NODE_CONTROL'
HSA_DBG_NODE_CONTROL__enumvalues = {
    1: 'HSA_DBG_NODE_CONTROL_FLAG_MAX',
}
HSA_DBG_NODE_CONTROL_FLAG_MAX = 1
HSA_DBG_NODE_CONTROL = ctypes.c_uint32 # enum
class struct__HsaDbgWaveMsgAMDGen2(Structure):
    pass

struct__HsaDbgWaveMsgAMDGen2._pack_ = 1 # source:False
struct__HsaDbgWaveMsgAMDGen2._fields_ = [
    ('Value', ctypes.c_uint32),
    ('Reserved2', ctypes.c_uint32),
]

HsaDbgWaveMsgAMDGen2 = struct__HsaDbgWaveMsgAMDGen2
class union__HsaDbgWaveMessageAMD(Union):
    _pack_ = 1 # source:False
    _fields_ = [
    ('WaveMsgInfoGen2', HsaDbgWaveMsgAMDGen2),
     ]

HsaDbgWaveMessageAMD = union__HsaDbgWaveMessageAMD
class struct__HsaDbgWaveMessage(Structure):
    pass

struct__HsaDbgWaveMessage._pack_ = 1 # source:False
struct__HsaDbgWaveMessage._fields_ = [
    ('MemoryVA', ctypes.POINTER(None)),
    ('DbgWaveMsg', HsaDbgWaveMessageAMD),
]

HsaDbgWaveMessage = struct__HsaDbgWaveMessage

# values for enumeration '_HSA_EVENTTYPE'
_HSA_EVENTTYPE__enumvalues = {
    0: 'HSA_EVENTTYPE_SIGNAL',
    1: 'HSA_EVENTTYPE_NODECHANGE',
    2: 'HSA_EVENTTYPE_DEVICESTATECHANGE',
    3: 'HSA_EVENTTYPE_HW_EXCEPTION',
    4: 'HSA_EVENTTYPE_SYSTEM_EVENT',
    5: 'HSA_EVENTTYPE_DEBUG_EVENT',
    6: 'HSA_EVENTTYPE_PROFILE_EVENT',
    7: 'HSA_EVENTTYPE_QUEUE_EVENT',
    8: 'HSA_EVENTTYPE_MEMORY',
    9: 'HSA_EVENTTYPE_MAXID',
    4294967295: 'HSA_EVENTTYPE_TYPE_SIZE',
}
HSA_EVENTTYPE_SIGNAL = 0
HSA_EVENTTYPE_NODECHANGE = 1
HSA_EVENTTYPE_DEVICESTATECHANGE = 2
HSA_EVENTTYPE_HW_EXCEPTION = 3
HSA_EVENTTYPE_SYSTEM_EVENT = 4
HSA_EVENTTYPE_DEBUG_EVENT = 5
HSA_EVENTTYPE_PROFILE_EVENT = 6
HSA_EVENTTYPE_QUEUE_EVENT = 7
HSA_EVENTTYPE_MEMORY = 8
HSA_EVENTTYPE_MAXID = 9
HSA_EVENTTYPE_TYPE_SIZE = 4294967295
_HSA_EVENTTYPE = ctypes.c_uint32 # enum
HSA_EVENTTYPE = _HSA_EVENTTYPE
HSA_EVENTTYPE__enumvalues = _HSA_EVENTTYPE__enumvalues

# values for enumeration '_HSA_DEBUG_EVENT_TYPE'
_HSA_DEBUG_EVENT_TYPE__enumvalues = {
    0: 'HSA_DEBUG_EVENT_TYPE_NONE',
    1: 'HSA_DEBUG_EVENT_TYPE_TRAP',
    2: 'HSA_DEBUG_EVENT_TYPE_VMFAULT',
    3: 'HSA_DEBUG_EVENT_TYPE_TRAP_VMFAULT',
}
HSA_DEBUG_EVENT_TYPE_NONE = 0
HSA_DEBUG_EVENT_TYPE_TRAP = 1
HSA_DEBUG_EVENT_TYPE_VMFAULT = 2
HSA_DEBUG_EVENT_TYPE_TRAP_VMFAULT = 3
_HSA_DEBUG_EVENT_TYPE = ctypes.c_uint32 # enum
HSA_DEBUG_EVENT_TYPE = _HSA_DEBUG_EVENT_TYPE
HSA_DEBUG_EVENT_TYPE__enumvalues = _HSA_DEBUG_EVENT_TYPE__enumvalues
HSA_EVENTID = ctypes.c_uint32
class struct__HsaSyncVar(Structure):
    pass

class union_union_hsakmttypes_h_972(Union):
    pass

union_union_hsakmttypes_h_972._pack_ = 1 # source:False
union_union_hsakmttypes_h_972._fields_ = [
    ('UserData', ctypes.POINTER(None)),
    ('UserDataPtrValue', ctypes.c_uint64),
]

struct__HsaSyncVar._pack_ = 1 # source:False
struct__HsaSyncVar._fields_ = [
    ('SyncVar', union_union_hsakmttypes_h_972),
    ('SyncVarSize', ctypes.c_uint64),
]

HsaSyncVar = struct__HsaSyncVar

# values for enumeration '_HSA_EVENTTYPE_NODECHANGE_FLAGS'
_HSA_EVENTTYPE_NODECHANGE_FLAGS__enumvalues = {
    0: 'HSA_EVENTTYPE_NODECHANGE_ADD',
    1: 'HSA_EVENTTYPE_NODECHANGE_REMOVE',
    4294967295: 'HSA_EVENTTYPE_NODECHANGE_SIZE',
}
HSA_EVENTTYPE_NODECHANGE_ADD = 0
HSA_EVENTTYPE_NODECHANGE_REMOVE = 1
HSA_EVENTTYPE_NODECHANGE_SIZE = 4294967295
_HSA_EVENTTYPE_NODECHANGE_FLAGS = ctypes.c_uint32 # enum
HSA_EVENTTYPE_NODECHANGE_FLAGS = _HSA_EVENTTYPE_NODECHANGE_FLAGS
HSA_EVENTTYPE_NODECHANGE_FLAGS__enumvalues = _HSA_EVENTTYPE_NODECHANGE_FLAGS__enumvalues
class struct__HsaNodeChange(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('Flags', HSA_EVENTTYPE_NODECHANGE_FLAGS),
     ]

HsaNodeChange = struct__HsaNodeChange

# values for enumeration '_HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS'
_HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS__enumvalues = {
    0: 'HSA_EVENTTYPE_DEVICESTATUSCHANGE_START',
    1: 'HSA_EVENTTYPE_DEVICESTATUSCHANGE_STOP',
    4294967295: 'HSA_EVENTTYPE_DEVICESTATUSCHANGE_SIZE',
}
HSA_EVENTTYPE_DEVICESTATUSCHANGE_START = 0
HSA_EVENTTYPE_DEVICESTATUSCHANGE_STOP = 1
HSA_EVENTTYPE_DEVICESTATUSCHANGE_SIZE = 4294967295
_HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS = ctypes.c_uint32 # enum
HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS = _HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS
HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS__enumvalues = _HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS__enumvalues
class struct__HsaDeviceStateChange(Structure):
    pass

struct__HsaDeviceStateChange._pack_ = 1 # source:False
struct__HsaDeviceStateChange._fields_ = [
    ('NodeId', ctypes.c_uint32),
    ('Device', HSA_DEVICE),
    ('Flags', HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS),
]

HsaDeviceStateChange = struct__HsaDeviceStateChange

# values for enumeration '_HSA_EVENTID_MEMORYFLAGS'
_HSA_EVENTID_MEMORYFLAGS__enumvalues = {
    0: 'HSA_EVENTID_MEMORY_RECOVERABLE',
    1: 'HSA_EVENTID_MEMORY_FATAL_PROCESS',
    2: 'HSA_EVENTID_MEMORY_FATAL_VM',
}
HSA_EVENTID_MEMORY_RECOVERABLE = 0
HSA_EVENTID_MEMORY_FATAL_PROCESS = 1
HSA_EVENTID_MEMORY_FATAL_VM = 2
_HSA_EVENTID_MEMORYFLAGS = ctypes.c_uint32 # enum
HSA_EVENTID_MEMORYFLAGS = _HSA_EVENTID_MEMORYFLAGS
HSA_EVENTID_MEMORYFLAGS__enumvalues = _HSA_EVENTID_MEMORYFLAGS__enumvalues
class struct__HsaAccessAttributeFailure(Structure):
    pass

struct__HsaAccessAttributeFailure._pack_ = 1 # source:False
struct__HsaAccessAttributeFailure._fields_ = [
    ('NotPresent', ctypes.c_uint32, 1),
    ('ReadOnly', ctypes.c_uint32, 1),
    ('NoExecute', ctypes.c_uint32, 1),
    ('GpuAccess', ctypes.c_uint32, 1),
    ('ECC', ctypes.c_uint32, 1),
    ('Imprecise', ctypes.c_uint32, 1),
    ('ErrorType', ctypes.c_uint32, 3),
    ('Reserved', ctypes.c_uint32, 23),
]

HsaAccessAttributeFailure = struct__HsaAccessAttributeFailure
class struct__HsaMemoryAccessFault(Structure):
    pass

struct__HsaMemoryAccessFault._pack_ = 1 # source:False
struct__HsaMemoryAccessFault._fields_ = [
    ('NodeId', ctypes.c_uint32),
    ('VirtualAddress', ctypes.c_uint64),
    ('Failure', HsaAccessAttributeFailure),
    ('Flags', HSA_EVENTID_MEMORYFLAGS),
]

HsaMemoryAccessFault = struct__HsaMemoryAccessFault

# values for enumeration '_HSA_EVENTID_HW_EXCEPTION_CAUSE'
_HSA_EVENTID_HW_EXCEPTION_CAUSE__enumvalues = {
    0: 'HSA_EVENTID_HW_EXCEPTION_GPU_HANG',
    1: 'HSA_EVENTID_HW_EXCEPTION_ECC',
}
HSA_EVENTID_HW_EXCEPTION_GPU_HANG = 0
HSA_EVENTID_HW_EXCEPTION_ECC = 1
_HSA_EVENTID_HW_EXCEPTION_CAUSE = ctypes.c_uint32 # enum
HSA_EVENTID_HW_EXCEPTION_CAUSE = _HSA_EVENTID_HW_EXCEPTION_CAUSE
HSA_EVENTID_HW_EXCEPTION_CAUSE__enumvalues = _HSA_EVENTID_HW_EXCEPTION_CAUSE__enumvalues
class struct__HsaHwException(Structure):
    pass

struct__HsaHwException._pack_ = 1 # source:False
struct__HsaHwException._fields_ = [
    ('NodeId', ctypes.c_uint32),
    ('ResetType', ctypes.c_uint32),
    ('MemoryLost', ctypes.c_uint32),
    ('ResetCause', HSA_EVENTID_HW_EXCEPTION_CAUSE),
]

HsaHwException = struct__HsaHwException
class struct__HsaEventData(Structure):
    pass

class union_union_hsakmttypes_h_1066(Union):
    pass

union_union_hsakmttypes_h_1066._pack_ = 1 # source:False
union_union_hsakmttypes_h_1066._fields_ = [
    ('SyncVar', HsaSyncVar),
    ('NodeChangeState', HsaNodeChange),
    ('DeviceState', HsaDeviceStateChange),
    ('MemoryAccessFault', HsaMemoryAccessFault),
    ('HwException', HsaHwException),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct__HsaEventData._pack_ = 1 # source:False
struct__HsaEventData._fields_ = [
    ('EventType', HSA_EVENTTYPE),
    ('EventData', union_union_hsakmttypes_h_1066),
    ('HWData1', ctypes.c_uint64),
    ('HWData2', ctypes.c_uint64),
    ('HWData3', ctypes.c_uint32),
]

HsaEventData = struct__HsaEventData
class struct__HsaEventDescriptor(Structure):
    pass

struct__HsaEventDescriptor._pack_ = 1 # source:False
struct__HsaEventDescriptor._fields_ = [
    ('EventType', HSA_EVENTTYPE),
    ('NodeId', ctypes.c_uint32),
    ('SyncVar', HsaSyncVar),
]

HsaEventDescriptor = struct__HsaEventDescriptor
class struct__HsaEvent(Structure):
    pass

struct__HsaEvent._pack_ = 1 # source:False
struct__HsaEvent._fields_ = [
    ('EventId', ctypes.c_uint32),
    ('EventData', HsaEventData),
]

HsaEvent = struct__HsaEvent

# values for enumeration '_HsaEventTimeout'
_HsaEventTimeout__enumvalues = {
    0: 'HSA_EVENTTIMEOUT_IMMEDIATE',
    4294967295: 'HSA_EVENTTIMEOUT_INFINITE',
}
HSA_EVENTTIMEOUT_IMMEDIATE = 0
HSA_EVENTTIMEOUT_INFINITE = 4294967295
_HsaEventTimeout = ctypes.c_uint32 # enum
HsaEventTimeOut = _HsaEventTimeout
HsaEventTimeOut__enumvalues = _HsaEventTimeout__enumvalues
class struct__HsaClockCounters(Structure):
    pass

struct__HsaClockCounters._pack_ = 1 # source:False
struct__HsaClockCounters._fields_ = [
    ('GPUClockCounter', ctypes.c_uint64),
    ('CPUClockCounter', ctypes.c_uint64),
    ('SystemClockCounter', ctypes.c_uint64),
    ('SystemClockFrequencyHz', ctypes.c_uint64),
]

HsaClockCounters = struct__HsaClockCounters
class struct__HSA_UUID(Structure):
    pass

struct__HSA_UUID._pack_ = 1 # source:False
struct__HSA_UUID._fields_ = [
    ('Data1', ctypes.c_uint32),
    ('Data2', ctypes.c_uint16),
    ('Data3', ctypes.c_uint16),
    ('Data4', ctypes.c_ubyte * 8),
]

HSA_UUID = struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_CB = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_CPF = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_CPG = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_DB = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_GDS = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_GRBM = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_GRBMSE = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_IA = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_MC = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_PASC = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_PASU = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_SPI = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_SRBM = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_SQ = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_SX = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_TA = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_TCA = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_TCC = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_TCP = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_TCS = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_TD = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_VGT = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_WD = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_IOMMUV2 = struct__HSA_UUID # Variable struct__HSA_UUID
HSA_PROFILEBLOCK_AMD_KERNEL_DRIVER = struct__HSA_UUID # Variable struct__HSA_UUID

# values for enumeration '_HSA_PROFILE_TYPE'
_HSA_PROFILE_TYPE__enumvalues = {
    0: 'HSA_PROFILE_TYPE_PRIVILEGED_IMMEDIATE',
    1: 'HSA_PROFILE_TYPE_PRIVILEGED_STREAMING',
    2: 'HSA_PROFILE_TYPE_NONPRIV_IMMEDIATE',
    3: 'HSA_PROFILE_TYPE_NONPRIV_STREAMING',
    4: 'HSA_PROFILE_TYPE_NUM',
    4294967295: 'HSA_PROFILE_TYPE_SIZE',
}
HSA_PROFILE_TYPE_PRIVILEGED_IMMEDIATE = 0
HSA_PROFILE_TYPE_PRIVILEGED_STREAMING = 1
HSA_PROFILE_TYPE_NONPRIV_IMMEDIATE = 2
HSA_PROFILE_TYPE_NONPRIV_STREAMING = 3
HSA_PROFILE_TYPE_NUM = 4
HSA_PROFILE_TYPE_SIZE = 4294967295
_HSA_PROFILE_TYPE = ctypes.c_uint32 # enum
HSA_PROFILE_TYPE = _HSA_PROFILE_TYPE
HSA_PROFILE_TYPE__enumvalues = _HSA_PROFILE_TYPE__enumvalues
class struct__HsaCounterFlags(Structure):
    pass

class union_union_hsakmttypes_h_1277(Union):
    pass

class struct_struct_hsakmttypes_h_1279(Structure):
    pass

struct_struct_hsakmttypes_h_1279._pack_ = 1 # source:False
struct_struct_hsakmttypes_h_1279._fields_ = [
    ('Global', ctypes.c_uint32, 1),
    ('Resettable', ctypes.c_uint32, 1),
    ('ReadOnly', ctypes.c_uint32, 1),
    ('Stream', ctypes.c_uint32, 1),
    ('Reserved', ctypes.c_uint32, 28),
]

union_union_hsakmttypes_h_1277._pack_ = 1 # source:False
union_union_hsakmttypes_h_1277._fields_ = [
    ('ui32', struct_struct_hsakmttypes_h_1279),
    ('Value', ctypes.c_uint32),
]

struct__HsaCounterFlags._pack_ = 1 # source:False
struct__HsaCounterFlags._anonymous_ = ('_0',)
struct__HsaCounterFlags._fields_ = [
    ('_0', union_union_hsakmttypes_h_1277),
]

HsaCounterFlags = struct__HsaCounterFlags
class struct__HsaCounter(Structure):
    pass

struct__HsaCounter._pack_ = 1 # source:False
struct__HsaCounter._fields_ = [
    ('Type', HSA_PROFILE_TYPE),
    ('CounterId', ctypes.c_uint64),
    ('CounterSizeInBits', ctypes.c_uint32),
    ('CounterMask', ctypes.c_uint64),
    ('Flags', HsaCounterFlags),
    ('BlockIndex', ctypes.c_uint32),
]

HsaCounter = struct__HsaCounter
class struct__HsaCounterBlockProperties(Structure):
    pass

struct__HsaCounterBlockProperties._pack_ = 1 # source:False
struct__HsaCounterBlockProperties._fields_ = [
    ('BlockId', HSA_UUID),
    ('NumCounters', ctypes.c_uint32),
    ('NumConcurrent', ctypes.c_uint32),
    ('Counters', struct__HsaCounter * 1),
]

HsaCounterBlockProperties = struct__HsaCounterBlockProperties
class struct__HsaCounterProperties(Structure):
    pass

struct__HsaCounterProperties._pack_ = 1 # source:False
struct__HsaCounterProperties._fields_ = [
    ('NumBlocks', ctypes.c_uint32),
    ('NumConcurrent', ctypes.c_uint32),
    ('Blocks', struct__HsaCounterBlockProperties * 1),
]

HsaCounterProperties = struct__HsaCounterProperties
HSATraceId = ctypes.c_uint64
class struct__HsaPmcTraceRoot(Structure):
    pass

struct__HsaPmcTraceRoot._pack_ = 1 # source:False
struct__HsaPmcTraceRoot._fields_ = [
    ('TraceBufferMinSizeBytes', ctypes.c_uint64),
    ('NumberOfPasses', ctypes.c_uint32),
    ('TraceId', ctypes.c_uint64),
]

HsaPmcTraceRoot = struct__HsaPmcTraceRoot
class struct__HsaGpuTileConfig(Structure):
    pass

struct__HsaGpuTileConfig._pack_ = 1 # source:False
struct__HsaGpuTileConfig._fields_ = [
    ('TileConfig', ctypes.POINTER(ctypes.c_uint32)),
    ('MacroTileConfig', ctypes.POINTER(ctypes.c_uint32)),
    ('NumTileConfigs', ctypes.c_uint32),
    ('NumMacroTileConfigs', ctypes.c_uint32),
    ('GbAddrConfig', ctypes.c_uint32),
    ('NumBanks', ctypes.c_uint32),
    ('NumRanks', ctypes.c_uint32),
    ('Reserved', ctypes.c_uint32 * 7),
]

HsaGpuTileConfig = struct__HsaGpuTileConfig

# values for enumeration '_HSA_POINTER_TYPE'
_HSA_POINTER_TYPE__enumvalues = {
    0: 'HSA_POINTER_UNKNOWN',
    1: 'HSA_POINTER_ALLOCATED',
    2: 'HSA_POINTER_REGISTERED_USER',
    3: 'HSA_POINTER_REGISTERED_GRAPHICS',
    4: 'HSA_POINTER_REGISTERED_SHARED',
    5: 'HSA_POINTER_RESERVED_ADDR',
}
HSA_POINTER_UNKNOWN = 0
HSA_POINTER_ALLOCATED = 1
HSA_POINTER_REGISTERED_USER = 2
HSA_POINTER_REGISTERED_GRAPHICS = 3
HSA_POINTER_REGISTERED_SHARED = 4
HSA_POINTER_RESERVED_ADDR = 5
_HSA_POINTER_TYPE = ctypes.c_uint32 # enum
HSA_POINTER_TYPE = _HSA_POINTER_TYPE
HSA_POINTER_TYPE__enumvalues = _HSA_POINTER_TYPE__enumvalues
class struct__HsaPointerInfo(Structure):
    pass

struct__HsaPointerInfo._pack_ = 1 # source:False
struct__HsaPointerInfo._fields_ = [
    ('Type', HSA_POINTER_TYPE),
    ('Node', ctypes.c_uint32),
    ('MemFlags', HsaMemFlags),
    ('CPUAddress', ctypes.POINTER(None)),
    ('GPUAddress', ctypes.c_uint64),
    ('SizeInBytes', ctypes.c_uint64),
    ('NRegisteredNodes', ctypes.c_uint32),
    ('NMappedNodes', ctypes.c_uint32),
    ('RegisteredNodes', ctypes.POINTER(ctypes.c_uint32)),
    ('MappedNodes', ctypes.POINTER(ctypes.c_uint32)),
    ('UserData', ctypes.POINTER(None)),
]

HsaPointerInfo = struct__HsaPointerInfo
HsaSharedMemoryHandle = ctypes.c_uint32 * 8
class struct__HsaMemoryRange(Structure):
    pass

struct__HsaMemoryRange._pack_ = 1 # source:False
struct__HsaMemoryRange._fields_ = [
    ('MemoryAddress', ctypes.POINTER(None)),
    ('SizeInBytes', ctypes.c_uint64),
]

HsaMemoryRange = struct__HsaMemoryRange

# values for enumeration '_HSA_SVM_FLAGS'
_HSA_SVM_FLAGS__enumvalues = {
    1: 'HSA_SVM_FLAG_HOST_ACCESS',
    2: 'HSA_SVM_FLAG_COHERENT',
    4: 'HSA_SVM_FLAG_HIVE_LOCAL',
    8: 'HSA_SVM_FLAG_GPU_RO',
    16: 'HSA_SVM_FLAG_GPU_EXEC',
    32: 'HSA_SVM_FLAG_GPU_READ_MOSTLY',
    64: 'HSA_SVM_FLAG_GPU_ALWAYS_MAPPED',
    128: 'HSA_SVM_FLAG_EXT_COHERENT',
}
HSA_SVM_FLAG_HOST_ACCESS = 1
HSA_SVM_FLAG_COHERENT = 2
HSA_SVM_FLAG_HIVE_LOCAL = 4
HSA_SVM_FLAG_GPU_RO = 8
HSA_SVM_FLAG_GPU_EXEC = 16
HSA_SVM_FLAG_GPU_READ_MOSTLY = 32
HSA_SVM_FLAG_GPU_ALWAYS_MAPPED = 64
HSA_SVM_FLAG_EXT_COHERENT = 128
_HSA_SVM_FLAGS = ctypes.c_uint32 # enum
HSA_SVM_FLAGS = _HSA_SVM_FLAGS
HSA_SVM_FLAGS__enumvalues = _HSA_SVM_FLAGS__enumvalues

# values for enumeration '_HSA_SVM_ATTR_TYPE'
_HSA_SVM_ATTR_TYPE__enumvalues = {
    0: 'HSA_SVM_ATTR_PREFERRED_LOC',
    1: 'HSA_SVM_ATTR_PREFETCH_LOC',
    2: 'HSA_SVM_ATTR_ACCESS',
    3: 'HSA_SVM_ATTR_ACCESS_IN_PLACE',
    4: 'HSA_SVM_ATTR_NO_ACCESS',
    5: 'HSA_SVM_ATTR_SET_FLAGS',
    6: 'HSA_SVM_ATTR_CLR_FLAGS',
    7: 'HSA_SVM_ATTR_GRANULARITY',
}
HSA_SVM_ATTR_PREFERRED_LOC = 0
HSA_SVM_ATTR_PREFETCH_LOC = 1
HSA_SVM_ATTR_ACCESS = 2
HSA_SVM_ATTR_ACCESS_IN_PLACE = 3
HSA_SVM_ATTR_NO_ACCESS = 4
HSA_SVM_ATTR_SET_FLAGS = 5
HSA_SVM_ATTR_CLR_FLAGS = 6
HSA_SVM_ATTR_GRANULARITY = 7
_HSA_SVM_ATTR_TYPE = ctypes.c_uint32 # enum
HSA_SVM_ATTR_TYPE = _HSA_SVM_ATTR_TYPE
HSA_SVM_ATTR_TYPE__enumvalues = _HSA_SVM_ATTR_TYPE__enumvalues
class struct__HSA_SVM_ATTRIBUTE(Structure):
    pass

struct__HSA_SVM_ATTRIBUTE._pack_ = 1 # source:False
struct__HSA_SVM_ATTRIBUTE._fields_ = [
    ('type', ctypes.c_uint32),
    ('value', ctypes.c_uint32),
]

HSA_SVM_ATTRIBUTE = struct__HSA_SVM_ATTRIBUTE

# values for enumeration '_HSA_SMI_EVENT'
_HSA_SMI_EVENT__enumvalues = {
    0: 'HSA_SMI_EVENT_NONE',
    1: 'HSA_SMI_EVENT_VMFAULT',
    2: 'HSA_SMI_EVENT_THERMAL_THROTTLE',
    3: 'HSA_SMI_EVENT_GPU_PRE_RESET',
    4: 'HSA_SMI_EVENT_GPU_POST_RESET',
    5: 'HSA_SMI_EVENT_MIGRATE_START',
    6: 'HSA_SMI_EVENT_MIGRATE_END',
    7: 'HSA_SMI_EVENT_PAGE_FAULT_START',
    8: 'HSA_SMI_EVENT_PAGE_FAULT_END',
    9: 'HSA_SMI_EVENT_QUEUE_EVICTION',
    10: 'HSA_SMI_EVENT_QUEUE_RESTORE',
    11: 'HSA_SMI_EVENT_UNMAP_FROM_GPU',
    12: 'HSA_SMI_EVENT_INDEX_MAX',
    64: 'HSA_SMI_EVENT_ALL_PROCESS',
}
HSA_SMI_EVENT_NONE = 0
HSA_SMI_EVENT_VMFAULT = 1
HSA_SMI_EVENT_THERMAL_THROTTLE = 2
HSA_SMI_EVENT_GPU_PRE_RESET = 3
HSA_SMI_EVENT_GPU_POST_RESET = 4
HSA_SMI_EVENT_MIGRATE_START = 5
HSA_SMI_EVENT_MIGRATE_END = 6
HSA_SMI_EVENT_PAGE_FAULT_START = 7
HSA_SMI_EVENT_PAGE_FAULT_END = 8
HSA_SMI_EVENT_QUEUE_EVICTION = 9
HSA_SMI_EVENT_QUEUE_RESTORE = 10
HSA_SMI_EVENT_UNMAP_FROM_GPU = 11
HSA_SMI_EVENT_INDEX_MAX = 12
HSA_SMI_EVENT_ALL_PROCESS = 64
_HSA_SMI_EVENT = ctypes.c_uint32 # enum
HSA_EVENT_TYPE = _HSA_SMI_EVENT
HSA_EVENT_TYPE__enumvalues = _HSA_SMI_EVENT__enumvalues

# values for enumeration '_HSA_MIGRATE_TRIGGERS'
_HSA_MIGRATE_TRIGGERS__enumvalues = {
    0: 'HSA_MIGRATE_TRIGGER_PREFETCH',
    1: 'HSA_MIGRATE_TRIGGER_PAGEFAULT_GPU',
    2: 'HSA_MIGRATE_TRIGGER_PAGEFAULT_CPU',
    3: 'HSA_MIGRATE_TRIGGER_TTM_EVICTION',
}
HSA_MIGRATE_TRIGGER_PREFETCH = 0
HSA_MIGRATE_TRIGGER_PAGEFAULT_GPU = 1
HSA_MIGRATE_TRIGGER_PAGEFAULT_CPU = 2
HSA_MIGRATE_TRIGGER_TTM_EVICTION = 3
_HSA_MIGRATE_TRIGGERS = ctypes.c_uint32 # enum
HSA_MIGRATE_TRIGGERS = _HSA_MIGRATE_TRIGGERS
HSA_MIGRATE_TRIGGERS__enumvalues = _HSA_MIGRATE_TRIGGERS__enumvalues

# values for enumeration '_HSA_QUEUE_EVICTION_TRIGGERS'
_HSA_QUEUE_EVICTION_TRIGGERS__enumvalues = {
    0: 'HSA_QUEUE_EVICTION_TRIGGER_SVM',
    1: 'HSA_QUEUE_EVICTION_TRIGGER_USERPTR',
    2: 'HSA_QUEUE_EVICTION_TRIGGER_TTM',
    3: 'HSA_QUEUE_EVICTION_TRIGGER_SUSPEND',
    4: 'HSA_QUEUE_EVICTION_CRIU_CHECKPOINT',
    5: 'HSA_QUEUE_EVICTION_CRIU_RESTORE',
}
HSA_QUEUE_EVICTION_TRIGGER_SVM = 0
HSA_QUEUE_EVICTION_TRIGGER_USERPTR = 1
HSA_QUEUE_EVICTION_TRIGGER_TTM = 2
HSA_QUEUE_EVICTION_TRIGGER_SUSPEND = 3
HSA_QUEUE_EVICTION_CRIU_CHECKPOINT = 4
HSA_QUEUE_EVICTION_CRIU_RESTORE = 5
_HSA_QUEUE_EVICTION_TRIGGERS = ctypes.c_uint32 # enum
HSA_QUEUE_EVICTION_TRIGGERS = _HSA_QUEUE_EVICTION_TRIGGERS
HSA_QUEUE_EVICTION_TRIGGERS__enumvalues = _HSA_QUEUE_EVICTION_TRIGGERS__enumvalues

# values for enumeration '_HSA_SVM_UNMAP_TRIGGERS'
_HSA_SVM_UNMAP_TRIGGERS__enumvalues = {
    0: 'HSA_SVM_UNMAP_TRIGGER_MMU_NOTIFY',
    1: 'HSA_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE',
    2: 'HSA_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU',
}
HSA_SVM_UNMAP_TRIGGER_MMU_NOTIFY = 0
HSA_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE = 1
HSA_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU = 2
_HSA_SVM_UNMAP_TRIGGERS = ctypes.c_uint32 # enum
HSA_SVM_UNMAP_TRIGGERS = _HSA_SVM_UNMAP_TRIGGERS
HSA_SVM_UNMAP_TRIGGERS__enumvalues = _HSA_SVM_UNMAP_TRIGGERS__enumvalues
HsaAMDGPUDeviceHandle = ctypes.POINTER(None)
try:
    hsaKmtOpenKFD = _libraries['libhsakmt.so'].hsaKmtOpenKFD
    hsaKmtOpenKFD.restype = HSAKMT_STATUS
    hsaKmtOpenKFD.argtypes = []
except AttributeError:
    pass
try:
    hsaKmtCloseKFD = _libraries['libhsakmt.so'].hsaKmtCloseKFD
    hsaKmtCloseKFD.restype = HSAKMT_STATUS
    hsaKmtCloseKFD.argtypes = []
except AttributeError:
    pass
try:
    hsaKmtGetVersion = _libraries['libhsakmt.so'].hsaKmtGetVersion
    hsaKmtGetVersion.restype = HSAKMT_STATUS
    hsaKmtGetVersion.argtypes = [ctypes.POINTER(struct__HsaVersionInfo)]
except AttributeError:
    pass
try:
    hsaKmtAcquireSystemProperties = _libraries['libhsakmt.so'].hsaKmtAcquireSystemProperties
    hsaKmtAcquireSystemProperties.restype = HSAKMT_STATUS
    hsaKmtAcquireSystemProperties.argtypes = [ctypes.POINTER(struct__HsaSystemProperties)]
except AttributeError:
    pass
try:
    hsaKmtReleaseSystemProperties = _libraries['libhsakmt.so'].hsaKmtReleaseSystemProperties
    hsaKmtReleaseSystemProperties.restype = HSAKMT_STATUS
    hsaKmtReleaseSystemProperties.argtypes = []
except AttributeError:
    pass
try:
    hsaKmtGetNodeProperties = _libraries['libhsakmt.so'].hsaKmtGetNodeProperties
    hsaKmtGetNodeProperties.restype = HSAKMT_STATUS
    hsaKmtGetNodeProperties.argtypes = [HSAuint32, ctypes.POINTER(struct__HsaNodeProperties)]
except AttributeError:
    pass
try:
    hsaKmtGetNodeMemoryProperties = _libraries['libhsakmt.so'].hsaKmtGetNodeMemoryProperties
    hsaKmtGetNodeMemoryProperties.restype = HSAKMT_STATUS
    hsaKmtGetNodeMemoryProperties.argtypes = [HSAuint32, HSAuint32, ctypes.POINTER(struct__HsaMemoryProperties)]
except AttributeError:
    pass
try:
    hsaKmtGetNodeCacheProperties = _libraries['libhsakmt.so'].hsaKmtGetNodeCacheProperties
    hsaKmtGetNodeCacheProperties.restype = HSAKMT_STATUS
    hsaKmtGetNodeCacheProperties.argtypes = [HSAuint32, HSAuint32, HSAuint32, ctypes.POINTER(struct__HaCacheProperties)]
except AttributeError:
    pass
try:
    hsaKmtGetNodeIoLinkProperties = _libraries['libhsakmt.so'].hsaKmtGetNodeIoLinkProperties
    hsaKmtGetNodeIoLinkProperties.restype = HSAKMT_STATUS
    hsaKmtGetNodeIoLinkProperties.argtypes = [HSAuint32, HSAuint32, ctypes.POINTER(struct__HsaIoLinkProperties)]
except AttributeError:
    pass
try:
    hsaKmtCreateEvent = _libraries['libhsakmt.so'].hsaKmtCreateEvent
    hsaKmtCreateEvent.restype = HSAKMT_STATUS
    hsaKmtCreateEvent.argtypes = [ctypes.POINTER(struct__HsaEventDescriptor), ctypes.c_bool, ctypes.c_bool, ctypes.POINTER(ctypes.POINTER(struct__HsaEvent))]
except AttributeError:
    pass
try:
    hsaKmtDestroyEvent = _libraries['libhsakmt.so'].hsaKmtDestroyEvent
    hsaKmtDestroyEvent.restype = HSAKMT_STATUS
    hsaKmtDestroyEvent.argtypes = [ctypes.POINTER(struct__HsaEvent)]
except AttributeError:
    pass
try:
    hsaKmtSetEvent = _libraries['libhsakmt.so'].hsaKmtSetEvent
    hsaKmtSetEvent.restype = HSAKMT_STATUS
    hsaKmtSetEvent.argtypes = [ctypes.POINTER(struct__HsaEvent)]
except AttributeError:
    pass
try:
    hsaKmtResetEvent = _libraries['libhsakmt.so'].hsaKmtResetEvent
    hsaKmtResetEvent.restype = HSAKMT_STATUS
    hsaKmtResetEvent.argtypes = [ctypes.POINTER(struct__HsaEvent)]
except AttributeError:
    pass
try:
    hsaKmtQueryEventState = _libraries['libhsakmt.so'].hsaKmtQueryEventState
    hsaKmtQueryEventState.restype = HSAKMT_STATUS
    hsaKmtQueryEventState.argtypes = [ctypes.POINTER(struct__HsaEvent)]
except AttributeError:
    pass
try:
    hsaKmtWaitOnEvent = _libraries['libhsakmt.so'].hsaKmtWaitOnEvent
    hsaKmtWaitOnEvent.restype = HSAKMT_STATUS
    hsaKmtWaitOnEvent.argtypes = [ctypes.POINTER(struct__HsaEvent), HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtWaitOnEvent_Ext = _libraries['libhsakmt.so'].hsaKmtWaitOnEvent_Ext
    hsaKmtWaitOnEvent_Ext.restype = HSAKMT_STATUS
    hsaKmtWaitOnEvent_Ext.argtypes = [ctypes.POINTER(struct__HsaEvent), HSAuint32, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtWaitOnMultipleEvents = _libraries['libhsakmt.so'].hsaKmtWaitOnMultipleEvents
    hsaKmtWaitOnMultipleEvents.restype = HSAKMT_STATUS
    hsaKmtWaitOnMultipleEvents.argtypes = [ctypes.POINTER(struct__HsaEvent) * 0, HSAuint32, ctypes.c_bool, HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtWaitOnMultipleEvents_Ext = _libraries['libhsakmt.so'].hsaKmtWaitOnMultipleEvents_Ext
    hsaKmtWaitOnMultipleEvents_Ext.restype = HSAKMT_STATUS
    hsaKmtWaitOnMultipleEvents_Ext.argtypes = [ctypes.POINTER(struct__HsaEvent) * 0, HSAuint32, ctypes.c_bool, HSAuint32, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtReportQueue = _libraries['FIXME_STUB'].hsaKmtReportQueue
    hsaKmtReportQueue.restype = HSAKMT_STATUS
    hsaKmtReportQueue.argtypes = [HSA_QUEUEID, ctypes.POINTER(struct__HsaQueueReport)]
except AttributeError:
    pass
try:
    hsaKmtCreateQueue = _libraries['libhsakmt.so'].hsaKmtCreateQueue
    hsaKmtCreateQueue.restype = HSAKMT_STATUS
    hsaKmtCreateQueue.argtypes = [HSAuint32, HSA_QUEUE_TYPE, HSAuint32, HSA_QUEUE_PRIORITY, ctypes.POINTER(None), HSAuint64, ctypes.POINTER(struct__HsaEvent), ctypes.POINTER(struct__HsaQueueResource)]
except AttributeError:
    pass
try:
    hsaKmtUpdateQueue = _libraries['libhsakmt.so'].hsaKmtUpdateQueue
    hsaKmtUpdateQueue.restype = HSAKMT_STATUS
    hsaKmtUpdateQueue.argtypes = [HSA_QUEUEID, HSAuint32, HSA_QUEUE_PRIORITY, ctypes.POINTER(None), HSAuint64, ctypes.POINTER(struct__HsaEvent)]
except AttributeError:
    pass
try:
    hsaKmtDestroyQueue = _libraries['libhsakmt.so'].hsaKmtDestroyQueue
    hsaKmtDestroyQueue.restype = HSAKMT_STATUS
    hsaKmtDestroyQueue.argtypes = [HSA_QUEUEID]
except AttributeError:
    pass
try:
    hsaKmtSetQueueCUMask = _libraries['libhsakmt.so'].hsaKmtSetQueueCUMask
    hsaKmtSetQueueCUMask.restype = HSAKMT_STATUS
    hsaKmtSetQueueCUMask.argtypes = [HSA_QUEUEID, HSAuint32, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtGetQueueInfo = _libraries['libhsakmt.so'].hsaKmtGetQueueInfo
    hsaKmtGetQueueInfo.restype = HSAKMT_STATUS
    hsaKmtGetQueueInfo.argtypes = [HSA_QUEUEID, ctypes.POINTER(struct_HsaQueueInfo)]
except AttributeError:
    pass
try:
    hsaKmtSetMemoryPolicy = _libraries['libhsakmt.so'].hsaKmtSetMemoryPolicy
    hsaKmtSetMemoryPolicy.restype = HSAKMT_STATUS
    hsaKmtSetMemoryPolicy.argtypes = [HSAuint32, HSAuint32, HSAuint32, ctypes.POINTER(None), HSAuint64]
except AttributeError:
    pass
try:
    hsaKmtAllocMemory = _libraries['libhsakmt.so'].hsaKmtAllocMemory
    hsaKmtAllocMemory.restype = HSAKMT_STATUS
    hsaKmtAllocMemory.argtypes = [HSAuint32, HSAuint64, HsaMemFlags, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsaKmtFreeMemory = _libraries['libhsakmt.so'].hsaKmtFreeMemory
    hsaKmtFreeMemory.restype = HSAKMT_STATUS
    hsaKmtFreeMemory.argtypes = [ctypes.POINTER(None), HSAuint64]
except AttributeError:
    pass
try:
    hsaKmtAvailableMemory = _libraries['libhsakmt.so'].hsaKmtAvailableMemory
    hsaKmtAvailableMemory.restype = HSAKMT_STATUS
    hsaKmtAvailableMemory.argtypes = [HSAuint32, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtRegisterMemory = _libraries['libhsakmt.so'].hsaKmtRegisterMemory
    hsaKmtRegisterMemory.restype = HSAKMT_STATUS
    hsaKmtRegisterMemory.argtypes = [ctypes.POINTER(None), HSAuint64]
except AttributeError:
    pass
try:
    hsaKmtRegisterMemoryToNodes = _libraries['libhsakmt.so'].hsaKmtRegisterMemoryToNodes
    hsaKmtRegisterMemoryToNodes.restype = HSAKMT_STATUS
    hsaKmtRegisterMemoryToNodes.argtypes = [ctypes.POINTER(None), HSAuint64, HSAuint64, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtRegisterMemoryWithFlags = _libraries['libhsakmt.so'].hsaKmtRegisterMemoryWithFlags
    hsaKmtRegisterMemoryWithFlags.restype = HSAKMT_STATUS
    hsaKmtRegisterMemoryWithFlags.argtypes = [ctypes.POINTER(None), HSAuint64, HsaMemFlags]
except AttributeError:
    pass
try:
    hsaKmtRegisterGraphicsHandleToNodes = _libraries['libhsakmt.so'].hsaKmtRegisterGraphicsHandleToNodes
    hsaKmtRegisterGraphicsHandleToNodes.restype = HSAKMT_STATUS
    hsaKmtRegisterGraphicsHandleToNodes.argtypes = [HSAuint64, ctypes.POINTER(struct__HsaGraphicsResourceInfo), HSAuint64, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtExportDMABufHandle = _libraries['libhsakmt.so'].hsaKmtExportDMABufHandle
    hsaKmtExportDMABufHandle.restype = HSAKMT_STATUS
    hsaKmtExportDMABufHandle.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtShareMemory = _libraries['libhsakmt.so'].hsaKmtShareMemory
    hsaKmtShareMemory.restype = HSAKMT_STATUS
    hsaKmtShareMemory.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.POINTER(ctypes.c_uint32 * 8)]
except AttributeError:
    pass
try:
    hsaKmtRegisterSharedHandle = _libraries['libhsakmt.so'].hsaKmtRegisterSharedHandle
    hsaKmtRegisterSharedHandle.restype = HSAKMT_STATUS
    hsaKmtRegisterSharedHandle.argtypes = [ctypes.POINTER(ctypes.c_uint32 * 8), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtRegisterSharedHandleToNodes = _libraries['libhsakmt.so'].hsaKmtRegisterSharedHandleToNodes
    hsaKmtRegisterSharedHandleToNodes.restype = HSAKMT_STATUS
    hsaKmtRegisterSharedHandleToNodes.argtypes = [ctypes.POINTER(ctypes.c_uint32 * 8), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), HSAuint64, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtProcessVMRead = _libraries['libhsakmt.so'].hsaKmtProcessVMRead
    hsaKmtProcessVMRead.restype = HSAKMT_STATUS
    hsaKmtProcessVMRead.argtypes = [HSAuint32, ctypes.POINTER(struct__HsaMemoryRange), HSAuint64, ctypes.POINTER(struct__HsaMemoryRange), HSAuint64, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtProcessVMWrite = _libraries['libhsakmt.so'].hsaKmtProcessVMWrite
    hsaKmtProcessVMWrite.restype = HSAKMT_STATUS
    hsaKmtProcessVMWrite.argtypes = [HSAuint32, ctypes.POINTER(struct__HsaMemoryRange), HSAuint64, ctypes.POINTER(struct__HsaMemoryRange), HSAuint64, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtDeregisterMemory = _libraries['libhsakmt.so'].hsaKmtDeregisterMemory
    hsaKmtDeregisterMemory.restype = HSAKMT_STATUS
    hsaKmtDeregisterMemory.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsaKmtMapMemoryToGPU = _libraries['libhsakmt.so'].hsaKmtMapMemoryToGPU
    hsaKmtMapMemoryToGPU.restype = HSAKMT_STATUS
    hsaKmtMapMemoryToGPU.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtMapMemoryToGPUNodes = _libraries['libhsakmt.so'].hsaKmtMapMemoryToGPUNodes
    hsaKmtMapMemoryToGPUNodes.restype = HSAKMT_STATUS
    hsaKmtMapMemoryToGPUNodes.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.POINTER(ctypes.c_uint64), HsaMemMapFlags, HSAuint64, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtUnmapMemoryToGPU = _libraries['libhsakmt.so'].hsaKmtUnmapMemoryToGPU
    hsaKmtUnmapMemoryToGPU.restype = HSAKMT_STATUS
    hsaKmtUnmapMemoryToGPU.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsaKmtMapGraphicHandle = _libraries['libhsakmt.so'].hsaKmtMapGraphicHandle
    hsaKmtMapGraphicHandle.restype = HSAKMT_STATUS
    hsaKmtMapGraphicHandle.argtypes = [HSAuint32, HSAuint64, HSAuint64, HSAuint64, HSAuint64, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtUnmapGraphicHandle = _libraries['libhsakmt.so'].hsaKmtUnmapGraphicHandle
    hsaKmtUnmapGraphicHandle.restype = HSAKMT_STATUS
    hsaKmtUnmapGraphicHandle.argtypes = [HSAuint32, HSAuint64, HSAuint64]
except AttributeError:
    pass
try:
    hsaKmtGetAMDGPUDeviceHandle = _libraries['libhsakmt.so'].hsaKmtGetAMDGPUDeviceHandle
    hsaKmtGetAMDGPUDeviceHandle.restype = HSAKMT_STATUS
    hsaKmtGetAMDGPUDeviceHandle.argtypes = [HSAuint32, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsaKmtAllocQueueGWS = _libraries['libhsakmt.so'].hsaKmtAllocQueueGWS
    hsaKmtAllocQueueGWS.restype = HSAKMT_STATUS
    hsaKmtAllocQueueGWS.argtypes = [HSA_QUEUEID, HSAuint32, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtDbgRegister = _libraries['libhsakmt.so'].hsaKmtDbgRegister
    hsaKmtDbgRegister.restype = HSAKMT_STATUS
    hsaKmtDbgRegister.argtypes = [HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtDbgUnregister = _libraries['libhsakmt.so'].hsaKmtDbgUnregister
    hsaKmtDbgUnregister.restype = HSAKMT_STATUS
    hsaKmtDbgUnregister.argtypes = [HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtDbgWavefrontControl = _libraries['libhsakmt.so'].hsaKmtDbgWavefrontControl
    hsaKmtDbgWavefrontControl.restype = HSAKMT_STATUS
    hsaKmtDbgWavefrontControl.argtypes = [HSAuint32, HSA_DBG_WAVEOP, HSA_DBG_WAVEMODE, HSAuint32, ctypes.POINTER(struct__HsaDbgWaveMessage)]
except AttributeError:
    pass
try:
    hsaKmtDbgAddressWatch = _libraries['libhsakmt.so'].hsaKmtDbgAddressWatch
    hsaKmtDbgAddressWatch.restype = HSAKMT_STATUS
    hsaKmtDbgAddressWatch.argtypes = [HSAuint32, HSAuint32, _HSA_DBG_WATCH_MODE * 0, ctypes.POINTER(None) * 0, ctypes.c_uint64 * 0, ctypes.POINTER(struct__HsaEvent) * 0]
except AttributeError:
    pass
try:
    hsaKmtRuntimeEnable = _libraries['libhsakmt.so'].hsaKmtRuntimeEnable
    hsaKmtRuntimeEnable.restype = HSAKMT_STATUS
    hsaKmtRuntimeEnable.argtypes = [ctypes.POINTER(None), ctypes.c_bool]
except AttributeError:
    pass
try:
    hsaKmtRuntimeDisable = _libraries['libhsakmt.so'].hsaKmtRuntimeDisable
    hsaKmtRuntimeDisable.restype = HSAKMT_STATUS
    hsaKmtRuntimeDisable.argtypes = []
except AttributeError:
    pass
try:
    hsaKmtGetRuntimeCapabilities = _libraries['libhsakmt.so'].hsaKmtGetRuntimeCapabilities
    hsaKmtGetRuntimeCapabilities.restype = HSAKMT_STATUS
    hsaKmtGetRuntimeCapabilities.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtDbgEnable = _libraries['libhsakmt.so'].hsaKmtDbgEnable
    hsaKmtDbgEnable.restype = HSAKMT_STATUS
    hsaKmtDbgEnable.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtDbgDisable = _libraries['libhsakmt.so'].hsaKmtDbgDisable
    hsaKmtDbgDisable.restype = HSAKMT_STATUS
    hsaKmtDbgDisable.argtypes = []
except AttributeError:
    pass
try:
    hsaKmtDbgGetDeviceData = _libraries['libhsakmt.so'].hsaKmtDbgGetDeviceData
    hsaKmtDbgGetDeviceData.restype = HSAKMT_STATUS
    hsaKmtDbgGetDeviceData.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsaKmtDbgGetQueueData = _libraries['libhsakmt.so'].hsaKmtDbgGetQueueData
    hsaKmtDbgGetQueueData.restype = HSAKMT_STATUS
    hsaKmtDbgGetQueueData.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_bool]
except AttributeError:
    pass
try:
    hsaKmtCheckRuntimeDebugSupport = _libraries['FIXME_STUB'].hsaKmtCheckRuntimeDebugSupport
    hsaKmtCheckRuntimeDebugSupport.restype = HSAKMT_STATUS
    hsaKmtCheckRuntimeDebugSupport.argtypes = []
except AttributeError:
    pass
class struct_kfd_ioctl_dbg_trap_args(Structure):
    pass

try:
    hsaKmtDebugTrapIoctl = _libraries['libhsakmt.so'].hsaKmtDebugTrapIoctl
    hsaKmtDebugTrapIoctl.restype = HSAKMT_STATUS
    hsaKmtDebugTrapIoctl.argtypes = [ctypes.POINTER(struct_kfd_ioctl_dbg_trap_args), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsaKmtGetClockCounters = _libraries['libhsakmt.so'].hsaKmtGetClockCounters
    hsaKmtGetClockCounters.restype = HSAKMT_STATUS
    hsaKmtGetClockCounters.argtypes = [HSAuint32, ctypes.POINTER(struct__HsaClockCounters)]
except AttributeError:
    pass
try:
    hsaKmtPmcGetCounterProperties = _libraries['libhsakmt.so'].hsaKmtPmcGetCounterProperties
    hsaKmtPmcGetCounterProperties.restype = HSAKMT_STATUS
    hsaKmtPmcGetCounterProperties.argtypes = [HSAuint32, ctypes.POINTER(ctypes.POINTER(struct__HsaCounterProperties))]
except AttributeError:
    pass
try:
    hsaKmtPmcRegisterTrace = _libraries['libhsakmt.so'].hsaKmtPmcRegisterTrace
    hsaKmtPmcRegisterTrace.restype = HSAKMT_STATUS
    hsaKmtPmcRegisterTrace.argtypes = [HSAuint32, HSAuint32, ctypes.POINTER(struct__HsaCounter), ctypes.POINTER(struct__HsaPmcTraceRoot)]
except AttributeError:
    pass
try:
    hsaKmtPmcUnregisterTrace = _libraries['libhsakmt.so'].hsaKmtPmcUnregisterTrace
    hsaKmtPmcUnregisterTrace.restype = HSAKMT_STATUS
    hsaKmtPmcUnregisterTrace.argtypes = [HSAuint32, HSATraceId]
except AttributeError:
    pass
try:
    hsaKmtPmcAcquireTraceAccess = _libraries['libhsakmt.so'].hsaKmtPmcAcquireTraceAccess
    hsaKmtPmcAcquireTraceAccess.restype = HSAKMT_STATUS
    hsaKmtPmcAcquireTraceAccess.argtypes = [HSAuint32, HSATraceId]
except AttributeError:
    pass
try:
    hsaKmtPmcReleaseTraceAccess = _libraries['libhsakmt.so'].hsaKmtPmcReleaseTraceAccess
    hsaKmtPmcReleaseTraceAccess.restype = HSAKMT_STATUS
    hsaKmtPmcReleaseTraceAccess.argtypes = [HSAuint32, HSATraceId]
except AttributeError:
    pass
try:
    hsaKmtPmcStartTrace = _libraries['libhsakmt.so'].hsaKmtPmcStartTrace
    hsaKmtPmcStartTrace.restype = HSAKMT_STATUS
    hsaKmtPmcStartTrace.argtypes = [HSATraceId, ctypes.POINTER(None), HSAuint64]
except AttributeError:
    pass
try:
    hsaKmtPmcQueryTrace = _libraries['libhsakmt.so'].hsaKmtPmcQueryTrace
    hsaKmtPmcQueryTrace.restype = HSAKMT_STATUS
    hsaKmtPmcQueryTrace.argtypes = [HSATraceId]
except AttributeError:
    pass
try:
    hsaKmtPmcStopTrace = _libraries['libhsakmt.so'].hsaKmtPmcStopTrace
    hsaKmtPmcStopTrace.restype = HSAKMT_STATUS
    hsaKmtPmcStopTrace.argtypes = [HSATraceId]
except AttributeError:
    pass
try:
    hsaKmtSetTrapHandler = _libraries['libhsakmt.so'].hsaKmtSetTrapHandler
    hsaKmtSetTrapHandler.restype = HSAKMT_STATUS
    hsaKmtSetTrapHandler.argtypes = [HSAuint32, ctypes.POINTER(None), HSAuint64, ctypes.POINTER(None), HSAuint64]
except AttributeError:
    pass
try:
    hsaKmtGetTileConfig = _libraries['libhsakmt.so'].hsaKmtGetTileConfig
    hsaKmtGetTileConfig.restype = HSAKMT_STATUS
    hsaKmtGetTileConfig.argtypes = [HSAuint32, ctypes.POINTER(struct__HsaGpuTileConfig)]
except AttributeError:
    pass
try:
    hsaKmtQueryPointerInfo = _libraries['libhsakmt.so'].hsaKmtQueryPointerInfo
    hsaKmtQueryPointerInfo.restype = HSAKMT_STATUS
    hsaKmtQueryPointerInfo.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct__HsaPointerInfo)]
except AttributeError:
    pass
try:
    hsaKmtSetMemoryUserData = _libraries['libhsakmt.so'].hsaKmtSetMemoryUserData
    hsaKmtSetMemoryUserData.restype = HSAKMT_STATUS
    hsaKmtSetMemoryUserData.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsaKmtSPMAcquire = _libraries['libhsakmt.so'].hsaKmtSPMAcquire
    hsaKmtSPMAcquire.restype = HSAKMT_STATUS
    hsaKmtSPMAcquire.argtypes = [HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtSPMRelease = _libraries['libhsakmt.so'].hsaKmtSPMRelease
    hsaKmtSPMRelease.restype = HSAKMT_STATUS
    hsaKmtSPMRelease.argtypes = [HSAuint32]
except AttributeError:
    pass
try:
    hsaKmtSPMSetDestBuffer = _libraries['libhsakmt.so'].hsaKmtSPMSetDestBuffer
    hsaKmtSPMSetDestBuffer.restype = HSAKMT_STATUS
    hsaKmtSPMSetDestBuffer.argtypes = [HSAuint32, HSAuint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsaKmtSVMSetAttr = _libraries['libhsakmt.so'].hsaKmtSVMSetAttr
    hsaKmtSVMSetAttr.restype = HSAKMT_STATUS
    hsaKmtSVMSetAttr.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.c_uint32, ctypes.POINTER(struct__HSA_SVM_ATTRIBUTE)]
except AttributeError:
    pass
try:
    hsaKmtSVMGetAttr = _libraries['libhsakmt.so'].hsaKmtSVMGetAttr
    hsaKmtSVMGetAttr.restype = HSAKMT_STATUS
    hsaKmtSVMGetAttr.argtypes = [ctypes.POINTER(None), HSAuint64, ctypes.c_uint32, ctypes.POINTER(struct__HSA_SVM_ATTRIBUTE)]
except AttributeError:
    pass
try:
    hsaKmtSetXNACKMode = _libraries['libhsakmt.so'].hsaKmtSetXNACKMode
    hsaKmtSetXNACKMode.restype = HSAKMT_STATUS
    hsaKmtSetXNACKMode.argtypes = [HSAint32]
except AttributeError:
    pass
try:
    hsaKmtGetXNACKMode = _libraries['libhsakmt.so'].hsaKmtGetXNACKMode
    hsaKmtGetXNACKMode.restype = HSAKMT_STATUS
    hsaKmtGetXNACKMode.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hsaKmtOpenSMI = _libraries['libhsakmt.so'].hsaKmtOpenSMI
    hsaKmtOpenSMI.restype = HSAKMT_STATUS
    hsaKmtOpenSMI.argtypes = [HSAuint32, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    hsaKmtReplaceAsanHeaderPage = _libraries['libhsakmt.so'].hsaKmtReplaceAsanHeaderPage
    hsaKmtReplaceAsanHeaderPage.restype = HSAKMT_STATUS
    hsaKmtReplaceAsanHeaderPage.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsaKmtReturnAsanHeaderPage = _libraries['libhsakmt.so'].hsaKmtReturnAsanHeaderPage
    hsaKmtReturnAsanHeaderPage.restype = HSAKMT_STATUS
    hsaKmtReturnAsanHeaderPage.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
__all__ = \
    ['HSAKMT_STATUS', 'HSAKMT_STATUS_BUFFER_TOO_SMALL',
    'HSAKMT_STATUS_DRIVER_MISMATCH', 'HSAKMT_STATUS_ERROR',
    'HSAKMT_STATUS_HSAMMU_UNAVAILABLE',
    'HSAKMT_STATUS_INVALID_HANDLE', 'HSAKMT_STATUS_INVALID_NODE_UNIT',
    'HSAKMT_STATUS_INVALID_PARAMETER',
    'HSAKMT_STATUS_KERNEL_ALREADY_OPENED',
    'HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR',
    'HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED',
    'HSAKMT_STATUS_MEMORY_ALIGNMENT',
    'HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED',
    'HSAKMT_STATUS_MEMORY_NOT_REGISTERED',
    'HSAKMT_STATUS_NOT_IMPLEMENTED', 'HSAKMT_STATUS_NOT_SUPPORTED',
    'HSAKMT_STATUS_NO_MEMORY', 'HSAKMT_STATUS_OUT_OF_RESOURCES',
    'HSAKMT_STATUS_SUCCESS', 'HSAKMT_STATUS_UNAVAILABLE',
    'HSAKMT_STATUS_WAIT_FAILURE', 'HSAKMT_STATUS_WAIT_TIMEOUT',
    'HSAKMT_STATUS__enumvalues', 'HSATraceId', 'HSA_CACHING_CACHED',
    'HSA_CACHING_NONCACHED', 'HSA_CACHING_NUM_CACHING',
    'HSA_CACHING_RESERVED', 'HSA_CACHING_SIZE', 'HSA_CACHING_TYPE',
    'HSA_CACHING_TYPE__enumvalues', 'HSA_CACHING_WRITECOMBINED',
    'HSA_CAPABILITY', 'HSA_DBG_EC_DEVICE_FATAL_HALT',
    'HSA_DBG_EC_DEVICE_MEMORY_VIOLATION', 'HSA_DBG_EC_DEVICE_NEW',
    'HSA_DBG_EC_DEVICE_QUEUE_DELETE', 'HSA_DBG_EC_DEVICE_RAS_ERROR',
    'HSA_DBG_EC_MAX', 'HSA_DBG_EC_NONE',
    'HSA_DBG_EC_PROCESS_DEVICE_REMOVE', 'HSA_DBG_EC_PROCESS_RUNTIME',
    'HSA_DBG_EC_QUEUE_NEW',
    'HSA_DBG_EC_QUEUE_PACKET_DISPATCH_CODE_INVALID',
    'HSA_DBG_EC_QUEUE_PACKET_DISPATCH_DIM_INVALID',
    'HSA_DBG_EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID',
    'HSA_DBG_EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID',
    'HSA_DBG_EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID',
    'HSA_DBG_EC_QUEUE_PACKET_RESERVED',
    'HSA_DBG_EC_QUEUE_PACKET_UNSUPPORTED',
    'HSA_DBG_EC_QUEUE_PACKET_VENDOR_UNSUPPORTED',
    'HSA_DBG_EC_QUEUE_PREEMPTION_ERROR',
    'HSA_DBG_EC_QUEUE_WAVE_ABORT',
    'HSA_DBG_EC_QUEUE_WAVE_APERTURE_VIOLATION',
    'HSA_DBG_EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION',
    'HSA_DBG_EC_QUEUE_WAVE_MATH_ERROR',
    'HSA_DBG_EC_QUEUE_WAVE_MEMORY_VIOLATION',
    'HSA_DBG_EC_QUEUE_WAVE_TRAP', 'HSA_DBG_MAX_WAVEMODE',
    'HSA_DBG_MAX_WAVEMSG', 'HSA_DBG_MAX_WAVEOP',
    'HSA_DBG_NODE_CONTROL', 'HSA_DBG_NODE_CONTROL_FLAG_MAX',
    'HSA_DBG_NUM_WAVEMODE', 'HSA_DBG_NUM_WAVEMSG',
    'HSA_DBG_NUM_WAVEOP', 'HSA_DBG_TRAP_EXCEPTION_CODE',
    'HSA_DBG_TRAP_EXCEPTION_CODE__enumvalues', 'HSA_DBG_TRAP_MASK',
    'HSA_DBG_TRAP_MASK_DBG_ADDRESS_WATCH',
    'HSA_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION',
    'HSA_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO',
    'HSA_DBG_TRAP_MASK_FP_INEXACT',
    'HSA_DBG_TRAP_MASK_FP_INPUT_DENOMAL',
    'HSA_DBG_TRAP_MASK_FP_INVALID', 'HSA_DBG_TRAP_MASK_FP_OVERFLOW',
    'HSA_DBG_TRAP_MASK_FP_UNDERFLOW',
    'HSA_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO',
    'HSA_DBG_TRAP_MASK__enumvalues', 'HSA_DBG_TRAP_OVERRIDE',
    'HSA_DBG_TRAP_OVERRIDE_NUM', 'HSA_DBG_TRAP_OVERRIDE_OR',
    'HSA_DBG_TRAP_OVERRIDE_REPLACE',
    'HSA_DBG_TRAP_OVERRIDE__enumvalues', 'HSA_DBG_WATCH_ALL',
    'HSA_DBG_WATCH_ATOMIC', 'HSA_DBG_WATCH_MODE',
    'HSA_DBG_WATCH_MODE__enumvalues', 'HSA_DBG_WATCH_NONREAD',
    'HSA_DBG_WATCH_NUM', 'HSA_DBG_WATCH_READ', 'HSA_DBG_WAVEMODE',
    'HSA_DBG_WAVEMODE_BROADCAST_PROCESS',
    'HSA_DBG_WAVEMODE_BROADCAST_PROCESS_CU',
    'HSA_DBG_WAVEMODE_SINGLE', 'HSA_DBG_WAVEMODE__enumvalues',
    'HSA_DBG_WAVEMSG_AUTO', 'HSA_DBG_WAVEMSG_ERROR',
    'HSA_DBG_WAVEMSG_TYPE', 'HSA_DBG_WAVEMSG_TYPE__enumvalues',
    'HSA_DBG_WAVEMSG_USER', 'HSA_DBG_WAVEOP', 'HSA_DBG_WAVEOP_DEBUG',
    'HSA_DBG_WAVEOP_HALT', 'HSA_DBG_WAVEOP_KILL',
    'HSA_DBG_WAVEOP_RESUME', 'HSA_DBG_WAVEOP_TRAP',
    'HSA_DBG_WAVEOP__enumvalues', 'HSA_DBG_WAVE_LAUNCH_MODE',
    'HSA_DBG_WAVE_LAUNCH_MODE_DISABLE',
    'HSA_DBG_WAVE_LAUNCH_MODE_HALT', 'HSA_DBG_WAVE_LAUNCH_MODE_KILL',
    'HSA_DBG_WAVE_LAUNCH_MODE_NORMAL', 'HSA_DBG_WAVE_LAUNCH_MODE_NUM',
    'HSA_DBG_WAVE_LAUNCH_MODE_SINGLE_STEP',
    'HSA_DBG_WAVE_LAUNCH_MODE__enumvalues', 'HSA_DEBUG_EVENT_TYPE',
    'HSA_DEBUG_EVENT_TYPE_NONE', 'HSA_DEBUG_EVENT_TYPE_TRAP',
    'HSA_DEBUG_EVENT_TYPE_TRAP_VMFAULT',
    'HSA_DEBUG_EVENT_TYPE_VMFAULT',
    'HSA_DEBUG_EVENT_TYPE__enumvalues', 'HSA_DEBUG_PROPERTIES',
    'HSA_DEVICE', 'HSA_DEVICE_CPU', 'HSA_DEVICE_GPU',
    'HSA_DEVICE__enumvalues', 'HSA_ENGINE_ID', 'HSA_ENGINE_VERSION',
    'HSA_EVENTID', 'HSA_EVENTID_HW_EXCEPTION_CAUSE',
    'HSA_EVENTID_HW_EXCEPTION_CAUSE__enumvalues',
    'HSA_EVENTID_HW_EXCEPTION_ECC',
    'HSA_EVENTID_HW_EXCEPTION_GPU_HANG', 'HSA_EVENTID_MEMORYFLAGS',
    'HSA_EVENTID_MEMORYFLAGS__enumvalues',
    'HSA_EVENTID_MEMORY_FATAL_PROCESS', 'HSA_EVENTID_MEMORY_FATAL_VM',
    'HSA_EVENTID_MEMORY_RECOVERABLE', 'HSA_EVENTTIMEOUT_IMMEDIATE',
    'HSA_EVENTTIMEOUT_INFINITE', 'HSA_EVENTTYPE',
    'HSA_EVENTTYPE_DEBUG_EVENT', 'HSA_EVENTTYPE_DEVICESTATECHANGE',
    'HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS',
    'HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS__enumvalues',
    'HSA_EVENTTYPE_DEVICESTATUSCHANGE_SIZE',
    'HSA_EVENTTYPE_DEVICESTATUSCHANGE_START',
    'HSA_EVENTTYPE_DEVICESTATUSCHANGE_STOP',
    'HSA_EVENTTYPE_HW_EXCEPTION', 'HSA_EVENTTYPE_MAXID',
    'HSA_EVENTTYPE_MEMORY', 'HSA_EVENTTYPE_NODECHANGE',
    'HSA_EVENTTYPE_NODECHANGE_ADD', 'HSA_EVENTTYPE_NODECHANGE_FLAGS',
    'HSA_EVENTTYPE_NODECHANGE_FLAGS__enumvalues',
    'HSA_EVENTTYPE_NODECHANGE_REMOVE',
    'HSA_EVENTTYPE_NODECHANGE_SIZE', 'HSA_EVENTTYPE_PROFILE_EVENT',
    'HSA_EVENTTYPE_QUEUE_EVENT', 'HSA_EVENTTYPE_SIGNAL',
    'HSA_EVENTTYPE_SYSTEM_EVENT', 'HSA_EVENTTYPE_TYPE_SIZE',
    'HSA_EVENTTYPE__enumvalues', 'HSA_EVENT_TYPE',
    'HSA_EVENT_TYPE__enumvalues', 'HSA_HANDLE', 'HSA_HEAPTYPE',
    'HSA_HEAPTYPE_DEVICE_SVM', 'HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE',
    'HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC', 'HSA_HEAPTYPE_GPU_GDS',
    'HSA_HEAPTYPE_GPU_LDS', 'HSA_HEAPTYPE_GPU_SCRATCH',
    'HSA_HEAPTYPE_MMIO_REMAP', 'HSA_HEAPTYPE_NUMHEAPTYPES',
    'HSA_HEAPTYPE_SIZE', 'HSA_HEAPTYPE_SYSTEM',
    'HSA_HEAPTYPE__enumvalues', 'HSA_IOLINKTYPE',
    'HSA_IOLINKTYPE_AMBA', 'HSA_IOLINKTYPE_HYPERTRANSPORT',
    'HSA_IOLINKTYPE_MIPI', 'HSA_IOLINKTYPE_NUMIOLINKTYPES',
    'HSA_IOLINKTYPE_PCIEXPRESS', 'HSA_IOLINKTYPE_SIZE',
    'HSA_IOLINKTYPE_UNDEFINED', 'HSA_IOLINKTYPE__enumvalues',
    'HSA_IOLINK_TYPE_ETHERNET_RDMA', 'HSA_IOLINK_TYPE_GZ',
    'HSA_IOLINK_TYPE_INFINIBAND', 'HSA_IOLINK_TYPE_OTHER',
    'HSA_IOLINK_TYPE_QPI_1_1', 'HSA_IOLINK_TYPE_RAPID_IO',
    'HSA_IOLINK_TYPE_RDMA_OTHER', 'HSA_IOLINK_TYPE_RESERVED1',
    'HSA_IOLINK_TYPE_RESERVED2', 'HSA_IOLINK_TYPE_RESERVED3',
    'HSA_IOLINK_TYPE_XGMI', 'HSA_IOLINK_TYPE_XGOP',
    'HSA_LINKPROPERTY', 'HSA_MEMORYPROPERTY', 'HSA_MIGRATE_TRIGGERS',
    'HSA_MIGRATE_TRIGGERS__enumvalues',
    'HSA_MIGRATE_TRIGGER_PAGEFAULT_CPU',
    'HSA_MIGRATE_TRIGGER_PAGEFAULT_GPU',
    'HSA_MIGRATE_TRIGGER_PREFETCH',
    'HSA_MIGRATE_TRIGGER_TTM_EVICTION', 'HSA_PAGE_SIZE',
    'HSA_PAGE_SIZE_1GB', 'HSA_PAGE_SIZE_2MB', 'HSA_PAGE_SIZE_4KB',
    'HSA_PAGE_SIZE_64KB', 'HSA_PAGE_SIZE__enumvalues',
    'HSA_POINTER_ALLOCATED', 'HSA_POINTER_REGISTERED_GRAPHICS',
    'HSA_POINTER_REGISTERED_SHARED', 'HSA_POINTER_REGISTERED_USER',
    'HSA_POINTER_RESERVED_ADDR', 'HSA_POINTER_TYPE',
    'HSA_POINTER_TYPE__enumvalues', 'HSA_POINTER_UNKNOWN',
    'HSA_PROFILEBLOCK_AMD_CB', 'HSA_PROFILEBLOCK_AMD_CPF',
    'HSA_PROFILEBLOCK_AMD_CPG', 'HSA_PROFILEBLOCK_AMD_DB',
    'HSA_PROFILEBLOCK_AMD_GDS', 'HSA_PROFILEBLOCK_AMD_GRBM',
    'HSA_PROFILEBLOCK_AMD_GRBMSE', 'HSA_PROFILEBLOCK_AMD_IA',
    'HSA_PROFILEBLOCK_AMD_IOMMUV2',
    'HSA_PROFILEBLOCK_AMD_KERNEL_DRIVER', 'HSA_PROFILEBLOCK_AMD_MC',
    'HSA_PROFILEBLOCK_AMD_PASC', 'HSA_PROFILEBLOCK_AMD_PASU',
    'HSA_PROFILEBLOCK_AMD_SPI', 'HSA_PROFILEBLOCK_AMD_SQ',
    'HSA_PROFILEBLOCK_AMD_SRBM', 'HSA_PROFILEBLOCK_AMD_SX',
    'HSA_PROFILEBLOCK_AMD_TA', 'HSA_PROFILEBLOCK_AMD_TCA',
    'HSA_PROFILEBLOCK_AMD_TCC', 'HSA_PROFILEBLOCK_AMD_TCP',
    'HSA_PROFILEBLOCK_AMD_TCS', 'HSA_PROFILEBLOCK_AMD_TD',
    'HSA_PROFILEBLOCK_AMD_VGT', 'HSA_PROFILEBLOCK_AMD_WD',
    'HSA_PROFILE_TYPE', 'HSA_PROFILE_TYPE_NONPRIV_IMMEDIATE',
    'HSA_PROFILE_TYPE_NONPRIV_STREAMING', 'HSA_PROFILE_TYPE_NUM',
    'HSA_PROFILE_TYPE_PRIVILEGED_IMMEDIATE',
    'HSA_PROFILE_TYPE_PRIVILEGED_STREAMING', 'HSA_PROFILE_TYPE_SIZE',
    'HSA_PROFILE_TYPE__enumvalues', 'HSA_QUEUEID',
    'HSA_QUEUE_COMPUTE', 'HSA_QUEUE_COMPUTE_AQL',
    'HSA_QUEUE_COMPUTE_OS', 'HSA_QUEUE_DMA_AQL',
    'HSA_QUEUE_DMA_AQL_XGMI', 'HSA_QUEUE_EVICTION_CRIU_CHECKPOINT',
    'HSA_QUEUE_EVICTION_CRIU_RESTORE', 'HSA_QUEUE_EVICTION_TRIGGERS',
    'HSA_QUEUE_EVICTION_TRIGGERS__enumvalues',
    'HSA_QUEUE_EVICTION_TRIGGER_SUSPEND',
    'HSA_QUEUE_EVICTION_TRIGGER_SVM',
    'HSA_QUEUE_EVICTION_TRIGGER_TTM',
    'HSA_QUEUE_EVICTION_TRIGGER_USERPTR',
    'HSA_QUEUE_MULTIMEDIA_DECODE', 'HSA_QUEUE_MULTIMEDIA_DECODE_OS',
    'HSA_QUEUE_MULTIMEDIA_ENCODE', 'HSA_QUEUE_MULTIMEDIA_ENCODE_OS',
    'HSA_QUEUE_PRIORITY', 'HSA_QUEUE_PRIORITY_ABOVE_NORMAL',
    'HSA_QUEUE_PRIORITY_BELOW_NORMAL', 'HSA_QUEUE_PRIORITY_HIGH',
    'HSA_QUEUE_PRIORITY_LOW', 'HSA_QUEUE_PRIORITY_MAXIMUM',
    'HSA_QUEUE_PRIORITY_MINIMUM', 'HSA_QUEUE_PRIORITY_NORMAL',
    'HSA_QUEUE_PRIORITY_NUM_PRIORITY', 'HSA_QUEUE_PRIORITY_SIZE',
    'HSA_QUEUE_PRIORITY__enumvalues', 'HSA_QUEUE_SDMA',
    'HSA_QUEUE_SDMA_OS', 'HSA_QUEUE_SDMA_XGMI', 'HSA_QUEUE_TYPE',
    'HSA_QUEUE_TYPE_SIZE', 'HSA_QUEUE_TYPE__enumvalues',
    'HSA_SMI_EVENT_ALL_PROCESS', 'HSA_SMI_EVENT_GPU_POST_RESET',
    'HSA_SMI_EVENT_GPU_PRE_RESET', 'HSA_SMI_EVENT_INDEX_MAX',
    'HSA_SMI_EVENT_MIGRATE_END', 'HSA_SMI_EVENT_MIGRATE_START',
    'HSA_SMI_EVENT_NONE', 'HSA_SMI_EVENT_PAGE_FAULT_END',
    'HSA_SMI_EVENT_PAGE_FAULT_START', 'HSA_SMI_EVENT_QUEUE_EVICTION',
    'HSA_SMI_EVENT_QUEUE_RESTORE', 'HSA_SMI_EVENT_THERMAL_THROTTLE',
    'HSA_SMI_EVENT_UNMAP_FROM_GPU', 'HSA_SMI_EVENT_VMFAULT',
    'HSA_SVM_ATTRIBUTE', 'HSA_SVM_ATTR_ACCESS',
    'HSA_SVM_ATTR_ACCESS_IN_PLACE', 'HSA_SVM_ATTR_CLR_FLAGS',
    'HSA_SVM_ATTR_GRANULARITY', 'HSA_SVM_ATTR_NO_ACCESS',
    'HSA_SVM_ATTR_PREFERRED_LOC', 'HSA_SVM_ATTR_PREFETCH_LOC',
    'HSA_SVM_ATTR_SET_FLAGS', 'HSA_SVM_ATTR_TYPE',
    'HSA_SVM_ATTR_TYPE__enumvalues', 'HSA_SVM_FLAGS',
    'HSA_SVM_FLAGS__enumvalues', 'HSA_SVM_FLAG_COHERENT',
    'HSA_SVM_FLAG_EXT_COHERENT', 'HSA_SVM_FLAG_GPU_ALWAYS_MAPPED',
    'HSA_SVM_FLAG_GPU_EXEC', 'HSA_SVM_FLAG_GPU_READ_MOSTLY',
    'HSA_SVM_FLAG_GPU_RO', 'HSA_SVM_FLAG_HIVE_LOCAL',
    'HSA_SVM_FLAG_HOST_ACCESS', 'HSA_SVM_UNMAP_TRIGGERS',
    'HSA_SVM_UNMAP_TRIGGERS__enumvalues',
    'HSA_SVM_UNMAP_TRIGGER_MMU_NOTIFY',
    'HSA_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE',
    'HSA_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU', 'HSA_UUID', 'HSAint16',
    'HSAint32', 'HSAint64', 'HSAint8', 'HSAuint16', 'HSAuint32',
    'HSAuint64', 'HSAuint8', 'HsaAMDGPUDeviceHandle',
    'HsaAccessAttributeFailure', 'HsaCComputeProperties',
    'HsaCacheProperties', 'HsaCacheType', 'HsaClockCounters',
    'HsaCounter', 'HsaCounterBlockProperties', 'HsaCounterFlags',
    'HsaCounterProperties', 'HsaDbgWaveMessage',
    'HsaDbgWaveMessageAMD', 'HsaDbgWaveMsgAMDGen2',
    'HsaDeviceStateChange', 'HsaEvent', 'HsaEventData',
    'HsaEventDescriptor', 'HsaEventTimeOut',
    'HsaEventTimeOut__enumvalues', 'HsaGpuTileConfig',
    'HsaGraphicsResourceInfo', 'HsaHwException',
    'HsaIoLinkProperties', 'HsaMemFlags', 'HsaMemMapFlags',
    'HsaMemoryAccessFault', 'HsaMemoryProperties', 'HsaMemoryRange',
    'HsaNodeChange', 'HsaNodeProperties', 'HsaPmcTraceRoot',
    'HsaPointerInfo', 'HsaQueueInfo', 'HsaQueueReport',
    'HsaQueueResource', 'HsaSharedMemoryHandle', 'HsaSyncVar',
    'HsaSystemProperties', 'HsaUserContextSaveAreaHeader',
    'HsaVersionInfo', 'MAX_HSA_DEVICE', '_HSAKMT_STATUS',
    '_HSA_CACHING_TYPE', '_HSA_DBG_TRAP_EXCEPTION_CODE',
    '_HSA_DBG_TRAP_MASK', '_HSA_DBG_TRAP_OVERRIDE',
    '_HSA_DBG_WATCH_MODE', '_HSA_DBG_WAVEMODE',
    '_HSA_DBG_WAVEMSG_TYPE', '_HSA_DBG_WAVEOP',
    '_HSA_DBG_WAVE_LAUNCH_MODE', '_HSA_DEBUG_EVENT_TYPE',
    '_HSA_DEVICE', '_HSA_EVENTID_HW_EXCEPTION_CAUSE',
    '_HSA_EVENTID_MEMORYFLAGS', '_HSA_EVENTTYPE',
    '_HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS',
    '_HSA_EVENTTYPE_NODECHANGE_FLAGS', '_HSA_HEAPTYPE',
    '_HSA_IOLINKTYPE', '_HSA_MIGRATE_TRIGGERS', '_HSA_PAGE_SIZE',
    '_HSA_POINTER_TYPE', '_HSA_PROFILE_TYPE',
    '_HSA_QUEUE_EVICTION_TRIGGERS', '_HSA_QUEUE_PRIORITY',
    '_HSA_QUEUE_TYPE', '_HSA_SMI_EVENT', '_HSA_SVM_ATTR_TYPE',
    '_HSA_SVM_FLAGS', '_HSA_SVM_UNMAP_TRIGGERS', '_HsaEventTimeout',
    'hsaKmtAcquireSystemProperties', 'hsaKmtAllocMemory',
    'hsaKmtAllocQueueGWS', 'hsaKmtAvailableMemory',
    'hsaKmtCheckRuntimeDebugSupport', 'hsaKmtCloseKFD',
    'hsaKmtCreateEvent', 'hsaKmtCreateQueue', 'hsaKmtDbgAddressWatch',
    'hsaKmtDbgDisable', 'hsaKmtDbgEnable', 'hsaKmtDbgGetDeviceData',
    'hsaKmtDbgGetQueueData', 'hsaKmtDbgRegister',
    'hsaKmtDbgUnregister', 'hsaKmtDbgWavefrontControl',
    'hsaKmtDebugTrapIoctl', 'hsaKmtDeregisterMemory',
    'hsaKmtDestroyEvent', 'hsaKmtDestroyQueue',
    'hsaKmtExportDMABufHandle', 'hsaKmtFreeMemory',
    'hsaKmtGetAMDGPUDeviceHandle', 'hsaKmtGetClockCounters',
    'hsaKmtGetNodeCacheProperties', 'hsaKmtGetNodeIoLinkProperties',
    'hsaKmtGetNodeMemoryProperties', 'hsaKmtGetNodeProperties',
    'hsaKmtGetQueueInfo', 'hsaKmtGetRuntimeCapabilities',
    'hsaKmtGetTileConfig', 'hsaKmtGetVersion', 'hsaKmtGetXNACKMode',
    'hsaKmtMapGraphicHandle', 'hsaKmtMapMemoryToGPU',
    'hsaKmtMapMemoryToGPUNodes', 'hsaKmtOpenKFD', 'hsaKmtOpenSMI',
    'hsaKmtPmcAcquireTraceAccess', 'hsaKmtPmcGetCounterProperties',
    'hsaKmtPmcQueryTrace', 'hsaKmtPmcRegisterTrace',
    'hsaKmtPmcReleaseTraceAccess', 'hsaKmtPmcStartTrace',
    'hsaKmtPmcStopTrace', 'hsaKmtPmcUnregisterTrace',
    'hsaKmtProcessVMRead', 'hsaKmtProcessVMWrite',
    'hsaKmtQueryEventState', 'hsaKmtQueryPointerInfo',
    'hsaKmtRegisterGraphicsHandleToNodes', 'hsaKmtRegisterMemory',
    'hsaKmtRegisterMemoryToNodes', 'hsaKmtRegisterMemoryWithFlags',
    'hsaKmtRegisterSharedHandle', 'hsaKmtRegisterSharedHandleToNodes',
    'hsaKmtReleaseSystemProperties', 'hsaKmtReplaceAsanHeaderPage',
    'hsaKmtReportQueue', 'hsaKmtResetEvent',
    'hsaKmtReturnAsanHeaderPage', 'hsaKmtRuntimeDisable',
    'hsaKmtRuntimeEnable', 'hsaKmtSPMAcquire', 'hsaKmtSPMRelease',
    'hsaKmtSPMSetDestBuffer', 'hsaKmtSVMGetAttr', 'hsaKmtSVMSetAttr',
    'hsaKmtSetEvent', 'hsaKmtSetMemoryPolicy',
    'hsaKmtSetMemoryUserData', 'hsaKmtSetQueueCUMask',
    'hsaKmtSetTrapHandler', 'hsaKmtSetXNACKMode', 'hsaKmtShareMemory',
    'hsaKmtUnmapGraphicHandle', 'hsaKmtUnmapMemoryToGPU',
    'hsaKmtUpdateQueue', 'hsaKmtWaitOnEvent', 'hsaKmtWaitOnEvent_Ext',
    'hsaKmtWaitOnMultipleEvents', 'hsaKmtWaitOnMultipleEvents_Ext',
    'struct_HsaQueueInfo', 'struct_HsaUserContextSaveAreaHeader',
    'struct__HSA_SVM_ATTRIBUTE', 'struct__HSA_UUID',
    'struct__HaCacheProperties', 'struct__HsaAccessAttributeFailure',
    'struct__HsaCComputeProperties', 'struct__HsaClockCounters',
    'struct__HsaCounter', 'struct__HsaCounterBlockProperties',
    'struct__HsaCounterFlags', 'struct__HsaCounterProperties',
    'struct__HsaDbgWaveMessage', 'struct__HsaDbgWaveMsgAMDGen2',
    'struct__HsaDeviceStateChange', 'struct__HsaEvent',
    'struct__HsaEventData', 'struct__HsaEventDescriptor',
    'struct__HsaGpuTileConfig', 'struct__HsaGraphicsResourceInfo',
    'struct__HsaHwException', 'struct__HsaIoLinkProperties',
    'struct__HsaMemFlags', 'struct__HsaMemMapFlags',
    'struct__HsaMemoryAccessFault', 'struct__HsaMemoryProperties',
    'struct__HsaMemoryRange', 'struct__HsaNodeChange',
    'struct__HsaNodeProperties', 'struct__HsaPmcTraceRoot',
    'struct__HsaPointerInfo', 'struct__HsaQueueReport',
    'struct__HsaQueueResource', 'struct__HsaSyncVar',
    'struct__HsaSystemProperties', 'struct__HsaVersionInfo',
    'struct_kfd_ioctl_dbg_trap_args',
    'struct_struct_hsakmttypes_h_1279',
    'struct_struct_hsakmttypes_h_165',
    'struct_struct_hsakmttypes_h_177',
    'struct_struct_hsakmttypes_h_188',
    'struct_struct_hsakmttypes_h_229',
    'struct_struct_hsakmttypes_h_359',
    'struct_struct_hsakmttypes_h_380',
    'struct_struct_hsakmttypes_h_407',
    'struct_struct_hsakmttypes_h_474',
    'struct_struct_hsakmttypes_h_516',
    'struct_struct_hsakmttypes_h_581', 'union_HSA_CAPABILITY',
    'union_HSA_DEBUG_PROPERTIES', 'union_HSA_ENGINE_ID',
    'union_HSA_ENGINE_VERSION', 'union_HSA_LINKPROPERTY',
    'union_HSA_MEMORYPROPERTY', 'union_HsaCacheType',
    'union__HsaDbgWaveMessageAMD', 'union_union_hsakmttypes_h_1066',
    'union_union_hsakmttypes_h_1277', 'union_union_hsakmttypes_h_377',
    'union_union_hsakmttypes_h_514', 'union_union_hsakmttypes_h_579',
    'union_union_hsakmttypes_h_743', 'union_union_hsakmttypes_h_751',
    'union_union_hsakmttypes_h_759', 'union_union_hsakmttypes_h_972']
