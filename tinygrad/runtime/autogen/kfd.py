# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, os


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





KFD_IOCTL_H_INCLUDED = True # macro
KFD_IOCTL_MAJOR_VERSION = 1 # macro
KFD_IOCTL_MINOR_VERSION = 6 # macro
KFD_IOC_QUEUE_TYPE_COMPUTE = 0x0 # macro
KFD_IOC_QUEUE_TYPE_SDMA = 0x1 # macro
KFD_IOC_QUEUE_TYPE_COMPUTE_AQL = 0x2 # macro
KFD_IOC_QUEUE_TYPE_SDMA_XGMI = 0x3 # macro
KFD_MAX_QUEUE_PERCENTAGE = 100 # macro
KFD_MAX_QUEUE_PRIORITY = 15 # macro
KFD_IOC_CACHE_POLICY_COHERENT = 0 # macro
KFD_IOC_CACHE_POLICY_NONCOHERENT = 1 # macro
NUM_OF_SUPPORTED_GPUS = 7 # macro
MAX_ALLOWED_NUM_POINTS = 100 # macro
MAX_ALLOWED_AW_BUFF_SIZE = 4096 # macro
MAX_ALLOWED_WAC_BUFF_SIZE = 128 # macro
KFD_IOC_EVENT_SIGNAL = 0 # macro
KFD_IOC_EVENT_NODECHANGE = 1 # macro
KFD_IOC_EVENT_DEVICESTATECHANGE = 2 # macro
KFD_IOC_EVENT_HW_EXCEPTION = 3 # macro
KFD_IOC_EVENT_SYSTEM_EVENT = 4 # macro
KFD_IOC_EVENT_DEBUG_EVENT = 5 # macro
KFD_IOC_EVENT_PROFILE_EVENT = 6 # macro
KFD_IOC_EVENT_QUEUE_EVENT = 7 # macro
KFD_IOC_EVENT_MEMORY = 8 # macro
KFD_IOC_WAIT_RESULT_COMPLETE = 0 # macro
KFD_IOC_WAIT_RESULT_TIMEOUT = 1 # macro
KFD_IOC_WAIT_RESULT_FAIL = 2 # macro
KFD_SIGNAL_EVENT_LIMIT = 4096 # macro
KFD_HW_EXCEPTION_WHOLE_GPU_RESET = 0 # macro
KFD_HW_EXCEPTION_PER_ENGINE_RESET = 1 # macro
KFD_HW_EXCEPTION_GPU_HANG = 0 # macro
KFD_HW_EXCEPTION_ECC = 1 # macro
KFD_MEM_ERR_NO_RAS = 0 # macro
KFD_MEM_ERR_SRAM_ECC = 1 # macro
KFD_MEM_ERR_POISON_CONSUMED = 2 # macro
KFD_MEM_ERR_GPU_HANG = 3 # macro
KFD_IOC_ALLOC_MEM_FLAGS_VRAM = (1<<0) # macro
KFD_IOC_ALLOC_MEM_FLAGS_GTT = (1<<1) # macro
KFD_IOC_ALLOC_MEM_FLAGS_USERPTR = (1<<2) # macro
KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL = (1<<3) # macro
KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP = (1<<4) # macro
KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE = (1<<31) # macro
KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE = (1<<30) # macro
KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC = (1<<29) # macro
KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE = (1<<28) # macro
KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM = (1<<27) # macro
KFD_IOC_ALLOC_MEM_FLAGS_COHERENT = (1<<26) # macro
KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED = (1<<25) # macro
# def KFD_SMI_EVENT_MASK_FROM_INDEX(i):  # macro
#    return (1<<((i)-1))
KFD_IOCTL_SVM_FLAG_HOST_ACCESS = 0x00000001 # macro
KFD_IOCTL_SVM_FLAG_COHERENT = 0x00000002 # macro
KFD_IOCTL_SVM_FLAG_HIVE_LOCAL = 0x00000004 # macro
KFD_IOCTL_SVM_FLAG_GPU_RO = 0x00000008 # macro
KFD_IOCTL_SVM_FLAG_GPU_EXEC = 0x00000010 # macro
KFD_IOCTL_SVM_FLAG_GPU_READ_MOSTLY = 0x00000020 # macro
AMDKFD_IOCTL_BASE = 'K' # macro
# def AMDKFD_IO(nr):  # macro
#    return _IO('K',nr)
# def AMDKFD_IOR(nr, type):  # macro
#    return _IOR('K',nr,type)
# def AMDKFD_IOW(nr, type):  # macro
#    return _IOW('K',nr,type)
# def AMDKFD_IOWR(nr, type):  # macro
#    return _IOWR('K',nr,type)
# AMDKFD_IOC_GET_VERSION = _IOR('K',nr,type) ( 0x01 , struct kfd_ioctl_get_version_args ) # macro
# AMDKFD_IOC_CREATE_QUEUE = _IOWR('K',nr,type) ( 0x02 , struct kfd_ioctl_create_queue_args ) # macro
# AMDKFD_IOC_DESTROY_QUEUE = _IOWR('K',nr,type) ( 0x03 , struct kfd_ioctl_destroy_queue_args ) # macro
# AMDKFD_IOC_SET_MEMORY_POLICY = _IOW('K',nr,type) ( 0x04 , struct kfd_ioctl_set_memory_policy_args ) # macro
# AMDKFD_IOC_GET_CLOCK_COUNTERS = _IOWR('K',nr,type) ( 0x05 , struct kfd_ioctl_get_clock_counters_args ) # macro
# AMDKFD_IOC_GET_PROCESS_APERTURES = _IOR('K',nr,type) ( 0x06 , struct kfd_ioctl_get_process_apertures_args ) # macro
# AMDKFD_IOC_UPDATE_QUEUE = _IOW('K',nr,type) ( 0x07 , struct kfd_ioctl_update_queue_args ) # macro
# AMDKFD_IOC_CREATE_EVENT = _IOWR('K',nr,type) ( 0x08 , struct kfd_ioctl_create_event_args ) # macro
# AMDKFD_IOC_DESTROY_EVENT = _IOW('K',nr,type) ( 0x09 , struct kfd_ioctl_destroy_event_args ) # macro
# AMDKFD_IOC_SET_EVENT = _IOW('K',nr,type) ( 0x0A , struct kfd_ioctl_set_event_args ) # macro
# AMDKFD_IOC_RESET_EVENT = _IOW('K',nr,type) ( 0x0B , struct kfd_ioctl_reset_event_args ) # macro
# AMDKFD_IOC_WAIT_EVENTS = _IOWR('K',nr,type) ( 0x0C , struct kfd_ioctl_wait_events_args ) # macro
# AMDKFD_IOC_DBG_REGISTER = _IOW('K',nr,type) ( 0x0D , struct kfd_ioctl_dbg_register_args ) # macro
# AMDKFD_IOC_DBG_UNREGISTER = _IOW('K',nr,type) ( 0x0E , struct kfd_ioctl_dbg_unregister_args ) # macro
# AMDKFD_IOC_DBG_ADDRESS_WATCH = _IOW('K',nr,type) ( 0x0F , struct kfd_ioctl_dbg_address_watch_args ) # macro
# AMDKFD_IOC_DBG_WAVE_CONTROL = _IOW('K',nr,type) ( 0x10 , struct kfd_ioctl_dbg_wave_control_args ) # macro
# AMDKFD_IOC_SET_SCRATCH_BACKING_VA = _IOWR('K',nr,type) ( 0x11 , struct kfd_ioctl_set_scratch_backing_va_args ) # macro
# AMDKFD_IOC_GET_TILE_CONFIG = _IOWR('K',nr,type) ( 0x12 , struct kfd_ioctl_get_tile_config_args ) # macro
# AMDKFD_IOC_SET_TRAP_HANDLER = _IOW('K',nr,type) ( 0x13 , struct kfd_ioctl_set_trap_handler_args ) # macro
# AMDKFD_IOC_GET_PROCESS_APERTURES_NEW = _IOWR('K',nr,type) ( 0x14 , struct kfd_ioctl_get_process_apertures_new_args ) # macro
# AMDKFD_IOC_ACQUIRE_VM = _IOW('K',nr,type) ( 0x15 , struct kfd_ioctl_acquire_vm_args ) # macro
# AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = _IOWR('K',nr,type) ( 0x16 , struct kfd_ioctl_alloc_memory_of_gpu_args ) # macro
# AMDKFD_IOC_FREE_MEMORY_OF_GPU = _IOW('K',nr,type) ( 0x17 , struct kfd_ioctl_free_memory_of_gpu_args ) # macro
# AMDKFD_IOC_MAP_MEMORY_TO_GPU = _IOWR('K',nr,type) ( 0x18 , struct kfd_ioctl_map_memory_to_gpu_args ) # macro
# AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU = _IOWR('K',nr,type) ( 0x19 , struct kfd_ioctl_unmap_memory_from_gpu_args ) # macro
# AMDKFD_IOC_SET_CU_MASK = _IOW('K',nr,type) ( 0x1A , struct kfd_ioctl_set_cu_mask_args ) # macro
# AMDKFD_IOC_GET_QUEUE_WAVE_STATE = _IOWR('K',nr,type) ( 0x1B , struct kfd_ioctl_get_queue_wave_state_args ) # macro
# AMDKFD_IOC_GET_DMABUF_INFO = _IOWR('K',nr,type) ( 0x1C , struct kfd_ioctl_get_dmabuf_info_args ) # macro
# AMDKFD_IOC_IMPORT_DMABUF = _IOWR('K',nr,type) ( 0x1D , struct kfd_ioctl_import_dmabuf_args ) # macro
# AMDKFD_IOC_ALLOC_QUEUE_GWS = _IOWR('K',nr,type) ( 0x1E , struct kfd_ioctl_alloc_queue_gws_args ) # macro
# AMDKFD_IOC_SMI_EVENTS = _IOWR('K',nr,type) ( 0x1F , struct kfd_ioctl_smi_events_args ) # macro
# AMDKFD_IOC_SVM = _IOWR('K',nr,type) ( 0x20 , struct kfd_ioctl_svm_args ) # macro
# AMDKFD_IOC_SET_XNACK_MODE = _IOWR('K',nr,type) ( 0x21 , struct kfd_ioctl_set_xnack_mode_args ) # macro
AMDKFD_COMMAND_START = 0x01 # macro
AMDKFD_COMMAND_END = 0x22 # macro
class struct_kfd_ioctl_get_version_args(Structure):
    pass

struct_kfd_ioctl_get_version_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_version_args._fields_ = [
    ('major_version', ctypes.c_uint32),
    ('minor_version', ctypes.c_uint32),
]

class struct_kfd_ioctl_create_queue_args(Structure):
    pass

struct_kfd_ioctl_create_queue_args._pack_ = 1 # source:False
struct_kfd_ioctl_create_queue_args._fields_ = [
    ('ring_base_address', ctypes.c_uint64),
    ('write_pointer_address', ctypes.c_uint64),
    ('read_pointer_address', ctypes.c_uint64),
    ('doorbell_offset', ctypes.c_uint64),
    ('ring_size', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('queue_type', ctypes.c_uint32),
    ('queue_percentage', ctypes.c_uint32),
    ('queue_priority', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
    ('eop_buffer_address', ctypes.c_uint64),
    ('eop_buffer_size', ctypes.c_uint64),
    ('ctx_save_restore_address', ctypes.c_uint64),
    ('ctx_save_restore_size', ctypes.c_uint32),
    ('ctl_stack_size', ctypes.c_uint32),
]

class struct_kfd_ioctl_destroy_queue_args(Structure):
    pass

struct_kfd_ioctl_destroy_queue_args._pack_ = 1 # source:False
struct_kfd_ioctl_destroy_queue_args._fields_ = [
    ('queue_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_update_queue_args(Structure):
    pass

struct_kfd_ioctl_update_queue_args._pack_ = 1 # source:False
struct_kfd_ioctl_update_queue_args._fields_ = [
    ('ring_base_address', ctypes.c_uint64),
    ('queue_id', ctypes.c_uint32),
    ('ring_size', ctypes.c_uint32),
    ('queue_percentage', ctypes.c_uint32),
    ('queue_priority', ctypes.c_uint32),
]

class struct_kfd_ioctl_set_cu_mask_args(Structure):
    pass

struct_kfd_ioctl_set_cu_mask_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_cu_mask_args._fields_ = [
    ('queue_id', ctypes.c_uint32),
    ('num_cu_mask', ctypes.c_uint32),
    ('cu_mask_ptr', ctypes.c_uint64),
]

class struct_kfd_ioctl_get_queue_wave_state_args(Structure):
    pass

struct_kfd_ioctl_get_queue_wave_state_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_queue_wave_state_args._fields_ = [
    ('ctl_stack_address', ctypes.c_uint64),
    ('ctl_stack_used_size', ctypes.c_uint32),
    ('save_area_used_size', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_set_memory_policy_args(Structure):
    pass

struct_kfd_ioctl_set_memory_policy_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_memory_policy_args._fields_ = [
    ('alternate_aperture_base', ctypes.c_uint64),
    ('alternate_aperture_size', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('default_policy', ctypes.c_uint32),
    ('alternate_policy', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_get_clock_counters_args(Structure):
    pass

struct_kfd_ioctl_get_clock_counters_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_clock_counters_args._fields_ = [
    ('gpu_clock_counter', ctypes.c_uint64),
    ('cpu_clock_counter', ctypes.c_uint64),
    ('system_clock_counter', ctypes.c_uint64),
    ('system_clock_freq', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_process_device_apertures(Structure):
    pass

struct_kfd_process_device_apertures._pack_ = 1 # source:False
struct_kfd_process_device_apertures._fields_ = [
    ('lds_base', ctypes.c_uint64),
    ('lds_limit', ctypes.c_uint64),
    ('scratch_base', ctypes.c_uint64),
    ('scratch_limit', ctypes.c_uint64),
    ('gpuvm_base', ctypes.c_uint64),
    ('gpuvm_limit', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_get_process_apertures_args(Structure):
    pass

struct_kfd_ioctl_get_process_apertures_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_process_apertures_args._fields_ = [
    ('process_apertures', struct_kfd_process_device_apertures * 7),
    ('num_of_nodes', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_get_process_apertures_new_args(Structure):
    pass

struct_kfd_ioctl_get_process_apertures_new_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_process_apertures_new_args._fields_ = [
    ('kfd_process_device_apertures_ptr', ctypes.c_uint64),
    ('num_of_nodes', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_register_args(Structure):
    pass

struct_kfd_ioctl_dbg_register_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_register_args._fields_ = [
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_unregister_args(Structure):
    pass

struct_kfd_ioctl_dbg_unregister_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_unregister_args._fields_ = [
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_address_watch_args(Structure):
    pass

struct_kfd_ioctl_dbg_address_watch_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_address_watch_args._fields_ = [
    ('content_ptr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('buf_size_in_bytes', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_wave_control_args(Structure):
    pass

struct_kfd_ioctl_dbg_wave_control_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_wave_control_args._fields_ = [
    ('content_ptr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('buf_size_in_bytes', ctypes.c_uint32),
]

class struct_kfd_ioctl_create_event_args(Structure):
    pass

struct_kfd_ioctl_create_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_create_event_args._fields_ = [
    ('event_page_offset', ctypes.c_uint64),
    ('event_trigger_data', ctypes.c_uint32),
    ('event_type', ctypes.c_uint32),
    ('auto_reset', ctypes.c_uint32),
    ('node_id', ctypes.c_uint32),
    ('event_id', ctypes.c_uint32),
    ('event_slot_index', ctypes.c_uint32),
]

class struct_kfd_ioctl_destroy_event_args(Structure):
    pass

struct_kfd_ioctl_destroy_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_destroy_event_args._fields_ = [
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_set_event_args(Structure):
    pass

struct_kfd_ioctl_set_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_event_args._fields_ = [
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_reset_event_args(Structure):
    pass

struct_kfd_ioctl_reset_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_reset_event_args._fields_ = [
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_memory_exception_failure(Structure):
    pass

struct_kfd_memory_exception_failure._pack_ = 1 # source:False
struct_kfd_memory_exception_failure._fields_ = [
    ('NotPresent', ctypes.c_uint32),
    ('ReadOnly', ctypes.c_uint32),
    ('NoExecute', ctypes.c_uint32),
    ('imprecise', ctypes.c_uint32),
]

class struct_kfd_hsa_memory_exception_data(Structure):
    pass

struct_kfd_hsa_memory_exception_data._pack_ = 1 # source:False
struct_kfd_hsa_memory_exception_data._fields_ = [
    ('failure', struct_kfd_memory_exception_failure),
    ('va', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('ErrorType', ctypes.c_uint32),
]

class struct_kfd_hsa_hw_exception_data(Structure):
    pass

struct_kfd_hsa_hw_exception_data._pack_ = 1 # source:False
struct_kfd_hsa_hw_exception_data._fields_ = [
    ('reset_type', ctypes.c_uint32),
    ('reset_cause', ctypes.c_uint32),
    ('memory_lost', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
]

class struct_kfd_event_data(Structure):
    pass

class union_kfd_event_data_0(Union):
    pass

union_kfd_event_data_0._pack_ = 1 # source:False
union_kfd_event_data_0._fields_ = [
    ('memory_exception_data', struct_kfd_hsa_memory_exception_data),
    ('hw_exception_data', struct_kfd_hsa_hw_exception_data),
    ('PADDING_0', ctypes.c_ubyte * 16),
]

struct_kfd_event_data._pack_ = 1 # source:False
struct_kfd_event_data._anonymous_ = ('_0',)
struct_kfd_event_data._fields_ = [
    ('_0', union_kfd_event_data_0),
    ('kfd_event_data_ext', ctypes.c_uint64),
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_wait_events_args(Structure):
    pass

struct_kfd_ioctl_wait_events_args._pack_ = 1 # source:False
struct_kfd_ioctl_wait_events_args._fields_ = [
    ('events_ptr', ctypes.c_uint64),
    ('num_events', ctypes.c_uint32),
    ('wait_for_all', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
    ('wait_result', ctypes.c_uint32),
]

class struct_kfd_ioctl_set_scratch_backing_va_args(Structure):
    pass

struct_kfd_ioctl_set_scratch_backing_va_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_scratch_backing_va_args._fields_ = [
    ('va_addr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_get_tile_config_args(Structure):
    pass

struct_kfd_ioctl_get_tile_config_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_tile_config_args._fields_ = [
    ('tile_config_ptr', ctypes.c_uint64),
    ('macro_tile_config_ptr', ctypes.c_uint64),
    ('num_tile_configs', ctypes.c_uint32),
    ('num_macro_tile_configs', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('gb_addr_config', ctypes.c_uint32),
    ('num_banks', ctypes.c_uint32),
    ('num_ranks', ctypes.c_uint32),
]

class struct_kfd_ioctl_set_trap_handler_args(Structure):
    pass

struct_kfd_ioctl_set_trap_handler_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_trap_handler_args._fields_ = [
    ('tba_addr', ctypes.c_uint64),
    ('tma_addr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_acquire_vm_args(Structure):
    pass

struct_kfd_ioctl_acquire_vm_args._pack_ = 1 # source:False
struct_kfd_ioctl_acquire_vm_args._fields_ = [
    ('drm_fd', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
]

class struct_kfd_ioctl_alloc_memory_of_gpu_args(Structure):
    pass

struct_kfd_ioctl_alloc_memory_of_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_alloc_memory_of_gpu_args._fields_ = [
    ('va_addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('handle', ctypes.c_uint64),
    ('mmap_offset', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_kfd_ioctl_free_memory_of_gpu_args(Structure):
    pass

struct_kfd_ioctl_free_memory_of_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_free_memory_of_gpu_args._fields_ = [
    ('handle', ctypes.c_uint64),
]

class struct_kfd_ioctl_map_memory_to_gpu_args(Structure):
    pass

struct_kfd_ioctl_map_memory_to_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_map_memory_to_gpu_args._fields_ = [
    ('handle', ctypes.c_uint64),
    ('device_ids_array_ptr', ctypes.c_uint64),
    ('n_devices', ctypes.c_uint32),
    ('n_success', ctypes.c_uint32),
]

class struct_kfd_ioctl_unmap_memory_from_gpu_args(Structure):
    pass

struct_kfd_ioctl_unmap_memory_from_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_unmap_memory_from_gpu_args._fields_ = [
    ('handle', ctypes.c_uint64),
    ('device_ids_array_ptr', ctypes.c_uint64),
    ('n_devices', ctypes.c_uint32),
    ('n_success', ctypes.c_uint32),
]

class struct_kfd_ioctl_alloc_queue_gws_args(Structure):
    pass

struct_kfd_ioctl_alloc_queue_gws_args._pack_ = 1 # source:False
struct_kfd_ioctl_alloc_queue_gws_args._fields_ = [
    ('queue_id', ctypes.c_uint32),
    ('num_gws', ctypes.c_uint32),
    ('first_gws', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_get_dmabuf_info_args(Structure):
    pass

struct_kfd_ioctl_get_dmabuf_info_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_dmabuf_info_args._fields_ = [
    ('size', ctypes.c_uint64),
    ('metadata_ptr', ctypes.c_uint64),
    ('metadata_size', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('dmabuf_fd', ctypes.c_uint32),
]

class struct_kfd_ioctl_import_dmabuf_args(Structure):
    pass

struct_kfd_ioctl_import_dmabuf_args._pack_ = 1 # source:False
struct_kfd_ioctl_import_dmabuf_args._fields_ = [
    ('va_addr', ctypes.c_uint64),
    ('handle', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('dmabuf_fd', ctypes.c_uint32),
]


# values for enumeration 'kfd_smi_event'
kfd_smi_event__enumvalues = {
    0: 'KFD_SMI_EVENT_NONE',
    1: 'KFD_SMI_EVENT_VMFAULT',
    2: 'KFD_SMI_EVENT_THERMAL_THROTTLE',
    3: 'KFD_SMI_EVENT_GPU_PRE_RESET',
    4: 'KFD_SMI_EVENT_GPU_POST_RESET',
}
KFD_SMI_EVENT_NONE = 0
KFD_SMI_EVENT_VMFAULT = 1
KFD_SMI_EVENT_THERMAL_THROTTLE = 2
KFD_SMI_EVENT_GPU_PRE_RESET = 3
KFD_SMI_EVENT_GPU_POST_RESET = 4
kfd_smi_event = ctypes.c_uint32 # enum
class struct_kfd_ioctl_smi_events_args(Structure):
    pass

struct_kfd_ioctl_smi_events_args._pack_ = 1 # source:False
struct_kfd_ioctl_smi_events_args._fields_ = [
    ('gpuid', ctypes.c_uint32),
    ('anon_fd', ctypes.c_uint32),
]


# values for enumeration 'kfd_mmio_remap'
kfd_mmio_remap__enumvalues = {
    0: 'KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL',
    4: 'KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL',
}
KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL = 0
KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL = 4
kfd_mmio_remap = ctypes.c_uint32 # enum

# values for enumeration 'kfd_ioctl_svm_op'
kfd_ioctl_svm_op__enumvalues = {
    0: 'KFD_IOCTL_SVM_OP_SET_ATTR',
    1: 'KFD_IOCTL_SVM_OP_GET_ATTR',
}
KFD_IOCTL_SVM_OP_SET_ATTR = 0
KFD_IOCTL_SVM_OP_GET_ATTR = 1
kfd_ioctl_svm_op = ctypes.c_uint32 # enum

# values for enumeration 'kfd_ioctl_svm_location'
kfd_ioctl_svm_location__enumvalues = {
    0: 'KFD_IOCTL_SVM_LOCATION_SYSMEM',
    4294967295: 'KFD_IOCTL_SVM_LOCATION_UNDEFINED',
}
KFD_IOCTL_SVM_LOCATION_SYSMEM = 0
KFD_IOCTL_SVM_LOCATION_UNDEFINED = 4294967295
kfd_ioctl_svm_location = ctypes.c_uint32 # enum

# values for enumeration 'kfd_ioctl_svm_attr_type'
kfd_ioctl_svm_attr_type__enumvalues = {
    0: 'KFD_IOCTL_SVM_ATTR_PREFERRED_LOC',
    1: 'KFD_IOCTL_SVM_ATTR_PREFETCH_LOC',
    2: 'KFD_IOCTL_SVM_ATTR_ACCESS',
    3: 'KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE',
    4: 'KFD_IOCTL_SVM_ATTR_NO_ACCESS',
    5: 'KFD_IOCTL_SVM_ATTR_SET_FLAGS',
    6: 'KFD_IOCTL_SVM_ATTR_CLR_FLAGS',
    7: 'KFD_IOCTL_SVM_ATTR_GRANULARITY',
}
KFD_IOCTL_SVM_ATTR_PREFERRED_LOC = 0
KFD_IOCTL_SVM_ATTR_PREFETCH_LOC = 1
KFD_IOCTL_SVM_ATTR_ACCESS = 2
KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE = 3
KFD_IOCTL_SVM_ATTR_NO_ACCESS = 4
KFD_IOCTL_SVM_ATTR_SET_FLAGS = 5
KFD_IOCTL_SVM_ATTR_CLR_FLAGS = 6
KFD_IOCTL_SVM_ATTR_GRANULARITY = 7
kfd_ioctl_svm_attr_type = ctypes.c_uint32 # enum
class struct_kfd_ioctl_svm_attribute(Structure):
    pass

struct_kfd_ioctl_svm_attribute._pack_ = 1 # source:False
struct_kfd_ioctl_svm_attribute._fields_ = [
    ('type', ctypes.c_uint32),
    ('value', ctypes.c_uint32),
]

class struct_kfd_ioctl_svm_args(Structure):
    pass

struct_kfd_ioctl_svm_args._pack_ = 1 # source:False
struct_kfd_ioctl_svm_args._fields_ = [
    ('start_addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('op', ctypes.c_uint32),
    ('nattr', ctypes.c_uint32),
    ('attrs', struct_kfd_ioctl_svm_attribute * 0),
]

class struct_kfd_ioctl_set_xnack_mode_args(Structure):
    pass

struct_kfd_ioctl_set_xnack_mode_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_xnack_mode_args._fields_ = [
    ('xnack_enabled', ctypes.c_int32),
]

AMD_HSA_KERNEL_CODE_H = True # macro
AMD_CONTROL_DIRECTIVES_ALIGN_BYTES = 64 # macro
# AMD_CONTROL_DIRECTIVES_ALIGN = __ALIGNED__ ( 64 ) # macro
AMD_ISA_ALIGN_BYTES = 256 # macro
AMD_KERNEL_CODE_ALIGN_BYTES = 64 # macro
# AMD_KERNEL_CODE_ALIGN = __ALIGNED__ ( 64 ) # macro
amd_kernel_code_version32_t = ctypes.c_uint32

# values for enumeration 'amd_kernel_code_version_t'
amd_kernel_code_version_t__enumvalues = {
    1: 'AMD_KERNEL_CODE_VERSION_MAJOR',
    1: 'AMD_KERNEL_CODE_VERSION_MINOR',
}
AMD_KERNEL_CODE_VERSION_MAJOR = 1
AMD_KERNEL_CODE_VERSION_MINOR = 1
amd_kernel_code_version_t = ctypes.c_uint32 # enum
amd_machine_kind16_t = ctypes.c_uint16

# values for enumeration 'amd_machine_kind_t'
amd_machine_kind_t__enumvalues = {
    0: 'AMD_MACHINE_KIND_UNDEFINED',
    1: 'AMD_MACHINE_KIND_AMDGPU',
}
AMD_MACHINE_KIND_UNDEFINED = 0
AMD_MACHINE_KIND_AMDGPU = 1
amd_machine_kind_t = ctypes.c_uint32 # enum
amd_machine_version16_t = ctypes.c_uint16

# values for enumeration 'amd_float_round_mode_t'
amd_float_round_mode_t__enumvalues = {
    0: 'AMD_FLOAT_ROUND_MODE_NEAREST_EVEN',
    1: 'AMD_FLOAT_ROUND_MODE_PLUS_INFINITY',
    2: 'AMD_FLOAT_ROUND_MODE_MINUS_INFINITY',
    3: 'AMD_FLOAT_ROUND_MODE_ZERO',
}
AMD_FLOAT_ROUND_MODE_NEAREST_EVEN = 0
AMD_FLOAT_ROUND_MODE_PLUS_INFINITY = 1
AMD_FLOAT_ROUND_MODE_MINUS_INFINITY = 2
AMD_FLOAT_ROUND_MODE_ZERO = 3
amd_float_round_mode_t = ctypes.c_uint32 # enum

# values for enumeration 'amd_float_denorm_mode_t'
amd_float_denorm_mode_t__enumvalues = {
    0: 'AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT',
    1: 'AMD_FLOAT_DENORM_MODE_FLUSH_OUTPUT',
    2: 'AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE',
    3: 'AMD_FLOAT_DENORM_MODE_NO_FLUSH',
}
AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT = 0
AMD_FLOAT_DENORM_MODE_FLUSH_OUTPUT = 1
AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE = 2
AMD_FLOAT_DENORM_MODE_NO_FLUSH = 3
amd_float_denorm_mode_t = ctypes.c_uint32 # enum
amd_compute_pgm_rsrc_one32_t = ctypes.c_uint32

# values for enumeration 'amd_compute_pgm_rsrc_one_t'
amd_compute_pgm_rsrc_one_t__enumvalues = {
    0: 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT',
    6: 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH',
    63: 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT',
    6: 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT',
    4: 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH',
    960: 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT',
    10: 'AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_SHIFT',
    2: 'AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_WIDTH',
    3072: 'AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY',
    12: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_SHIFT',
    2: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_WIDTH',
    12288: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32',
    14: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_SHIFT',
    2: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_WIDTH',
    49152: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64',
    16: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_SHIFT',
    2: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_WIDTH',
    196608: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32',
    18: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT',
    2: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_WIDTH',
    786432: 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64',
    20: 'AMD_COMPUTE_PGM_RSRC_ONE_PRIV_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_ONE_PRIV_WIDTH',
    1048576: 'AMD_COMPUTE_PGM_RSRC_ONE_PRIV',
    21: 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_WIDTH',
    2097152: 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP',
    22: 'AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_WIDTH',
    4194304: 'AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE',
    23: 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_WIDTH',
    8388608: 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE',
    24: 'AMD_COMPUTE_PGM_RSRC_ONE_BULKY_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_ONE_BULKY_WIDTH',
    16777216: 'AMD_COMPUTE_PGM_RSRC_ONE_BULKY',
    25: 'AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_WIDTH',
    33554432: 'AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER',
    26: 'AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT',
    6: 'AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_WIDTH',
    -67108864: 'AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1',
}
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT = 0
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH = 6
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT = 63
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT = 6
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH = 4
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT = 960
AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_SHIFT = 10
AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_WIDTH = 2
AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY = 3072
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_SHIFT = 12
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_WIDTH = 2
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32 = 12288
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_SHIFT = 14
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_WIDTH = 2
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64 = 49152
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_SHIFT = 16
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_WIDTH = 2
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32 = 196608
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT = 18
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_WIDTH = 2
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64 = 786432
AMD_COMPUTE_PGM_RSRC_ONE_PRIV_SHIFT = 20
AMD_COMPUTE_PGM_RSRC_ONE_PRIV_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_ONE_PRIV = 1048576
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_SHIFT = 21
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP = 2097152
AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_SHIFT = 22
AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE = 4194304
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_SHIFT = 23
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE = 8388608
AMD_COMPUTE_PGM_RSRC_ONE_BULKY_SHIFT = 24
AMD_COMPUTE_PGM_RSRC_ONE_BULKY_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_ONE_BULKY = 16777216
AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_SHIFT = 25
AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER = 33554432
AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT = 26
AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_WIDTH = 6
AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1 = -67108864
amd_compute_pgm_rsrc_one_t = ctypes.c_int32 # enum

# values for enumeration 'amd_system_vgpr_workitem_id_t'
amd_system_vgpr_workitem_id_t__enumvalues = {
    0: 'AMD_SYSTEM_VGPR_WORKITEM_ID_X',
    1: 'AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y',
    2: 'AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z',
    3: 'AMD_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED',
}
AMD_SYSTEM_VGPR_WORKITEM_ID_X = 0
AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y = 1
AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z = 2
AMD_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED = 3
amd_system_vgpr_workitem_id_t = ctypes.c_uint32 # enum
amd_compute_pgm_rsrc_two32_t = ctypes.c_uint32

# values for enumeration 'amd_compute_pgm_rsrc_two_t'
amd_compute_pgm_rsrc_two_t__enumvalues = {
    0: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_WIDTH',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT',
    5: 'AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_WIDTH',
    62: 'AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT',
    6: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_WIDTH',
    64: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER',
    7: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_WIDTH',
    128: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X',
    8: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_WIDTH',
    256: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y',
    9: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_WIDTH',
    512: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z',
    10: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_WIDTH',
    1024: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO',
    11: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_SHIFT',
    2: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_WIDTH',
    6144: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID',
    13: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_WIDTH',
    8192: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH',
    14: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_WIDTH',
    16384: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION',
    15: 'AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT',
    9: 'AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_WIDTH',
    16744448: 'AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE',
    24: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_WIDTH',
    16777216: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION',
    25: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_WIDTH',
    33554432: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE',
    26: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_WIDTH',
    67108864: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO',
    27: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_WIDTH',
    134217728: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW',
    28: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_WIDTH',
    268435456: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW',
    29: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_WIDTH',
    536870912: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT',
    30: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_WIDTH',
    1073741824: 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO',
    31: 'AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_SHIFT',
    1: 'AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_WIDTH',
    -2147483648: 'AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1',
}
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_SHIFT = 0
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = 1
AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT = 1
AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_WIDTH = 5
AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT = 62
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_SHIFT = 6
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER = 64
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT = 7
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X = 128
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT = 8
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y = 256
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT = 9
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z = 512
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_SHIFT = 10
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO = 1024
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_SHIFT = 11
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_WIDTH = 2
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID = 6144
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_SHIFT = 13
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH = 8192
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_SHIFT = 14
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION = 16384
AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT = 15
AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_WIDTH = 9
AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE = 16744448
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_SHIFT = 24
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION = 16777216
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_SHIFT = 25
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE = 33554432
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_SHIFT = 26
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO = 67108864
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_SHIFT = 27
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW = 134217728
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_SHIFT = 28
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW = 268435456
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_SHIFT = 29
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT = 536870912
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_SHIFT = 30
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO = 1073741824
AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_SHIFT = 31
AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_WIDTH = 1
AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1 = -2147483648
amd_compute_pgm_rsrc_two_t = ctypes.c_int32 # enum

# values for enumeration 'amd_element_byte_size_t'
amd_element_byte_size_t__enumvalues = {
    0: 'AMD_ELEMENT_BYTE_SIZE_2',
    1: 'AMD_ELEMENT_BYTE_SIZE_4',
    2: 'AMD_ELEMENT_BYTE_SIZE_8',
    3: 'AMD_ELEMENT_BYTE_SIZE_16',
}
AMD_ELEMENT_BYTE_SIZE_2 = 0
AMD_ELEMENT_BYTE_SIZE_4 = 1
AMD_ELEMENT_BYTE_SIZE_8 = 2
AMD_ELEMENT_BYTE_SIZE_16 = 3
amd_element_byte_size_t = ctypes.c_uint32 # enum
amd_kernel_code_properties32_t = ctypes.c_uint32

# values for enumeration 'amd_kernel_code_properties_t'
amd_kernel_code_properties_t__enumvalues = {
    0: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_WIDTH',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_WIDTH',
    2: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR',
    2: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_WIDTH',
    4: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR',
    3: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_WIDTH',
    8: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR',
    4: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_WIDTH',
    16: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID',
    5: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_WIDTH',
    32: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT',
    6: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_WIDTH',
    64: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE',
    7: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_WIDTH',
    128: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X',
    8: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_WIDTH',
    256: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y',
    9: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_WIDTH',
    512: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z',
    10: 'AMD_KERNEL_CODE_PROPERTIES_RESERVED1_SHIFT',
    6: 'AMD_KERNEL_CODE_PROPERTIES_RESERVED1_WIDTH',
    64512: 'AMD_KERNEL_CODE_PROPERTIES_RESERVED1',
    16: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_WIDTH',
    65536: 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS',
    17: 'AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_SHIFT',
    2: 'AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_WIDTH',
    393216: 'AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE',
    19: 'AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_WIDTH',
    524288: 'AMD_KERNEL_CODE_PROPERTIES_IS_PTR64',
    20: 'AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_WIDTH',
    1048576: 'AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK',
    21: 'AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_WIDTH',
    2097152: 'AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED',
    22: 'AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_SHIFT',
    1: 'AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_WIDTH',
    4194304: 'AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED',
    23: 'AMD_KERNEL_CODE_PROPERTIES_RESERVED2_SHIFT',
    9: 'AMD_KERNEL_CODE_PROPERTIES_RESERVED2_WIDTH',
    -8388608: 'AMD_KERNEL_CODE_PROPERTIES_RESERVED2',
}
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT = 0
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_SHIFT = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR = 2
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_SHIFT = 2
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR = 4
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT = 3
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR = 8
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_SHIFT = 4
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID = 16
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_SHIFT = 5
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT = 32
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT = 6
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE = 64
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_SHIFT = 7
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X = 128
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_SHIFT = 8
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y = 256
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_SHIFT = 9
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z = 512
AMD_KERNEL_CODE_PROPERTIES_RESERVED1_SHIFT = 10
AMD_KERNEL_CODE_PROPERTIES_RESERVED1_WIDTH = 6
AMD_KERNEL_CODE_PROPERTIES_RESERVED1 = 64512
AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_SHIFT = 16
AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS = 65536
AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_SHIFT = 17
AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_WIDTH = 2
AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE = 393216
AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_SHIFT = 19
AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_IS_PTR64 = 524288
AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT = 20
AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK = 1048576
AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_SHIFT = 21
AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED = 2097152
AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_SHIFT = 22
AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_WIDTH = 1
AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED = 4194304
AMD_KERNEL_CODE_PROPERTIES_RESERVED2_SHIFT = 23
AMD_KERNEL_CODE_PROPERTIES_RESERVED2_WIDTH = 9
AMD_KERNEL_CODE_PROPERTIES_RESERVED2 = -8388608
amd_kernel_code_properties_t = ctypes.c_int32 # enum
amd_powertwo8_t = ctypes.c_ubyte

# values for enumeration 'amd_powertwo_t'
amd_powertwo_t__enumvalues = {
    0: 'AMD_POWERTWO_1',
    1: 'AMD_POWERTWO_2',
    2: 'AMD_POWERTWO_4',
    3: 'AMD_POWERTWO_8',
    4: 'AMD_POWERTWO_16',
    5: 'AMD_POWERTWO_32',
    6: 'AMD_POWERTWO_64',
    7: 'AMD_POWERTWO_128',
    8: 'AMD_POWERTWO_256',
}
AMD_POWERTWO_1 = 0
AMD_POWERTWO_2 = 1
AMD_POWERTWO_4 = 2
AMD_POWERTWO_8 = 3
AMD_POWERTWO_16 = 4
AMD_POWERTWO_32 = 5
AMD_POWERTWO_64 = 6
AMD_POWERTWO_128 = 7
AMD_POWERTWO_256 = 8
amd_powertwo_t = ctypes.c_uint32 # enum
amd_enabled_control_directive64_t = ctypes.c_uint64

# values for enumeration 'amd_enabled_control_directive_t'
amd_enabled_control_directive_t__enumvalues = {
    1: 'AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS',
    2: 'AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS',
    4: 'AMD_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE',
    8: 'AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE',
    16: 'AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE',
    32: 'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM',
    64: 'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE',
    128: 'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE',
    256: 'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS',
}
AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS = 1
AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS = 2
AMD_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE = 4
AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE = 8
AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE = 16
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM = 32
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE = 64
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE = 128
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS = 256
amd_enabled_control_directive_t = ctypes.c_uint32 # enum
amd_exception_kind16_t = ctypes.c_uint16

# values for enumeration 'amd_exception_kind_t'
amd_exception_kind_t__enumvalues = {
    1: 'AMD_EXCEPTION_KIND_INVALID_OPERATION',
    2: 'AMD_EXCEPTION_KIND_DIVISION_BY_ZERO',
    4: 'AMD_EXCEPTION_KIND_OVERFLOW',
    8: 'AMD_EXCEPTION_KIND_UNDERFLOW',
    16: 'AMD_EXCEPTION_KIND_INEXACT',
}
AMD_EXCEPTION_KIND_INVALID_OPERATION = 1
AMD_EXCEPTION_KIND_DIVISION_BY_ZERO = 2
AMD_EXCEPTION_KIND_OVERFLOW = 4
AMD_EXCEPTION_KIND_UNDERFLOW = 8
AMD_EXCEPTION_KIND_INEXACT = 16
amd_exception_kind_t = ctypes.c_uint32 # enum
class struct_amd_control_directives_s(Structure):
    pass

struct_amd_control_directives_s._pack_ = 1 # source:False
struct_amd_control_directives_s._fields_ = [
    ('enabled_control_directives', ctypes.c_uint64),
    ('enable_break_exceptions', ctypes.c_uint16),
    ('enable_detect_exceptions', ctypes.c_uint16),
    ('max_dynamic_group_size', ctypes.c_uint32),
    ('max_flat_grid_size', ctypes.c_uint64),
    ('max_flat_workgroup_size', ctypes.c_uint32),
    ('required_dim', ctypes.c_ubyte),
    ('reserved1', ctypes.c_ubyte * 3),
    ('required_grid_size', ctypes.c_uint64 * 3),
    ('required_workgroup_size', ctypes.c_uint32 * 3),
    ('reserved2', ctypes.c_ubyte * 60),
]

amd_control_directives_t = struct_amd_control_directives_s
class struct_amd_kernel_code_s(Structure):
    pass

struct_amd_kernel_code_s._pack_ = 1 # source:False
struct_amd_kernel_code_s._fields_ = [
    ('amd_kernel_code_version_major', ctypes.c_uint32),
    ('amd_kernel_code_version_minor', ctypes.c_uint32),
    ('amd_machine_kind', ctypes.c_uint16),
    ('amd_machine_version_major', ctypes.c_uint16),
    ('amd_machine_version_minor', ctypes.c_uint16),
    ('amd_machine_version_stepping', ctypes.c_uint16),
    ('kernel_code_entry_byte_offset', ctypes.c_int64),
    ('kernel_code_prefetch_byte_offset', ctypes.c_int64),
    ('kernel_code_prefetch_byte_size', ctypes.c_uint64),
    ('max_scratch_backing_memory_byte_size', ctypes.c_uint64),
    ('compute_pgm_rsrc1', ctypes.c_uint32),
    ('compute_pgm_rsrc2', ctypes.c_uint32),
    ('kernel_code_properties', ctypes.c_uint32),
    ('workitem_private_segment_byte_size', ctypes.c_uint32),
    ('workgroup_group_segment_byte_size', ctypes.c_uint32),
    ('gds_segment_byte_size', ctypes.c_uint32),
    ('kernarg_segment_byte_size', ctypes.c_uint64),
    ('workgroup_fbarrier_count', ctypes.c_uint32),
    ('wavefront_sgpr_count', ctypes.c_uint16),
    ('workitem_vgpr_count', ctypes.c_uint16),
    ('reserved_vgpr_first', ctypes.c_uint16),
    ('reserved_vgpr_count', ctypes.c_uint16),
    ('reserved_sgpr_first', ctypes.c_uint16),
    ('reserved_sgpr_count', ctypes.c_uint16),
    ('debug_wavefront_private_segment_offset_sgpr', ctypes.c_uint16),
    ('debug_private_segment_buffer_sgpr', ctypes.c_uint16),
    ('kernarg_segment_alignment', ctypes.c_ubyte),
    ('group_segment_alignment', ctypes.c_ubyte),
    ('private_segment_alignment', ctypes.c_ubyte),
    ('wavefront_size', ctypes.c_ubyte),
    ('call_convention', ctypes.c_int32),
    ('reserved1', ctypes.c_ubyte * 12),
    ('runtime_loader_kernel_symbol', ctypes.c_uint64),
    ('control_directives', amd_control_directives_t),
]

amd_kernel_code_t = struct_amd_kernel_code_s
class struct_amd_runtime_loader_debug_info_s(Structure):
    pass

struct_amd_runtime_loader_debug_info_s._pack_ = 1 # source:False
struct_amd_runtime_loader_debug_info_s._fields_ = [
    ('elf_raw', ctypes.POINTER(None)),
    ('elf_size', ctypes.c_uint64),
    ('kernel_name', ctypes.POINTER(ctypes.c_char)),
    ('owning_segment', ctypes.POINTER(None)),
]

amd_runtime_loader_debug_info_t = struct_amd_runtime_loader_debug_info_s
__all__ = \
    ['AMDKFD_COMMAND_END', 'AMDKFD_COMMAND_START',
    'AMDKFD_IOCTL_BASE', 'AMD_COMPUTE_PGM_RSRC_ONE_BULKY',
    'AMD_COMPUTE_PGM_RSRC_ONE_BULKY_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_BULKY_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER',
    'AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE',
    'AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP',
    'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE',
    'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT',
    'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT',
    'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY',
    'AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_PRIV',
    'AMD_COMPUTE_PGM_RSRC_ONE_PRIV_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_PRIV_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1',
    'AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE',
    'AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1',
    'AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_WIDTH',
    'AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT',
    'AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT',
    'AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_WIDTH',
    'AMD_CONTROL_DIRECTIVES_ALIGN_BYTES', 'AMD_ELEMENT_BYTE_SIZE_16',
    'AMD_ELEMENT_BYTE_SIZE_2', 'AMD_ELEMENT_BYTE_SIZE_4',
    'AMD_ELEMENT_BYTE_SIZE_8',
    'AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS',
    'AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS',
    'AMD_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE',
    'AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE',
    'AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE',
    'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM',
    'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE',
    'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE',
    'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS',
    'AMD_EXCEPTION_KIND_DIVISION_BY_ZERO',
    'AMD_EXCEPTION_KIND_INEXACT',
    'AMD_EXCEPTION_KIND_INVALID_OPERATION',
    'AMD_EXCEPTION_KIND_OVERFLOW', 'AMD_EXCEPTION_KIND_UNDERFLOW',
    'AMD_FLOAT_DENORM_MODE_FLUSH_OUTPUT',
    'AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE',
    'AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT',
    'AMD_FLOAT_DENORM_MODE_NO_FLUSH',
    'AMD_FLOAT_ROUND_MODE_MINUS_INFINITY',
    'AMD_FLOAT_ROUND_MODE_NEAREST_EVEN',
    'AMD_FLOAT_ROUND_MODE_PLUS_INFINITY', 'AMD_FLOAT_ROUND_MODE_ZERO',
    'AMD_HSA_KERNEL_CODE_H', 'AMD_ISA_ALIGN_BYTES',
    'AMD_KERNEL_CODE_ALIGN_BYTES',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED',
    'AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK',
    'AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_IS_PTR64',
    'AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED',
    'AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE',
    'AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_RESERVED1',
    'AMD_KERNEL_CODE_PROPERTIES_RESERVED1_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_RESERVED1_WIDTH',
    'AMD_KERNEL_CODE_PROPERTIES_RESERVED2',
    'AMD_KERNEL_CODE_PROPERTIES_RESERVED2_SHIFT',
    'AMD_KERNEL_CODE_PROPERTIES_RESERVED2_WIDTH',
    'AMD_KERNEL_CODE_VERSION_MAJOR', 'AMD_KERNEL_CODE_VERSION_MINOR',
    'AMD_MACHINE_KIND_AMDGPU', 'AMD_MACHINE_KIND_UNDEFINED',
    'AMD_POWERTWO_1', 'AMD_POWERTWO_128', 'AMD_POWERTWO_16',
    'AMD_POWERTWO_2', 'AMD_POWERTWO_256', 'AMD_POWERTWO_32',
    'AMD_POWERTWO_4', 'AMD_POWERTWO_64', 'AMD_POWERTWO_8',
    'AMD_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED',
    'AMD_SYSTEM_VGPR_WORKITEM_ID_X',
    'AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y',
    'AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z', 'KFD_HW_EXCEPTION_ECC',
    'KFD_HW_EXCEPTION_GPU_HANG', 'KFD_HW_EXCEPTION_PER_ENGINE_RESET',
    'KFD_HW_EXCEPTION_WHOLE_GPU_RESET', 'KFD_IOCTL_H_INCLUDED',
    'KFD_IOCTL_MAJOR_VERSION', 'KFD_IOCTL_MINOR_VERSION',
    'KFD_IOCTL_SVM_ATTR_ACCESS', 'KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE',
    'KFD_IOCTL_SVM_ATTR_CLR_FLAGS', 'KFD_IOCTL_SVM_ATTR_GRANULARITY',
    'KFD_IOCTL_SVM_ATTR_NO_ACCESS',
    'KFD_IOCTL_SVM_ATTR_PREFERRED_LOC',
    'KFD_IOCTL_SVM_ATTR_PREFETCH_LOC', 'KFD_IOCTL_SVM_ATTR_SET_FLAGS',
    'KFD_IOCTL_SVM_FLAG_COHERENT', 'KFD_IOCTL_SVM_FLAG_GPU_EXEC',
    'KFD_IOCTL_SVM_FLAG_GPU_READ_MOSTLY', 'KFD_IOCTL_SVM_FLAG_GPU_RO',
    'KFD_IOCTL_SVM_FLAG_HIVE_LOCAL', 'KFD_IOCTL_SVM_FLAG_HOST_ACCESS',
    'KFD_IOCTL_SVM_LOCATION_SYSMEM',
    'KFD_IOCTL_SVM_LOCATION_UNDEFINED', 'KFD_IOCTL_SVM_OP_GET_ATTR',
    'KFD_IOCTL_SVM_OP_SET_ATTR',
    'KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM',
    'KFD_IOC_ALLOC_MEM_FLAGS_COHERENT',
    'KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL',
    'KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE',
    'KFD_IOC_ALLOC_MEM_FLAGS_GTT',
    'KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP',
    'KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE',
    'KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC',
    'KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED',
    'KFD_IOC_ALLOC_MEM_FLAGS_USERPTR', 'KFD_IOC_ALLOC_MEM_FLAGS_VRAM',
    'KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE',
    'KFD_IOC_CACHE_POLICY_COHERENT',
    'KFD_IOC_CACHE_POLICY_NONCOHERENT', 'KFD_IOC_EVENT_DEBUG_EVENT',
    'KFD_IOC_EVENT_DEVICESTATECHANGE', 'KFD_IOC_EVENT_HW_EXCEPTION',
    'KFD_IOC_EVENT_MEMORY', 'KFD_IOC_EVENT_NODECHANGE',
    'KFD_IOC_EVENT_PROFILE_EVENT', 'KFD_IOC_EVENT_QUEUE_EVENT',
    'KFD_IOC_EVENT_SIGNAL', 'KFD_IOC_EVENT_SYSTEM_EVENT',
    'KFD_IOC_QUEUE_TYPE_COMPUTE', 'KFD_IOC_QUEUE_TYPE_COMPUTE_AQL',
    'KFD_IOC_QUEUE_TYPE_SDMA', 'KFD_IOC_QUEUE_TYPE_SDMA_XGMI',
    'KFD_IOC_WAIT_RESULT_COMPLETE', 'KFD_IOC_WAIT_RESULT_FAIL',
    'KFD_IOC_WAIT_RESULT_TIMEOUT', 'KFD_MAX_QUEUE_PERCENTAGE',
    'KFD_MAX_QUEUE_PRIORITY', 'KFD_MEM_ERR_GPU_HANG',
    'KFD_MEM_ERR_NO_RAS', 'KFD_MEM_ERR_POISON_CONSUMED',
    'KFD_MEM_ERR_SRAM_ECC', 'KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL',
    'KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL', 'KFD_SIGNAL_EVENT_LIMIT',
    'KFD_SMI_EVENT_GPU_POST_RESET', 'KFD_SMI_EVENT_GPU_PRE_RESET',
    'KFD_SMI_EVENT_NONE', 'KFD_SMI_EVENT_THERMAL_THROTTLE',
    'KFD_SMI_EVENT_VMFAULT', 'MAX_ALLOWED_AW_BUFF_SIZE',
    'MAX_ALLOWED_NUM_POINTS', 'MAX_ALLOWED_WAC_BUFF_SIZE',
    'NUM_OF_SUPPORTED_GPUS', 'amd_compute_pgm_rsrc_one32_t',
    'amd_compute_pgm_rsrc_one_t', 'amd_compute_pgm_rsrc_two32_t',
    'amd_compute_pgm_rsrc_two_t', 'amd_control_directives_t',
    'amd_element_byte_size_t', 'amd_enabled_control_directive64_t',
    'amd_enabled_control_directive_t', 'amd_exception_kind16_t',
    'amd_exception_kind_t', 'amd_float_denorm_mode_t',
    'amd_float_round_mode_t', 'amd_kernel_code_properties32_t',
    'amd_kernel_code_properties_t', 'amd_kernel_code_t',
    'amd_kernel_code_version32_t', 'amd_kernel_code_version_t',
    'amd_machine_kind16_t', 'amd_machine_kind_t',
    'amd_machine_version16_t', 'amd_powertwo8_t', 'amd_powertwo_t',
    'amd_runtime_loader_debug_info_t',
    'amd_system_vgpr_workitem_id_t', 'kfd_ioctl_svm_attr_type',
    'kfd_ioctl_svm_location', 'kfd_ioctl_svm_op', 'kfd_mmio_remap',
    'kfd_smi_event', 'struct_amd_control_directives_s',
    'struct_amd_kernel_code_s',
    'struct_amd_runtime_loader_debug_info_s', 'struct_kfd_event_data',
    'struct_kfd_hsa_hw_exception_data',
    'struct_kfd_hsa_memory_exception_data',
    'struct_kfd_ioctl_acquire_vm_args',
    'struct_kfd_ioctl_alloc_memory_of_gpu_args',
    'struct_kfd_ioctl_alloc_queue_gws_args',
    'struct_kfd_ioctl_create_event_args',
    'struct_kfd_ioctl_create_queue_args',
    'struct_kfd_ioctl_dbg_address_watch_args',
    'struct_kfd_ioctl_dbg_register_args',
    'struct_kfd_ioctl_dbg_unregister_args',
    'struct_kfd_ioctl_dbg_wave_control_args',
    'struct_kfd_ioctl_destroy_event_args',
    'struct_kfd_ioctl_destroy_queue_args',
    'struct_kfd_ioctl_free_memory_of_gpu_args',
    'struct_kfd_ioctl_get_clock_counters_args',
    'struct_kfd_ioctl_get_dmabuf_info_args',
    'struct_kfd_ioctl_get_process_apertures_args',
    'struct_kfd_ioctl_get_process_apertures_new_args',
    'struct_kfd_ioctl_get_queue_wave_state_args',
    'struct_kfd_ioctl_get_tile_config_args',
    'struct_kfd_ioctl_get_version_args',
    'struct_kfd_ioctl_import_dmabuf_args',
    'struct_kfd_ioctl_map_memory_to_gpu_args',
    'struct_kfd_ioctl_reset_event_args',
    'struct_kfd_ioctl_set_cu_mask_args',
    'struct_kfd_ioctl_set_event_args',
    'struct_kfd_ioctl_set_memory_policy_args',
    'struct_kfd_ioctl_set_scratch_backing_va_args',
    'struct_kfd_ioctl_set_trap_handler_args',
    'struct_kfd_ioctl_set_xnack_mode_args',
    'struct_kfd_ioctl_smi_events_args', 'struct_kfd_ioctl_svm_args',
    'struct_kfd_ioctl_svm_attribute',
    'struct_kfd_ioctl_unmap_memory_from_gpu_args',
    'struct_kfd_ioctl_update_queue_args',
    'struct_kfd_ioctl_wait_events_args',
    'struct_kfd_memory_exception_failure',
    'struct_kfd_process_device_apertures', 'union_kfd_event_data_0']
