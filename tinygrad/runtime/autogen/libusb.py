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



_libraries = {}
_libraries['libusb-0.1.so.4.4.4'] = ctypes.CDLL('/lib/x86_64-linux-gnu/libusb-0.1.so.4.4.4')
c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16



__USB_H__ = True # macro
USB_CLASS_PER_INTERFACE = 0 # macro
USB_CLASS_AUDIO = 1 # macro
USB_CLASS_COMM = 2 # macro
USB_CLASS_HID = 3 # macro
USB_CLASS_PRINTER = 7 # macro
USB_CLASS_PTP = 6 # macro
USB_CLASS_MASS_STORAGE = 8 # macro
USB_CLASS_HUB = 9 # macro
USB_CLASS_DATA = 10 # macro
USB_CLASS_VENDOR_SPEC = 0xff # macro
USB_DT_DEVICE = 0x01 # macro
USB_DT_CONFIG = 0x02 # macro
USB_DT_STRING = 0x03 # macro
USB_DT_INTERFACE = 0x04 # macro
USB_DT_ENDPOINT = 0x05 # macro
USB_DT_HID = 0x21 # macro
USB_DT_REPORT = 0x22 # macro
USB_DT_PHYSICAL = 0x23 # macro
USB_DT_HUB = 0x29 # macro
USB_DT_DEVICE_SIZE = 18 # macro
USB_DT_CONFIG_SIZE = 9 # macro
USB_DT_INTERFACE_SIZE = 9 # macro
USB_DT_ENDPOINT_SIZE = 7 # macro
USB_DT_ENDPOINT_AUDIO_SIZE = 9 # macro
USB_DT_HUB_NONVAR_SIZE = 7 # macro
USB_MAXENDPOINTS = 32 # macro
USB_ENDPOINT_ADDRESS_MASK = 0x0f # macro
USB_ENDPOINT_DIR_MASK = 0x80 # macro
USB_ENDPOINT_TYPE_MASK = 0x03 # macro
USB_ENDPOINT_TYPE_CONTROL = 0 # macro
USB_ENDPOINT_TYPE_ISOCHRONOUS = 1 # macro
USB_ENDPOINT_TYPE_BULK = 2 # macro
USB_ENDPOINT_TYPE_INTERRUPT = 3 # macro
USB_MAXINTERFACES = 32 # macro
USB_MAXALTSETTING = 128 # macro
USB_MAXCONFIG = 8 # macro
USB_REQ_GET_STATUS = 0x00 # macro
USB_REQ_CLEAR_FEATURE = 0x01 # macro
USB_REQ_SET_FEATURE = 0x03 # macro
USB_REQ_SET_ADDRESS = 0x05 # macro
USB_REQ_GET_DESCRIPTOR = 0x06 # macro
USB_REQ_SET_DESCRIPTOR = 0x07 # macro
USB_REQ_GET_CONFIGURATION = 0x08 # macro
USB_REQ_SET_CONFIGURATION = 0x09 # macro
USB_REQ_GET_INTERFACE = 0x0A # macro
USB_REQ_SET_INTERFACE = 0x0B # macro
USB_REQ_SYNCH_FRAME = 0x0C # macro
USB_TYPE_STANDARD = (0x00<<5) # macro
USB_TYPE_CLASS = (0x01<<5) # macro
USB_TYPE_VENDOR = (0x02<<5) # macro
USB_TYPE_RESERVED = (0x03<<5) # macro
USB_RECIP_DEVICE = 0x00 # macro
USB_RECIP_INTERFACE = 0x01 # macro
USB_RECIP_ENDPOINT = 0x02 # macro
USB_RECIP_OTHER = 0x03 # macro
USB_ENDPOINT_IN = 0x80 # macro
USB_ENDPOINT_OUT = 0x00 # macro
USB_ERROR_BEGIN = 500000 # macro
# USB_LE16_TO_CPU = (x) # macro
# LIBUSB_PATH_MAX = PATH_MAX # macro
LIBUSB_HAS_GET_DRIVER_NP = 1 # macro
LIBUSB_HAS_DETACH_KERNEL_DRIVER_NP = 1 # macro
class struct_usb_descriptor_header(Structure):
    pass

struct_usb_descriptor_header._pack_ = 1 # source:True
struct_usb_descriptor_header._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
]

class struct_usb_string_descriptor(Structure):
    pass

struct_usb_string_descriptor._pack_ = 1 # source:True
struct_usb_string_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('wData', ctypes.c_uint16 * 1),
]

class struct_usb_hid_descriptor(Structure):
    pass

struct_usb_hid_descriptor._pack_ = 1 # source:True
struct_usb_hid_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bcdHID', ctypes.c_uint16),
    ('bCountryCode', ctypes.c_ubyte),
    ('bNumDescriptors', ctypes.c_ubyte),
]

class struct_usb_endpoint_descriptor(Structure):
    pass

struct_usb_endpoint_descriptor._pack_ = 1 # source:False
struct_usb_endpoint_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bEndpointAddress', ctypes.c_ubyte),
    ('bmAttributes', ctypes.c_ubyte),
    ('wMaxPacketSize', ctypes.c_uint16),
    ('bInterval', ctypes.c_ubyte),
    ('bRefresh', ctypes.c_ubyte),
    ('bSynchAddress', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('extra', ctypes.POINTER(ctypes.c_ubyte)),
    ('extralen', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_usb_interface_descriptor(Structure):
    pass

struct_usb_interface_descriptor._pack_ = 1 # source:False
struct_usb_interface_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bInterfaceNumber', ctypes.c_ubyte),
    ('bAlternateSetting', ctypes.c_ubyte),
    ('bNumEndpoints', ctypes.c_ubyte),
    ('bInterfaceClass', ctypes.c_ubyte),
    ('bInterfaceSubClass', ctypes.c_ubyte),
    ('bInterfaceProtocol', ctypes.c_ubyte),
    ('iInterface', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('endpoint', ctypes.POINTER(struct_usb_endpoint_descriptor)),
    ('extra', ctypes.POINTER(ctypes.c_ubyte)),
    ('extralen', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_usb_interface(Structure):
    pass

struct_usb_interface._pack_ = 1 # source:False
struct_usb_interface._fields_ = [
    ('altsetting', ctypes.POINTER(struct_usb_interface_descriptor)),
    ('num_altsetting', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_usb_config_descriptor(Structure):
    pass

struct_usb_config_descriptor._pack_ = 1 # source:False
struct_usb_config_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('wTotalLength', ctypes.c_uint16),
    ('bNumInterfaces', ctypes.c_ubyte),
    ('bConfigurationValue', ctypes.c_ubyte),
    ('iConfiguration', ctypes.c_ubyte),
    ('bmAttributes', ctypes.c_ubyte),
    ('MaxPower', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('interface', ctypes.POINTER(struct_usb_interface)),
    ('extra', ctypes.POINTER(ctypes.c_ubyte)),
    ('extralen', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_usb_device_descriptor(Structure):
    pass

struct_usb_device_descriptor._pack_ = 1 # source:True
struct_usb_device_descriptor._fields_ = [
    ('bLength', ctypes.c_ubyte),
    ('bDescriptorType', ctypes.c_ubyte),
    ('bcdUSB', ctypes.c_uint16),
    ('bDeviceClass', ctypes.c_ubyte),
    ('bDeviceSubClass', ctypes.c_ubyte),
    ('bDeviceProtocol', ctypes.c_ubyte),
    ('bMaxPacketSize0', ctypes.c_ubyte),
    ('idVendor', ctypes.c_uint16),
    ('idProduct', ctypes.c_uint16),
    ('bcdDevice', ctypes.c_uint16),
    ('iManufacturer', ctypes.c_ubyte),
    ('iProduct', ctypes.c_ubyte),
    ('iSerialNumber', ctypes.c_ubyte),
    ('bNumConfigurations', ctypes.c_ubyte),
]

class struct_usb_ctrl_setup(Structure):
    pass

struct_usb_ctrl_setup._pack_ = 1 # source:True
struct_usb_ctrl_setup._fields_ = [
    ('bRequestType', ctypes.c_ubyte),
    ('bRequest', ctypes.c_ubyte),
    ('wValue', ctypes.c_uint16),
    ('wIndex', ctypes.c_uint16),
    ('wLength', ctypes.c_uint16),
]

class struct_usb_dev_handle(Structure):
    pass

usb_dev_handle = struct_usb_dev_handle
class struct_usb_bus(Structure):
    pass

class struct_usb_device(Structure):
    pass

struct_usb_bus._pack_ = 1 # source:False
struct_usb_bus._fields_ = [
    ('next', ctypes.POINTER(struct_usb_bus)),
    ('prev', ctypes.POINTER(struct_usb_bus)),
    ('dirname', ctypes.c_char * 4097),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('devices', ctypes.POINTER(struct_usb_device)),
    ('location', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('root_dev', ctypes.POINTER(struct_usb_device)),
]

usb_busses = ctypes.POINTER(struct_usb_bus)() # Variable ctypes.POINTER(struct_usb_bus)
try:
    usb_open = _libraries['libusb-0.1.so.4.4.4'].usb_open
    usb_open.restype = ctypes.POINTER(struct_usb_dev_handle)
    usb_open.argtypes = [ctypes.POINTER(struct_usb_device)]
except AttributeError:
    pass
try:
    usb_close = _libraries['libusb-0.1.so.4.4.4'].usb_close
    usb_close.restype = ctypes.c_int32
    usb_close.argtypes = [ctypes.POINTER(struct_usb_dev_handle)]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    usb_get_string = _libraries['libusb-0.1.so.4.4.4'].usb_get_string
    usb_get_string.restype = ctypes.c_int32
    usb_get_string.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    usb_get_string_simple = _libraries['libusb-0.1.so.4.4.4'].usb_get_string_simple
    usb_get_string_simple.restype = ctypes.c_int32
    usb_get_string_simple.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    usb_get_descriptor_by_endpoint = _libraries['libusb-0.1.so.4.4.4'].usb_get_descriptor_by_endpoint
    usb_get_descriptor_by_endpoint.restype = ctypes.c_int32
    usb_get_descriptor_by_endpoint.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.POINTER(None), ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_get_descriptor = _libraries['libusb-0.1.so.4.4.4'].usb_get_descriptor
    usb_get_descriptor.restype = ctypes.c_int32
    usb_get_descriptor.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_ubyte, ctypes.c_ubyte, ctypes.POINTER(None), ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_bulk_write = _libraries['libusb-0.1.so.4.4.4'].usb_bulk_write
    usb_bulk_write.restype = ctypes.c_int32
    usb_bulk_write.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_bulk_read = _libraries['libusb-0.1.so.4.4.4'].usb_bulk_read
    usb_bulk_read.restype = ctypes.c_int32
    usb_bulk_read.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_interrupt_write = _libraries['libusb-0.1.so.4.4.4'].usb_interrupt_write
    usb_interrupt_write.restype = ctypes.c_int32
    usb_interrupt_write.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_interrupt_read = _libraries['libusb-0.1.so.4.4.4'].usb_interrupt_read
    usb_interrupt_read.restype = ctypes.c_int32
    usb_interrupt_read.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_control_msg = _libraries['libusb-0.1.so.4.4.4'].usb_control_msg
    usb_control_msg.restype = ctypes.c_int32
    usb_control_msg.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_set_configuration = _libraries['libusb-0.1.so.4.4.4'].usb_set_configuration
    usb_set_configuration.restype = ctypes.c_int32
    usb_set_configuration.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_claim_interface = _libraries['libusb-0.1.so.4.4.4'].usb_claim_interface
    usb_claim_interface.restype = ctypes.c_int32
    usb_claim_interface.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_release_interface = _libraries['libusb-0.1.so.4.4.4'].usb_release_interface
    usb_release_interface.restype = ctypes.c_int32
    usb_release_interface.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_set_altinterface = _libraries['libusb-0.1.so.4.4.4'].usb_set_altinterface
    usb_set_altinterface.restype = ctypes.c_int32
    usb_set_altinterface.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_resetep = _libraries['libusb-0.1.so.4.4.4'].usb_resetep
    usb_resetep.restype = ctypes.c_int32
    usb_resetep.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_uint32]
except AttributeError:
    pass
try:
    usb_clear_halt = _libraries['libusb-0.1.so.4.4.4'].usb_clear_halt
    usb_clear_halt.restype = ctypes.c_int32
    usb_clear_halt.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_uint32]
except AttributeError:
    pass
try:
    usb_reset = _libraries['libusb-0.1.so.4.4.4'].usb_reset
    usb_reset.restype = ctypes.c_int32
    usb_reset.argtypes = [ctypes.POINTER(struct_usb_dev_handle)]
except AttributeError:
    pass
try:
    usb_get_driver_np = _libraries['libusb-0.1.so.4.4.4'].usb_get_driver_np
    usb_get_driver_np.restype = ctypes.c_int32
    usb_get_driver_np.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    usb_detach_kernel_driver_np = _libraries['libusb-0.1.so.4.4.4'].usb_detach_kernel_driver_np
    usb_detach_kernel_driver_np.restype = ctypes.c_int32
    usb_detach_kernel_driver_np.argtypes = [ctypes.POINTER(struct_usb_dev_handle), ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_strerror = _libraries['libusb-0.1.so.4.4.4'].usb_strerror
    usb_strerror.restype = ctypes.POINTER(ctypes.c_char)
    usb_strerror.argtypes = []
except AttributeError:
    pass
try:
    usb_init = _libraries['libusb-0.1.so.4.4.4'].usb_init
    usb_init.restype = None
    usb_init.argtypes = []
except AttributeError:
    pass
try:
    usb_set_debug = _libraries['libusb-0.1.so.4.4.4'].usb_set_debug
    usb_set_debug.restype = None
    usb_set_debug.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    usb_find_busses = _libraries['libusb-0.1.so.4.4.4'].usb_find_busses
    usb_find_busses.restype = ctypes.c_int32
    usb_find_busses.argtypes = []
except AttributeError:
    pass
try:
    usb_find_devices = _libraries['libusb-0.1.so.4.4.4'].usb_find_devices
    usb_find_devices.restype = ctypes.c_int32
    usb_find_devices.argtypes = []
except AttributeError:
    pass
try:
    usb_device = _libraries['libusb-0.1.so.4.4.4'].usb_device
    usb_device.restype = ctypes.POINTER(struct_usb_device)
    usb_device.argtypes = [ctypes.POINTER(struct_usb_dev_handle)]
except AttributeError:
    pass
try:
    usb_get_busses = _libraries['libusb-0.1.so.4.4.4'].usb_get_busses
    usb_get_busses.restype = ctypes.POINTER(struct_usb_bus)
    usb_get_busses.argtypes = []
except AttributeError:
    pass
struct_usb_device._pack_ = 1 # source:False
struct_usb_device._fields_ = [
    ('next', ctypes.POINTER(struct_usb_device)),
    ('prev', ctypes.POINTER(struct_usb_device)),
    ('filename', ctypes.c_char * 4097),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('bus', ctypes.POINTER(struct_usb_bus)),
    ('descriptor', struct_usb_device_descriptor),
    ('PADDING_1', ctypes.c_ubyte * 6),
    ('config', ctypes.POINTER(struct_usb_config_descriptor)),
    ('dev', ctypes.POINTER(None)),
    ('devnum', ctypes.c_ubyte),
    ('num_children', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte * 6),
    ('children', ctypes.POINTER(ctypes.POINTER(struct_usb_device))),
]

__all__ = \
    ['LIBUSB_HAS_DETACH_KERNEL_DRIVER_NP', 'LIBUSB_HAS_GET_DRIVER_NP',
    'USB_CLASS_AUDIO', 'USB_CLASS_COMM', 'USB_CLASS_DATA',
    'USB_CLASS_HID', 'USB_CLASS_HUB', 'USB_CLASS_MASS_STORAGE',
    'USB_CLASS_PER_INTERFACE', 'USB_CLASS_PRINTER', 'USB_CLASS_PTP',
    'USB_CLASS_VENDOR_SPEC', 'USB_DT_CONFIG', 'USB_DT_CONFIG_SIZE',
    'USB_DT_DEVICE', 'USB_DT_DEVICE_SIZE', 'USB_DT_ENDPOINT',
    'USB_DT_ENDPOINT_AUDIO_SIZE', 'USB_DT_ENDPOINT_SIZE',
    'USB_DT_HID', 'USB_DT_HUB', 'USB_DT_HUB_NONVAR_SIZE',
    'USB_DT_INTERFACE', 'USB_DT_INTERFACE_SIZE', 'USB_DT_PHYSICAL',
    'USB_DT_REPORT', 'USB_DT_STRING', 'USB_ENDPOINT_ADDRESS_MASK',
    'USB_ENDPOINT_DIR_MASK', 'USB_ENDPOINT_IN', 'USB_ENDPOINT_OUT',
    'USB_ENDPOINT_TYPE_BULK', 'USB_ENDPOINT_TYPE_CONTROL',
    'USB_ENDPOINT_TYPE_INTERRUPT', 'USB_ENDPOINT_TYPE_ISOCHRONOUS',
    'USB_ENDPOINT_TYPE_MASK', 'USB_ERROR_BEGIN', 'USB_MAXALTSETTING',
    'USB_MAXCONFIG', 'USB_MAXENDPOINTS', 'USB_MAXINTERFACES',
    'USB_RECIP_DEVICE', 'USB_RECIP_ENDPOINT', 'USB_RECIP_INTERFACE',
    'USB_RECIP_OTHER', 'USB_REQ_CLEAR_FEATURE',
    'USB_REQ_GET_CONFIGURATION', 'USB_REQ_GET_DESCRIPTOR',
    'USB_REQ_GET_INTERFACE', 'USB_REQ_GET_STATUS',
    'USB_REQ_SET_ADDRESS', 'USB_REQ_SET_CONFIGURATION',
    'USB_REQ_SET_DESCRIPTOR', 'USB_REQ_SET_FEATURE',
    'USB_REQ_SET_INTERFACE', 'USB_REQ_SYNCH_FRAME', 'USB_TYPE_CLASS',
    'USB_TYPE_RESERVED', 'USB_TYPE_STANDARD', 'USB_TYPE_VENDOR',
    '__USB_H__', 'size_t', 'struct_usb_bus',
    'struct_usb_config_descriptor', 'struct_usb_ctrl_setup',
    'struct_usb_descriptor_header', 'struct_usb_dev_handle',
    'struct_usb_device', 'struct_usb_device_descriptor',
    'struct_usb_endpoint_descriptor', 'struct_usb_hid_descriptor',
    'struct_usb_interface', 'struct_usb_interface_descriptor',
    'struct_usb_string_descriptor', 'usb_bulk_read', 'usb_bulk_write',
    'usb_busses', 'usb_claim_interface', 'usb_clear_halt',
    'usb_close', 'usb_control_msg', 'usb_detach_kernel_driver_np',
    'usb_dev_handle', 'usb_device', 'usb_find_busses',
    'usb_find_devices', 'usb_get_busses', 'usb_get_descriptor',
    'usb_get_descriptor_by_endpoint', 'usb_get_driver_np',
    'usb_get_string', 'usb_get_string_simple', 'usb_init',
    'usb_interrupt_read', 'usb_interrupt_write', 'usb_open',
    'usb_release_interface', 'usb_reset', 'usb_resetep',
    'usb_set_altinterface', 'usb_set_configuration', 'usb_set_debug',
    'usb_strerror']
