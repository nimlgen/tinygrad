import re, ctypes

from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_mp_13_0_0, amdgpu_nbio_4_3_0, amdgpu_discovery, amdgpu_mmhub_3_0_0, amdgpu_gc_11_0_0, amdgpu_osssys_6_0_0
from tinygrad.runtime.support.am.amdev import AMDev

REGISTER_NAMES = {

}

def parse_amdgpu_logs(log_content, register_names=None):
    """
    Parse AMDGPU logs and replace register numbers with their names.
    
    :param log_content: String containing the log content
    :param register_names: Optional dictionary of register addresses to names
    :return: Processed log content
    """
    # Use the provided register names or default to REGISTER_NAMES
    register_map = register_names or REGISTER_NAMES
    
    final = ""
    def replace_register(match):
        register = match.group(1)
        return f"Reading register {register_map.get(int(register, base=16), register)}"

    # Regex pattern to match "Reading register 0x1a700" type lines
    pattern = r'Reading register (0x[0-9a-fA-F]+)'
    
    # Replace register numbers with names
    processed_log = re.sub(pattern, replace_register, log_content)

    def replace_register_2(match):
        register = match.group(1)
        return f"Writing register {register_map.get(int(register, base=16), register)}"

    # Regex pattern to match "Reading register 0x1a700" type lines
    pattern = r'Writing register (0x[0-9a-fA-F]+)'

    # Replace register numbers with names
    processed_log = re.sub(pattern, replace_register_2, processed_log)

    lines = processed_log.split('\n')
    
    # Remove trace sections
    in_trace = False
    cleaned_lines = []
    
    for line in lines:
        if "cut here" in line:
            in_trace = True
            continue
        if "end trace" in line:
            in_trace = False
            continue
        if not in_trace:
            # Remove timestamp and amdgpu prefix
            cleaned_line = re.sub(r'^\[\d+\.\d+\]\s*(?:amdgpu\s+[^:]+:\s*amdgpu:\s*)?', '', line)
            if cleaned_line.strip():  # Only add non-empty lines
                cleaned_lines.append(cleaned_line)

    return '\n'.join(cleaned_lines)

def main():
    def check(x): assert x == 0

    check(libpciaccess.pci_system_init())

    pci_iter = libpciaccess.pci_id_match_iterator_create(None)
    print(pci_iter)

    pcidev = None
    while True:
        pcidev = libpciaccess.pci_device_next(pci_iter)
        if not pcidev: break
        dev_fmt = "{:04x}:{:02x}:{:02x}.{:d}".format(pcidev.contents.domain_16, pcidev.contents.bus, pcidev.contents.dev, pcidev.contents.func)
        print(dev_fmt, hex(pcidev.contents.vendor_id), hex(pcidev.contents.device_id))
        
        if pcidev.contents.vendor_id == 0x1002 and pcidev.contents.device_id == 0x744c:
            dev_fmt = "{:04x}:{:02x}:{:02x}.{:d}".format(pcidev.contents.domain_16, pcidev.contents.bus, pcidev.contents.dev, pcidev.contents.func)
            # if dev_fmt == "0000:03:00.0": continue # skip it, use for kernel hacking.
            # if dev_fmt == "0000:86:00.0": continue # skip it, use for kernel hacking.
            # if dev_fmt == "0000:c6:00.0": continue # skip it, use for kernel hacking.
            # if dev_fmt == "0000:44:00.0": continue # skip it, use for kernel hacking.
            # if dev_fmt == "0000:83:00.0": continue # skip it, use for kernel hacking.
            # if dev_fmt == "0000:c3:00.0": continue # skip it, use for kernel hacking.
            # print(dev_fmt)
            # exit(0)
            break

    assert pcidev is not None
    pcidev = pcidev.contents

    libpciaccess.pci_device_probe(ctypes.byref(pcidev))

    adev = AMDev(pcidev)

    def _prepare_registers(modules):
        for base, m in modules:
            for k, regval in m.__dict__.items():
                if k.startswith("reg") and not k.endswith("_BASE_IDX") and (base_idx:=getattr(m, f"{k}_BASE_IDX", None)) is not None:
                    REGISTER_NAMES[adev.reg_off(base, 0, regval, base_idx)] = k

    _prepare_registers([("MP0", amdgpu_mp_13_0_0), ("NBIO", amdgpu_nbio_4_3_0), ("MMHUB", amdgpu_mmhub_3_0_0), ("GC", amdgpu_gc_11_0_0), ("OSSSYS", amdgpu_osssys_6_0_0)])

    with open('/home/nimlgen/tinygrad/z.z', 'r') as f:
        log_content = f.read()

    # with open('/home/nimlgen/amdgpu_ubuntu_22_04/logs.txt', 'r') as f:
    #     log_content = f.read()

    # Process the log content
    processed_log = parse_amdgpu_logs(log_content)

    # Print the processed log
    # print(processed_log)

    # Optionally, write to a file
    with open('/home/nimlgen/amdgpu_ubuntu_22_04/x5.logs', 'w') as f:
        f.write(processed_log)

if __name__ == '__main__':
    main()