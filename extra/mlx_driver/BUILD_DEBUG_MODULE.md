# Debug mlx5_core module

Source: `~/mlx_build/drivers/net/ethernet/mellanox/mlx5/core/`
Output: `~/mlx_build/drivers/net/ethernet/mellanox/mlx5/core/mlx5_core.ko`

## Edit, build, load, capture, unload

```bash
# 1. Edit (e.g. add pr_err to cmd.c or main.c)
vim ~/mlx_build/drivers/net/ethernet/mellanox/mlx5/core/cmd.c

# 2. Build
cd ~/mlx_build && make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/ethernet/mellanox/mlx5/core modules

# 3. Reset FW + unload stock module
echo y | sudo mstfwreset -d 41:00.0 reset; sleep 3
sudo rmmod mlx5_ib mlx5_fwctl mlx5_core 2>/dev/null; sleep 1

# 4. Load debug module + capture
sudo dmesg -C
sudo insmod ~/mlx_build/drivers/net/ethernet/mellanox/mlx5/core/mlx5_core.ko
sleep 15
sudo dmesg > ~/tinygrad/extra/mlx_driver/kernel_trace_full.txt

# 5. Unload
sudo rmmod mlx5_core
```

## Tips
- Add `return -ENODEV;` in `main.c` after INIT_HCA to stop early (prevents networking init flood + D-state hangs)
- Use `pr_err()` not `mlx5_core_dbg()` — dbg requires dyndbg which isn't active at boot
- Never `make modules_install` — only `insmod` from build path
- If module stuck in D-state: reboot
- If FW crashes (synd=0x01): `echo y | sudo mstfwreset -d 41:00.0 reset`
