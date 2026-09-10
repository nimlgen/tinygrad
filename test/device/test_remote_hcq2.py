import contextlib, os, select, socket, subprocess, sys, tempfile, textwrap, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

class TestRemoteHCQ2(unittest.TestCase):
  def run_remote(self, code:str, nodes:int=1):
    env = os.environ | {"PYTHONPATH": str(ROOT), "REMOTE": "", "DEV": "MOCKPCI+AMD", "DEBUG": "0", "RDMA": "0"}
    peers = []
    with contextlib.ExitStack() as stack:
      for _ in range(nodes):
        node_env = env | {"TMPDIR": stack.enter_context(tempfile.TemporaryDirectory())}
        with socket.socket() as sock:
          sock.bind(("127.0.0.1", 0))
          port = sock.getsockname()[1]
        server = stack.enter_context(subprocess.Popen([sys.executable, "-u", "extra/remote/serve.py", str(port)], cwd=ROOT, env=node_env,
                                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
        stack.callback(server.terminate)
        self.assertTrue(select.select([server.stdout], [], [], 30)[0], "remote server did not start")
        self.assertEqual(server.stdout.readline().decode().strip(), f"listening on {port}")
        peers.append(f"127.0.0.1:{port}")
      code = "from tinygrad.runtime.support import hcq2\nhcq2.STAGING_SIZE = 4096\n" + textwrap.dedent(code)
      result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                              env=env | {"DEV": "PCI+AMD", "REMOTE": ",".join(peers)}, capture_output=True, text=True, timeout=240)
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_kernel_chain_and_jit(self):
    self.run_remote('''
      import numpy as np
      from tinygrad import Device, Tensor, TinyJit
      from tinygrad.uop.ops import UOp
      assert Device["AMD"].remote_peer is not None
      x = Tensor(np.arange(32, dtype=np.int32)).to("AMD").realize()
      np.testing.assert_equal(((x + 1).contiguous() * 2).numpy(), (np.arange(32) + 1) * 2)
      np.testing.assert_equal((x + UOp.variable("offset", -8, 8).bind(-3)).numpy(), np.arange(32) - 3)
      @TinyJit
      def f(x): return ((x + 1).contiguous() * 3).contiguous().realize()
      for i in range(5):
        x = Tensor(np.arange(32, dtype=np.int32) + i).to("AMD").realize()
        np.testing.assert_equal(f(x).numpy(), (np.arange(32) + i + 1) * 3)
    ''')

  def test_peer_batches(self):
    self.run_remote('''
      from tinygrad import Device, Tensor, TinyJit
      from test.helpers import call_is_hcq
      assert Device["AMD"].peer_group != Device["AMD:1"].peer_group
      @TinyJit
      def f(a, b):
        a, b = (a + 1).contiguous(), (b * 2).contiguous()
        a.realize(b)
        return a, b
      for i in range(5):
        a = Tensor.full((8,), i, device="AMD").contiguous().realize()
        b = Tensor.full((8,), i + 1, device="AMD:1").contiguous().realize()
        a, b = f(a, b)
        assert a.tolist() == [i + 1] * 8
        assert b.tolist() == [(i + 1) * 2] * 8
      calls = [c.without_after for c in f.captured.linear.src if call_is_hcq(c)]
      assert len(calls) == 2, calls
      assert {c.arg.aux.device for c in calls} == {("AMD",), ("AMD:1",)}
    ''', nodes=2)

  def test_peer_copy_staging(self):
    self.run_remote('''
      import numpy as np
      from tinygrad import Tensor
      from unittest.mock import patch
      from tinygrad.runtime.support import hcq2
      # A short staging buffer exercises multiple chunks without a large mock transfer.
      with patch.object(hcq2, "STAGING_SIZE", 64):
        x = Tensor(np.arange(65, dtype=np.int32)).to("AMD").realize()
        np.testing.assert_equal(x.to("AMD:1").numpy(), np.arange(65))
    ''', nodes=2)

  def test_bad_posted_exec_disconnects(self):
    self.run_remote('''
      import os, socket
      from tinygrad.runtime.support.system import RemotePCIDevice, RemoteCmd
      host, port = os.environ["REMOTE"].split(":")
      with socket.create_connection((host, int(port)), timeout=5) as sock:
        RemotePCIDevice._post(sock, RemoteCmd.EXEC_PROG, 0, 0, 0)
        assert sock.recv(1) == b"", "a failed posted command must close the connection"
    ''')

if __name__ == "__main__": unittest.main()
