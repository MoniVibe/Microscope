import subprocess
import sys


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    try:
        import torch  # type: ignore

        torch_ver = torch.__version__
        cuda_ok = torch.cuda.is_available()
        ndev = torch.cuda.device_count() if cuda_ok else 0
    except Exception:
        torch_ver = "not installed"
        cuda_ok = False
        ndev = 0
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch_ver}")
    print(f"cuda_available={cuda_ok}")
    print(f"cuda_devices={ndev}")
    print(f"commit={git_commit_hash()}")


if __name__ == "__main__":
    main()
